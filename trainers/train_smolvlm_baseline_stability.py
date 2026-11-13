#!/usr/bin/env python3
"""Fine-tune the SmolVLM-500M EarthVQA baseline with a reproducible protocol.

This script consumes the immutable QA-index manifest produced by
scripts/prepare_earthvqa_samples.py. It supports a one-update dry run,
gradient accumulation, gradient clipping, deterministic epoch shuffling,
periodic resumable checkpoints, and answer-only supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import transformers
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from ordered_preprocessing import (
    build_worker_processors,
    compute_sample_order_sha256,
    ordered_encoded_batches,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "EarthVQA"
ANNOTATION_PATH = DATA_ROOT / "Train_QA.json"
IMAGE_DIR = DATA_ROOT / "Train" / "images_png"
DEFAULT_MODEL = PROJECT_ROOT / "pretrained" / "SmolVLM-500M-Instruct"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifests"
    / "earthvqa_baseline_30k_seed42.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "baseline_earthvqa_seed42"
PROMPT_PREFIX = (
    "Answer the question about this remote-sensing image with only the "
    "concise answer.\nQuestion: "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--train-scope",
        choices=("connector_lm_head", "text_connector", "full"),
        default="text_connector",
        help=(
            "text_connector freezes the vision encoder while training the "
            "text model, multimodal connector, and LM head."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Optimizer updates between checkpoint-last writes; <=0 disables.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Stop after this many optimizer updates; <=0 runs all epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess-workers", type=int, default=12)
    parser.add_argument("--preprocess-prefetch", type=int, default=24)
    parser.add_argument("--cpu-torch-threads", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="A checkpoint-last directory produced by this script.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write model checkpoints; useful for a one-update smoke run.",
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **serializable_args(args),
        "initialization": "original pretrained checkpoint; resume_from=null",
        "train_sample_order_sha256": getattr(args, "train_sample_order_sha256", None),
        "data_loader": {
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": None,
            "persistent_workers": False,
            "ordered_encode_thread_workers": args.preprocess_workers,
            "ordered_encode_max_inflight": args.preprocess_prefetch,
            "ordered_delivery": True,
            "independent_processor_per_thread": args.preprocess_workers > 0,
            "cpu_torch_threads": torch.get_num_threads(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarthVQADataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        annotations: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.records = records
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_id = str(record["image_id"])
        qa_index = int(record["qa_index"])
        qa = self.annotations[image_id][qa_index]
        question = str(qa["Question"]).strip()
        answer = str(qa["Answer"]).strip()
        original_type = str(qa.get("Type", "MISSING"))
        if original_type != str(record["type"]):
            raise RuntimeError(
                f"Manifest/source type mismatch for {image_id}[{qa_index}]"
            )
        image_path = IMAGE_DIR / Path(image_id).name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        return {
            "image_id": image_id,
            "qa_index": qa_index,
            "type": original_type,
            "question": question,
            "answer": answer,
            "image_path": image_path,
        }


def single_item_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise RuntimeError(
            "Current exact masking path requires --batch-size 1; "
            "use gradient accumulation for a larger effective batch."
        )
    return batch[0]


def configure_trainable(
    model: Idefics3ForConditionalGeneration,
    scope: str,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    trainable_parameters: list[torch.nn.Parameter] = []
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if scope == "full":
            trainable = True
        elif scope == "text_connector":
            trainable = (
                "text_model" in name
                or "connector" in name
                or name.startswith("lm_head")
            )
        else:
            trainable = "connector" in name or name.startswith("lm_head")
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_parameters.append(parameter)
            trainable_names.append(name)
    if not trainable_parameters:
        raise RuntimeError(f"No parameters selected for train scope: {scope}")
    return trainable_parameters, trainable_names


def encode_example(
    processor: AutoProcessor,
    example: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, int]:
    with Image.open(example["image_path"]) as source_image:
        image = source_image.convert("RGB")

    user_message = {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": PROMPT_PREFIX + example["question"],
            },
        ],
    }
    prompt_messages = [user_message]
    full_messages = [
        user_message,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["answer"]}],
        },
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    prompt_inputs = processor(
        text=prompt_text,
        images=[image],
        return_tensors="pt",
    )
    full_inputs = processor(
        text=full_text,
        images=[image],
        return_tensors="pt",
    )

    prompt_ids = prompt_inputs["input_ids"]
    full_ids = full_inputs["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    full_length = int(full_ids.shape[1])
    if full_length <= prompt_length:
        raise RuntimeError(
            f"No answer tokens for {example['image_id']}: "
            f"prompt={prompt_length}, full={full_length}"
        )
    if not torch.equal(full_ids[:, :prompt_length], prompt_ids):
        raise RuntimeError(
            f"Prompt is not a prefix for {example['image_id']} "
            f"QA index {example['qa_index']}"
        )

    labels = full_ids.clone()
    labels[:, :prompt_length] = -100
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    supervised_tokens = int((labels != -100).sum().item())
    if supervised_tokens < 1:
        raise RuntimeError("Answer-only mask contains no supervised tokens")
    return dict(full_inputs), labels, prompt_length, supervised_tokens


def move_tensors(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def save_checkpoint(
    checkpoint_dir: Path,
    model: Idefics3ForConditionalGeneration,
    processor: AutoProcessor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    next_batch_index: int,
    global_update: int,
    examples_seen: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    processor.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "global_update": global_update,
            "examples_seen": examples_seen,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    (checkpoint_dir / "run_config.json").write_text(
        json.dumps(run_metadata(args), indent=2) + "\n",
        encoding="utf-8",
    )


def load_resume_state(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, int]:
    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return {
        "epoch": int(state["epoch"]),
        "next_batch_index": int(state["next_batch_index"]),
        "global_update": int(state["global_update"]),
        "examples_seen": int(state["examples_seen"]),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("--batch-size must currently be 1")
    if args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.preprocess_workers < 0:
        raise ValueError("--preprocess-workers must be >= 0")
    if args.preprocess_workers == 0 and args.preprocess_prefetch != 0:
        raise ValueError("--preprocess-prefetch must be 0 for serial preprocessing")
    if args.preprocess_workers > 0 and args.preprocess_prefetch < args.preprocess_workers:
        raise ValueError("--preprocess-prefetch must be >= --preprocess-workers")
    if args.cpu_torch_threads < 0:
        raise ValueError("--cpu-torch-threads must be >= 0")
    for path in (args.model_dir, args.manifest, ANNOTATION_PATH, IMAGE_DIR):
        if not path.exists():
            raise FileNotFoundError(path)

    set_seed(args.seed)
    if args.cpu_torch_threads > 0:
        torch.set_num_threads(args.cpu_torch_threads)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_hash = manifest.get("source_annotation_sha256")
    actual_hash = sha256_path(ANNOTATION_PATH)
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError(
            "Train_QA.json hash differs from the manifest: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    train_records = list(manifest["splits"]["train"])
    if not train_records:
        raise RuntimeError("Training manifest is empty")
    annotations = json.loads(ANNOTATION_PATH.read_text(encoding="utf-8"))
    dataset = EarthVQADataset(train_records, annotations)
    order_hash, order_count = compute_sample_order_sha256(dataset, args.seed, single_item_collate)
    if order_count != len(dataset):
        raise RuntimeError(f"Order hash covered {order_count} of {len(dataset)} records")
    args.train_sample_order_sha256 = order_hash

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_source = args.resume_from or args.model_dir
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
    )
    worker_processors = build_worker_processors(args.model_dir, args.preprocess_workers)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        str(model_source),
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    trainable_parameters, trainable_names = configure_trainable(
        model,
        args.train_scope,
    )
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())

    updates_per_epoch = math.ceil(
        len(dataset) / args.gradient_accumulation
    )
    planned_updates = updates_per_epoch * args.epochs
    if args.max_updates > 0:
        planned_updates = min(planned_updates, args.max_updates)
    warmup_steps = int(round(planned_updates * args.warmup_ratio))

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, planned_updates),
    )

    resume = {
        "epoch": 0,
        "next_batch_index": 0,
        "global_update": 0,
        "examples_seen": 0,
    }
    if args.resume_from is not None:
        resume = load_resume_state(args.resume_from, optimizer, scheduler)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_sample_order.json").write_text(json.dumps({"seed": args.seed, "records": order_count, "sha256": order_hash}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_metadata(args), indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = args.output_dir / "train_metrics.jsonl"

    print(f"Device: {device}; dtype: {dtype}")
    print(f"Manifest: {args.manifest}")
    print(f"Training QA records: {len(dataset)}")
    print(f"Train scope: {args.train_scope}")
    print(
        f"Trainable parameters: {trainable_count:,}/{total_count:,} "
        f"({100.0 * trainable_count / total_count:.2f}%)"
    )
    print(f"Trainable tensors: {len(trainable_names)}")
    print(f"First trainable tensors: {trainable_names[:5]}")
    print(
        f"Effective batch: "
        f"{args.batch_size * args.gradient_accumulation}"
    )
    print(
        f"Planned optimizer updates: {planned_updates}; "
        f"warmup: {warmup_steps}"
    )
    print(
        "Ordered CPU preprocessing: "
        f"workers={args.preprocess_workers} max_inflight={args.preprocess_prefetch} "
        f"torch_threads={torch.get_num_threads()} order_sha256={order_hash}"
    )
    print(
        "Image splitting: "
        f"{getattr(processor.image_processor, 'do_image_splitting', 'unknown')}"
    )
    if args.resume_from is not None:
        print(f"Resuming from: {args.resume_from}")
        print(f"Resume state: {resume}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_update = resume["global_update"]
    examples_seen = resume["examples_seen"]
    start_time = time.time()
    last_update_time = start_time
    stop_training = False

    for epoch in range(resume["epoch"], args.epochs):
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=single_item_collate,
        )
        skip_batches = (
            resume["next_batch_index"]
            if epoch == resume["epoch"]
            else 0
        )
        pending = 0
        accumulated_loss = 0.0
        accumulated_data_wait = 0.0
        encoded_iterator = ordered_encoded_batches(
            loader,
            skip_batches,
            processor,
            worker_processors,
            args.preprocess_prefetch,
            encode_example,
        )

        for batch_index, example, encoded, data_wait_seconds in encoded_iterator:
            inputs, labels, _, _ = encoded
            accumulated_data_wait += data_wait_seconds
            inputs = move_tensors(inputs, device)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=device == "cuda",
            ):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch} "
                    f"batch={batch_index}: {loss.item()}"
                )
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach().item())
            pending += 1
            examples_seen += 1

            is_last_batch = batch_index + 1 == len(loader)
            if pending < args.gradient_accumulation and not is_last_batch:
                continue
            if pending != args.gradient_accumulation:
                correction = args.gradient_accumulation / pending
                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"Non-finite gradient norm at update {global_update + 1}"
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

            mean_loss = accumulated_loss / pending
            accumulated_loss = 0.0
            batch_data_wait_seconds = accumulated_data_wait
            accumulated_data_wait = 0.0
            pending = 0
            elapsed = time.time() - start_time
            update_elapsed = time.time() - last_update_time
            last_update_time = time.time()
            peak_mib = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if device == "cuda"
                else 0.0
            )
            metric = {
                "epoch": epoch + 1,
                "update": global_update,
                "examples_seen": examples_seen,
                "loss": mean_loss,
                "grad_norm_before_clip": float(grad_norm.item()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": elapsed,
                "update_elapsed_seconds": update_elapsed,
                "data_wait_seconds": batch_data_wait_seconds,
                "peak_allocated_vram_mib": peak_mib,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric) + "\n")

            if (
                global_update == 1
                or global_update % args.log_every == 0
                or global_update == planned_updates
            ):
                print(
                    f"epoch={epoch + 1} update={global_update}/"
                    f"{planned_updates} loss={mean_loss:.6f} "
                    f"grad={grad_norm.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"peak_vram={peak_mib:.0f}MiB"
                )

            if (
                not args.no_save
                and args.checkpoint_every > 0
                and global_update % args.checkpoint_every == 0
            ):
                save_checkpoint(
                    args.output_dir / "checkpoint-last",
                    model,
                    processor,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    batch_index + 1,
                    global_update,
                    examples_seen,
                )
                print(f"Saved checkpoint at update {global_update}")

            if args.max_updates > 0 and global_update >= args.max_updates:
                stop_training = True
                break

        resume["next_batch_index"] = 0
        if stop_training:
            break

    if not args.no_save:
        final_dir = args.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.config.use_cache = True
        model.save_pretrained(final_dir, safe_serialization=True)
        processor.save_pretrained(final_dir)
        (final_dir / "run_config.json").write_text(
            json.dumps(run_metadata(args), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved final model: {final_dir}")

    total_elapsed = time.time() - start_time
    print(
        f"TRAINING COMPLETE: updates={global_update} "
        f"examples={examples_seen} elapsed={total_elapsed:.1f}s"
    )
    if args.no_save:
        print("NO-SAVE MODE: model weights were not written")


if __name__ == "__main__":
    main()
