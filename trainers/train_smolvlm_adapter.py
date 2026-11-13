#!/usr/bin/env python3
"""Fine-tune B group: SmolVLM-500M + fixed ordinary bottleneck Adapter.

This script consumes the immutable QA-index manifest produced by
scripts/prepare_earthvqa_samples.py. It supports a one-update dry run,
gradient accumulation, gradient clipping, deterministic epoch shuffling,
periodic resumable checkpoints, and answer-only supervision. The baseline
data path, prompt, masking, optimizer, schedule, and budget are preserved.
This is a capacity control, not a tuned method variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "b_adapter_earthvqa_seed42"
PROMPT_PREFIX = (
    "Answer the question about this remote-sensing image with only the "
    "concise answer.\nQuestion: "
)

# Running this trainer directly places trainers/ first on
# sys.path. Add the existing project root so the sibling package is importable.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smolvlm_adapter import (  # noqa: E402
    attach_visual_adapter,
    load_visual_adapter,
    save_adapter_pretrained,
)
from smolvlm_adapter.integration_example import (  # noqa: E402
    assert_optimizer_contains_adapter,
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
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=0,
        help="Ordered CPU encode_example thread workers; 0 preserves serial encoding.",
    )
    parser.add_argument(
        "--preprocess-prefetch",
        type=int,
        default=0,
        help="Maximum ordered encoded samples in flight; required when workers > 0.",
    )
    parser.add_argument(
        "--cpu-torch-threads",
        type=int,
        default=0,
        help="CPU intra-op threads; 0 keeps the environment default.",
    )
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


def run_metadata(args: argparse.Namespace, adapter_handle: Any) -> dict[str, Any]:
    """Keep the baseline arguments intact and record the fixed B-group design."""
    return {
        **serializable_args(args),
        "experiment_group": "B",
        "method": "SmolVLM-500M-Instruct + ordinary bottleneck Adapter",
        "initialization": "original pretrained checkpoint, never baseline final",
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
        "adapter": adapter_handle.metadata(),
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


_ENCODE_THREAD_LOCAL = threading.local()


def build_worker_processors(model_dir: Path, count: int) -> list[Any]:
    """Create independent processors without advancing the training RNG state."""
    if count <= 0:
        return []
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    try:
        return [
            AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
            for _ in range(count)
        ]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _initialize_encode_thread(processors: queue.SimpleQueue) -> None:
    _ENCODE_THREAD_LOCAL.processor = processors.get()


def _encode_in_thread(example: dict[str, Any]):
    processor = getattr(_ENCODE_THREAD_LOCAL, "processor", None)
    if processor is None:
        raise RuntimeError("Ordered preprocessing thread has no processor")
    return encode_example(processor, example)


def ordered_encoded_batches(
    loader: DataLoader,
    skip_batches: int,
    serial_processor: Any,
    worker_processors: list[Any],
    max_inflight: int,
):
    """Yield encoded samples in the original DataLoader order.

    Only deterministic CPU preprocessing runs in worker threads. DataLoader
    iteration, CUDA transfer, forward/backward and optimizer work stay on the
    main thread. Prefetched-but-unconsumed records never affect checkpoint
    counters or the optimizer.
    """
    source = iter(enumerate(loader))
    for _ in range(skip_batches):
        try:
            next(source)
        except StopIteration as error:
            raise RuntimeError("Resume skip exceeds the epoch DataLoader") from error

    if not worker_processors:
        for batch_index, example in source:
            yield batch_index, example, encode_example(serial_processor, example)
        return

    processor_queue: queue.SimpleQueue = queue.SimpleQueue()
    for processor in worker_processors:
        processor_queue.put(processor)
    executor = ThreadPoolExecutor(
        max_workers=len(worker_processors),
        initializer=_initialize_encode_thread,
        initargs=(processor_queue,),
        thread_name_prefix="earthvqa-encode",
    )
    pending: deque[tuple[int, dict[str, Any], Any]] = deque()

    def fill() -> None:
        while len(pending) < max_inflight:
            try:
                batch_index, example = next(source)
            except StopIteration:
                break
            pending.append(
                (batch_index, example, executor.submit(_encode_in_thread, example))
            )

    try:
        fill()
        while pending:
            batch_index, example, future = pending.popleft()
            try:
                encoded = future.result()
            except Exception as error:
                raise RuntimeError(
                    "Ordered CPU preprocessing failed for "
                    f"batch={batch_index}, image={example.get('image_id')}, "
                    f"qa_index={example.get('qa_index')}"
                ) from error
            yield batch_index, example, encoded
            fill()
    finally:
        for _, _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


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
    adapter_handle: Any,
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
    save_adapter_pretrained(
        model,
        adapter_handle,
        checkpoint_dir,
        safe_serialization=True,
    )
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
        json.dumps(run_metadata(args, adapter_handle), indent=2) + "\n",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_source = args.resume_from or args.model_dir
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
    )
    worker_processors = build_worker_processors(
        args.model_dir,
        args.preprocess_workers,
    )
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

    # Restore the outer CPU RNG state after deterministic Adapter creation so
    # the matched baseline seed stream is unchanged outside the method module.
    with torch.random.fork_rng(devices=[]):
        if args.resume_from is None:
            adapter_handle = attach_visual_adapter(
                model,
                hidden_size=960,
                bottleneck_size=615,
            )
        else:
            adapter_handle = load_visual_adapter(model, args.resume_from)
    adapter_metadata = adapter_handle.metadata()
    if adapter_metadata["hidden_size"] != 960 or adapter_metadata["bottleneck_size"] != 615:
        raise RuntimeError("B group requires the fixed 960->615->960 Adapter")
    if adapter_metadata["added_parameters"] != 1_182_375:
        raise RuntimeError(
            "B-group Adapter parameter mismatch: "
            f"{adapter_metadata['added_parameters']} != 1,182,375"
        )

    adapter_parameter_ids = {
        id(parameter) for parameter in adapter_handle.adapter.parameters()
    }
    for name, parameter in model.named_parameters():
        if id(parameter) in adapter_parameter_ids:
            parameter.requires_grad_(True)
            trainable_parameters.append(parameter)
            trainable_names.append(name)

    if len({id(parameter) for parameter in trainable_parameters}) != len(trainable_parameters):
        raise RuntimeError("Duplicate parameter detected in optimizer input list")
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
    assert_optimizer_contains_adapter(optimizer, adapter_handle)
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
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_metadata(args, adapter_handle), indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = args.output_dir / "train_metrics.jsonl"

    print(f"Device: {device}; dtype: {dtype}")
    print(f"Manifest: {args.manifest}")
    print(f"Training QA records: {len(dataset)}")
    print(f"Train scope: {args.train_scope}")
    print(f"Adapter: {json.dumps(adapter_handle.metadata(), sort_keys=True)}")
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
        "Image splitting: "
        f"{getattr(processor.image_processor, 'do_image_splitting', 'unknown')}"
    )
    print(
        "Ordered CPU preprocessing: "
        f"workers={args.preprocess_workers} "
        f"max_inflight={args.preprocess_prefetch} "
        f"torch_threads={torch.get_num_threads()}"
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

        encoded_iterator = ordered_encoded_batches(
            loader,
            skip_batches,
            processor,
            worker_processors,
            args.preprocess_prefetch,
        )
        for batch_index, example, encoded in encoded_iterator:
            inputs, labels, _, _ = encoded
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

            down_grad = adapter_handle.adapter.down.weight.grad
            up_grad = adapter_handle.adapter.up.weight.grad
            up_bias_grad = adapter_handle.adapter.up.bias.grad
            if down_grad is None or up_grad is None or up_bias_grad is None:
                raise RuntimeError("Adapter gradients are missing")
            down_grad_norm = float(down_grad.float().norm().item())
            up_grad_norm = float(up_grad.float().norm().item())
            up_bias_grad_norm = float(up_bias_grad.float().norm().item())
            adapter_grad_norms = (
                down_grad_norm,
                up_grad_norm,
                up_bias_grad_norm,
            )
            if not all(math.isfinite(value) for value in adapter_grad_norms):
                raise RuntimeError("Adapter gradients must be finite")
            # W2 starts at zero, so W1 receives zero gradient until W2 has
            # actually changed. With the matched warmup schedule, optimizer
            # update 1 uses lr=0; therefore W1 can also be zero on update 2.
            # Gate W1 on the real pre-step W2 state instead of update number.
            up_nonzero_before_step = int(
                torch.count_nonzero(adapter_handle.adapter.up.weight).item()
            )
            if global_update == 0 and (
                up_grad_norm == 0.0 or up_bias_grad_norm == 0.0
            ):
                raise RuntimeError(
                    "Zero-initialized Adapter W2 must receive gradient on update 1"
                )
            if up_nonzero_before_step > 0 and (
                down_grad_norm == 0.0 or up_grad_norm == 0.0
            ):
                raise RuntimeError(
                    "Both Adapter projections must receive gradient once W2 is nonzero"
                )

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                args.max_grad_norm,
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"Non-finite gradient norm at update {global_update + 1}"
                )

            optimizer_lr_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if (
                optimizer_lr_used > 0.0
                and up_grad_norm > 0.0
                and torch.count_nonzero(adapter_handle.adapter.up.weight).item() == 0
            ):
                raise RuntimeError("A positive-lr optimizer step did not update Adapter W2")
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

            mean_loss = accumulated_loss / pending
            accumulated_loss = 0.0
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
                "optimizer_learning_rate_used": optimizer_lr_used,
                "elapsed_seconds": elapsed,
                "update_elapsed_seconds": update_elapsed,
                "peak_allocated_vram_mib": peak_mib,
                "adapter_down_grad_norm_before_clip": down_grad_norm,
                "adapter_up_grad_norm_before_clip": up_grad_norm,
                "adapter_up_bias_grad_norm_before_clip": up_bias_grad_norm,
                "adapter_up_nonzero_before_step": up_nonzero_before_step,
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
                    f"peak_vram={peak_mib:.0f}MiB "
                    f"adapter_down_g={down_grad_norm:.4f} "
                    f"adapter_up_g={up_grad_norm:.4f}"
                )

            if (
                not args.no_save
                and args.checkpoint_every > 0
                and global_update % args.checkpoint_every == 0
            ):
                save_checkpoint(
                    args.output_dir / "checkpoint-last",
                    model,
                    adapter_handle,
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

        encoded_iterator.close()
        resume["next_batch_index"] = 0
        if stop_training:
            break

    if not args.no_save:
        final_dir = args.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.config.use_cache = True
        save_adapter_pretrained(
            model,
            adapter_handle,
            final_dir,
            safe_serialization=True,
        )
        processor.save_pretrained(final_dir)
        (final_dir / "run_config.json").write_text(
            json.dumps(run_metadata(args, adapter_handle), indent=2) + "\n",
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
