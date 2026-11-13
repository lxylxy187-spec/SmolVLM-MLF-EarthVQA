#!/usr/bin/env python3
"""Evaluate B-group SmolVLM + ordinary Adapter with the unchanged EarthVQA protocol."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "b_adapter_earthvqa_seed42" / "final"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifests"
    / "earthvqa_baseline_30k_seed42.json"
)
TRAIN_ANNOTATIONS = PROJECT_ROOT / "data" / "raw" / "EarthVQA" / "Train_QA.json"
VAL_ANNOTATIONS = PROJECT_ROOT / "data" / "raw" / "EarthVQA" / "Val_QA.json"
TRAIN_IMAGES = PROJECT_ROOT / "data" / "raw" / "EarthVQA" / "Train" / "images_png"
VAL_IMAGES = PROJECT_ROOT / "data" / "raw" / "EarthVQA" / "Val" / "images_png"
PROMPT_PREFIX = (
    "Answer the question about this remote-sensing image with only the concise answer.\n"
    "Question: "
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smolvlm_adapter import load_visual_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("dev", "official_val"), default="dev")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required for official_val because it is reserved as the final test set.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_answer(value: Any) -> str:
    text = str(value if value is not None else "").lower()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(gold: str, prediction: str) -> float:
    gold_tokens = gold.split()
    pred_tokens = prediction.split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = sum((Counter(gold_tokens) & Counter(pred_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def unwrap_manifest_split(manifest: dict[str, Any], split: str) -> list[Any]:
    aliases = (split, f"{split}_records", f"{split}_samples")
    node: Any = None
    for container in (manifest.get("splits"), manifest):
        if not isinstance(container, dict):
            continue
        for key in aliases:
            if key in container:
                node = container[key]
                break
        if node is not None:
            break
    if node is None:
        raise KeyError(f"Could not find split '{split}' in {manifest.keys()}")
    while isinstance(node, dict):
        for key in ("records", "samples", "items", "qa_records"):
            if key in node:
                node = node[key]
                break
        else:
            raise TypeError(f"Unsupported split object keys: {node.keys()}")
    if not isinstance(node, list):
        raise TypeError(f"Expected a list for split '{split}', got {type(node).__name__}")
    return node


def annotation_key(annotations: dict[str, Any], image_id: str) -> str:
    name = Path(str(image_id)).name
    candidates = (str(image_id), name, Path(name).stem, f"{Path(name).stem}.png")
    for candidate in candidates:
        if candidate in annotations:
            return candidate
    raise KeyError(f"Image id '{image_id}' is absent from the annotation file")


def record_from_index(
    item: Any,
    annotations: dict[str, Any],
    default_index: int,
) -> dict[str, Any]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        image_id, qa_index = item[0], item[1]
        item = {}
    elif isinstance(item, dict):
        image_id = next(
            (
                item[key]
                for key in ("image_id", "image", "image_name", "filename")
                if key in item
            ),
            None,
        )
        qa_index = next(
            (
                item[key]
                for key in ("qa_index", "question_index", "index")
                if key in item
            ),
            None,
        )
    else:
        raise TypeError(f"Unsupported manifest record: {item!r}")

    if image_id is None:
        raise KeyError(f"Manifest record has no image id: {item}")
    key = annotation_key(annotations, str(image_id))

    if isinstance(item, dict) and "Question" in item:
        qa = item
        if qa_index is None:
            qa_index = default_index
    else:
        if qa_index is None:
            raise KeyError(f"Manifest record has no QA index: {item}")
        qa = annotations[key][int(qa_index)]

    return {
        "image_id": Path(key).name,
        "qa_index": int(qa_index),
        "type": str(qa.get("Type", "UNKNOWN")),
        "question": str(qa.get("Question", "")),
        "answer": qa.get("Answer", ""),
    }


def build_records(
    split: str,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], Path]:
    if split == "dev":
        manifest = load_json(manifest_path)
        annotations = load_json(TRAIN_ANNOTATIONS)
        raw_records = unwrap_manifest_split(manifest, "dev")
        records = [
            record_from_index(item, annotations, index)
            for index, item in enumerate(raw_records)
        ]
        return records, TRAIN_IMAGES

    annotations = load_json(VAL_ANNOTATIONS)
    records: list[dict[str, Any]] = []
    for image_id, qas in annotations.items():
        for qa_index, qa in enumerate(qas):
            records.append(
                {
                    "image_id": Path(image_id).name,
                    "qa_index": qa_index,
                    "type": str(qa.get("Type", "UNKNOWN")),
                    "question": str(qa.get("Question", "")),
                    "answer": qa.get("Answer", ""),
                }
            )
    return records, VAL_IMAGES


def image_path_for(root: Path, image_id: str) -> Path:
    direct = root / Path(image_id).name
    if direct.exists():
        return direct
    stem = Path(image_id).stem
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for '{image_id}' under {root}")


def build_prompt(processor: Any, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_PREFIX + question},
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def result_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["image_id"]), int(record["qa_index"])


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def summarize(
    rows: list[dict[str, Any]],
    split: str,
    model_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No predictions were produced")

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["type"])].append(row)

    per_type: dict[str, dict[str, float | int]] = {}
    for qa_type, items in sorted(by_type.items()):
        per_type[qa_type] = {
            "count": len(items),
            "exact_match": sum(float(item["exact_match"]) for item in items) / len(items),
            "token_f1": sum(float(item["token_f1"]) for item in items) / len(items),
        }

    yes_no = [
        row
        for row in rows
        if row["normalized_gold"] in {"yes", "no"}
    ]
    recalls: dict[str, float] = {}
    for label in ("yes", "no"):
        class_rows = [row for row in yes_no if row["normalized_gold"] == label]
        if class_rows:
            recalls[label] = (
                sum(row["normalized_prediction"] == label for row in class_rows)
                / len(class_rows)
            )

    summary: dict[str, Any] = {
        "split": split,
        "model_dir": str(model_dir.resolve()),
        "samples": len(rows),
        "exact_match": sum(float(row["exact_match"]) for row in rows) / len(rows),
        "token_f1": sum(float(row["token_f1"]) for row in rows) / len(rows),
        "macro_type_accuracy": (
            sum(float(metrics["exact_match"]) for metrics in per_type.values())
            / len(per_type)
        ),
        "per_type": per_type,
        "yes_no_samples": len(yes_no),
        "yes_no_recall": recalls,
        "yes_no_balanced_accuracy": (
            sum(recalls.values()) / len(recalls) if recalls else None
        ),
        "elapsed_seconds_this_run": elapsed_seconds,
        "generation_seconds_total": sum(
            float(row.get("generation_seconds", 0.0)) for row in rows
        ),
        "mean_generation_seconds": (
            sum(float(row.get("generation_seconds", 0.0)) for row in rows)
            / len(rows)
        ),
    }
    return summary


def main() -> None:
    args = parse_args()
    if args.split == "official_val" and not args.confirm_final_test:
        raise SystemExit(
            "official_val is the reserved final test set. "
            "After all model choices are frozen, rerun with --confirm-final-test."
        )
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")

    records, image_root = build_records(args.split, args.manifest)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    output_dir = args.output_dir or (args.model_dir / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.split}_predictions.jsonl"
    metrics_path = output_dir / f"{args.split}_metrics.json"

    if args.overwrite:
        predictions_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)

    existing = read_existing(predictions_path)
    completed = {result_key(row) for row in existing}
    pending = [record for record in records if result_key(record) not in completed]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}; dtype: {dtype}", flush=True)
    print(f"Model: {args.model_dir.resolve()}", flush=True)
    print(
        f"Split: {args.split}; total={len(records)} "
        f"completed={len(existing)} pending={len(pending)}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.model.vision_model.requires_grad_(False)
    adapter_handle = load_visual_adapter(model, args.model_dir)
    model.eval()

    started = time.time()
    mode = "a" if predictions_path.exists() else "w"
    with predictions_path.open(mode, encoding="utf-8") as output_handle:
        for pending_index, record in enumerate(pending, 1):
            path = image_path_for(image_root, record["image_id"])
            with Image.open(path) as opened:
                image = opened.convert("RGB")

            prompt = build_prompt(processor, record["question"])
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[-1]

            generation_started = time.perf_counter()
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=dtype):
                        generated = model.generate(
                            **inputs,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=args.max_new_tokens,
                        )
                else:
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=args.max_new_tokens,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started

            prediction = processor.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
            )[0].strip()
            normalized_gold = normalize_answer(record["answer"])
            normalized_prediction = normalize_answer(prediction)
            row = {
                **record,
                "gold": str(record["answer"]),
                "prediction": prediction,
                "normalized_gold": normalized_gold,
                "normalized_prediction": normalized_prediction,
                "exact_match": normalized_gold == normalized_prediction,
                "token_f1": token_f1(normalized_gold, normalized_prediction),
                "generation_seconds": generation_seconds,
            }
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()

            processed = len(existing) + pending_index
            if (
                pending_index == 1
                or processed % args.log_every == 0
                or pending_index == len(pending)
            ):
                elapsed = time.time() - started
                print(
                    f"{processed}/{len(records)} "
                    f"latest_em={int(row['exact_match'])} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    all_rows = read_existing(predictions_path)
    allowed = {result_key(record) for record in records}
    all_rows = [row for row in all_rows if result_key(row) in allowed]
    summary = summarize(
        all_rows,
        args.split,
        args.model_dir,
        time.time() - started,
    )
    summary["adapter"] = adapter_handle.metadata()
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Predictions: {predictions_path.resolve()}", flush=True)
    print(f"Metrics: {metrics_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
