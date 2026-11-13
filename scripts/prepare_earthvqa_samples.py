#!/usr/bin/env python3
"""Create a reproducible, QA-level EarthVQA train/dev manifest.

The original annotations and images are never modified. Records contain only
image IDs, QA indices, and original question types, so every experiment can
reuse exactly the same examples without duplicating the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "EarthVQA"
ANNOTATION_PATH = DATA_ROOT / "Train_QA.json"
DEFAULT_IMAGE_SPLIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "earthvqa_train_dev_seed42.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manifests"
    / "earthvqa_baseline_30k_seed42.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-split", type=Path, default=DEFAULT_IMAGE_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=30_000,
        help="QA records sampled from the training images; <=0 keeps all.",
    )
    parser.add_argument(
        "--dev-samples",
        type=int,
        default=0,
        help="QA records sampled from dev images; <=0 keeps all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_valid_answer(qa: dict[str, Any]) -> bool:
    if "Answer" not in qa or qa["Answer"] is None:
        return False
    answer = qa["Answer"]
    return not isinstance(answer, str) or bool(answer.strip())


def build_records(
    annotations: dict[str, list[dict[str, Any]]],
    image_ids: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for image_id in image_ids:
        if image_id not in annotations:
            raise KeyError(f"Missing annotation entry for image: {image_id}")
        for qa_index, qa in enumerate(annotations[image_id]):
            question = str(qa.get("Question", "")).strip()
            if not question or not has_valid_answer(qa):
                continue
            records.append(
                {
                    "image_id": image_id,
                    "qa_index": qa_index,
                    "type": str(qa.get("Type", "MISSING")),
                }
            )
    return records


def allocate_quotas(type_counts: Counter[str], limit: int) -> dict[str, int]:
    total = sum(type_counts.values())
    if limit <= 0 or limit >= total:
        return dict(type_counts)

    raw = {
        name: limit * count / total
        for name, count in sorted(type_counts.items())
    }
    quotas = {name: int(value) for name, value in raw.items()}
    remaining = limit - sum(quotas.values())
    ranking = sorted(
        raw,
        key=lambda name: (-(raw[name] - quotas[name]), name),
    )
    for name in ranking[:remaining]:
        quotas[name] += 1
    return quotas


def stratified_sample(
    records: list[dict[str, Any]],
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["type"]].append(record)

    counts = Counter({name: len(items) for name, items in groups.items()})
    quotas = allocate_quotas(counts, limit)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for name in sorted(groups):
        group = list(groups[name])
        rng.shuffle(group)
        selected.extend(group[: quotas[name]])
    rng.shuffle(selected)
    return selected, quotas


def summarize(records: list[dict[str, Any]]) -> tuple[int, Counter[str]]:
    images = {record["image_id"] for record in records}
    counts = Counter(record["type"] for record in records)
    return len(images), counts


def main() -> None:
    args = parse_args()
    for path in (ANNOTATION_PATH, args.image_split):
        if not path.is_file():
            raise FileNotFoundError(path)

    annotations = json.loads(ANNOTATION_PATH.read_text(encoding="utf-8"))
    image_split = json.loads(args.image_split.read_text(encoding="utf-8"))
    train_ids = list(image_split["train_image_ids"])
    dev_ids = list(image_split["dev_image_ids"])

    overlap = set(train_ids) & set(dev_ids)
    if overlap:
        raise RuntimeError(f"Train/dev image leakage: {len(overlap)} images")

    train_all = build_records(annotations, train_ids)
    dev_all = build_records(annotations, dev_ids)
    train_records, train_quotas = stratified_sample(
        train_all,
        args.train_samples,
        args.seed,
    )
    dev_records, dev_quotas = stratified_sample(
        dev_all,
        args.dev_samples,
        args.seed + 1,
    )

    output = {
        "version": 1,
        "dataset": "EarthVQA",
        "seed": args.seed,
        "source_annotation": str(ANNOTATION_PATH),
        "source_annotation_sha256": sha256_path(ANNOTATION_PATH),
        "image_split_manifest": str(args.image_split),
        "image_split_manifest_sha256": sha256_path(args.image_split),
        "selection": {
            "train_requested": args.train_samples,
            "dev_requested": args.dev_samples,
            "strategy": "seeded proportional stratification by original Type",
            "train_quotas": dict(sorted(train_quotas.items())),
            "dev_quotas": dict(sorted(dev_quotas.items())),
        },
        "splits": {
            "train": train_records,
            "dev": dev_records,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Source annotation SHA256: {output['source_annotation_sha256']}")
    print(f"Image split SHA256: {output['image_split_manifest_sha256']}")
    for split_name, records in output["splits"].items():
        unique_images, counts = summarize(records)
        print(
            f"\n{split_name}: QA={len(records)} "
            f"unique_images={unique_images}"
        )
        for type_name, count in sorted(counts.items()):
            print(f"  {type_name}: {count}")
    print(f"\nWrote immutable-index manifest: {args.output}")


if __name__ == "__main__":
    main()
