#!/usr/bin/env python3
"""Create a deterministic image-level EarthVQA Train/Dev manifest.

The source QA JSON and images are read-only. The output contains IDs and counts,
not copied or rewritten QA records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "EarthVQA"
ANNOTATION_PATH = DATA_ROOT / "Train_QA.json"
IMAGE_DIR = DATA_ROOT / "Train" / "images_png"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_types(image_ids: list[str], annotations: dict) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for image_id in image_ids:
        for qa in annotations[image_id]:
            counts[str(qa.get("Type", "MISSING"))] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.dev_fraction < 1.0:
        raise SystemExit("--dev-fraction must be between 0 and 1")
    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(f"Missing annotations: {ANNOTATION_PATH}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Missing image directory: {IMAGE_DIR}")

    with ANNOTATION_PATH.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, dict) or len(annotations) < 2:
        raise ValueError("Expected an image_id -> QA-list JSON object")

    image_ids = sorted(str(image_id) for image_id in annotations)
    missing = [
        image_id
        for image_id in image_ids
        if not (IMAGE_DIR / Path(image_id).name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} annotation image IDs have no image file; "
            f"examples: {missing[:5]}"
        )

    ranked = sorted(
        image_ids,
        key=lambda image_id: hashlib.sha256(
            f"{args.seed}:{image_id}".encode("utf-8")
        ).hexdigest(),
    )
    dev_size = max(1, round(len(image_ids) * args.dev_fraction))
    dev_set = set(ranked[:dev_size])
    train_ids = [image_id for image_id in image_ids if image_id not in dev_set]
    dev_ids = [image_id for image_id in image_ids if image_id in dev_set]

    manifest = {
        "format_version": 1,
        "dataset": "EarthVQA",
        "source_split": "official Train",
        "split_unit": "image_id",
        "seed": args.seed,
        "dev_fraction_requested": args.dev_fraction,
        "annotation_sha256": sha256_file(ANNOTATION_PATH),
        "train_image_ids": train_ids,
        "dev_image_ids": dev_ids,
        "qa_counts_by_type": {
            "train": count_types(train_ids, annotations),
            "dev": count_types(dev_ids, annotations),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"earthvqa_train_dev_seed{args.seed}.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing == manifest:
            print(f"Manifest already exists and matches: {output_path}")
            return
        raise FileExistsError(
            f"Refusing to replace a different manifest: {output_path}"
        )
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Source SHA256: {manifest['annotation_sha256']}")
    print(f"Images: total={len(image_ids)} train={len(train_ids)} dev={len(dev_ids)}")
    for split_name in ("train", "dev"):
        qa_total = sum(manifest["qa_counts_by_type"][split_name].values())
        print(f"\n{split_name}: images={len(manifest[split_name + '_image_ids'])} QA={qa_total}")
        for qa_type, count in manifest["qa_counts_by_type"][split_name].items():
            print(f"  {qa_type}: {count}")
    print(f"\nWrote ID-only manifest: {output_path}")


if __name__ == "__main__":
    main()
