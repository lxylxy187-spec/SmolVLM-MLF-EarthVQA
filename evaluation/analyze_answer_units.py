#!/usr/bin/env python3
"""Compute deterministic concept-unit coverage metrics from prediction JSONL.

The metric is supplementary to EM/Token-F1. It is evaluated only on EarthVQA
complex-category samples whose gold answer contains at least two parsed units.
The same frozen parser must be used for Baseline and C.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any


PARSER_VERSION = "earthvqa_template_units_v1"
COMPLEX_TYPES = {"Comprehensive Analysis", "Object Situation Analysis"}
GENERIC_SHARED_HEADS = ("areas", "area", "roads", "road", "buildings", "building", "lanes", "lane")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def clean_surface(value: Any) -> str:
    text = str(value if value is not None else "").lower().replace("-", " ")
    text = text.translate(str.maketrans({char: " " if char != "," else "," for char in string.punctuation}))
    return " ".join(text.split())


def canonical_unit(value: str) -> str:
    unit = " ".join(value.strip(" ,").split())
    unit = re.sub(r"^(?:there (?:are|is)|this is)\s+", "", unit)
    unit = re.sub(r"^(?:the|a|an)\s+", "", unit)
    unit = unit.replace("agriculture land", "agricultural land")
    if unit and not unit.startswith("no "):
        for head in GENERIC_SHARED_HEADS:
            suffix = f" {head}"
            if unit.endswith(suffix):
                unit = unit[: -len(suffix)].strip()
                break
    return unit


def split_items(value: str, propagate_no_for_nor: bool = False) -> list[str]:
    had_nor = bool(re.search(r"\s+nor\s+", value))
    items = [
        canonical_unit(item)
        for item in re.split(r"\s*,\s*|\s+(?:and|nor)\s+", value)
        if item.strip(" ,")
    ]
    if propagate_no_for_nor and had_nor and items and items[0].startswith("no "):
        items = [items[0]] + [item if item.startswith("no ") else f"no {item}" for item in items[1:]]
    return [item for item in items if item]


def answer_units(value: Any) -> list[str]:
    text = clean_surface(value)
    if not text:
        return []

    match = re.match(r"^there (?:are|is)\s+(.+)$", text)
    if match:
        return sorted(set(split_items(match.group(1), propagate_no_for_nor=True)))

    match = re.match(r"^this is\s+(.+?)\s+with\s+(.+)$", text)
    if match:
        status = canonical_unit(match.group(1))
        objects = split_items(match.group(2))
        return sorted(set([status, *objects]))

    if " need" in text or " needs" in text:
        return sorted(set(split_items(text)))

    return [canonical_unit(text)]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def token_length(value: Any) -> int:
    return len(clean_surface(value).replace(",", " ").split())


def aggregate(details: list[dict[str, Any]]) -> dict[str, Any]:
    if not details:
        return {
            "samples": 0,
            "gold_units": 0,
            "predicted_units": 0,
            "matched_units": 0,
            "answer_unit_precision_micro": None,
            "answer_unit_recall_micro": None,
            "answer_unit_precision_macro": None,
            "answer_unit_recall_macro": None,
            "complete_coverage_rate": None,
            "omission_rate": None,
            "exact_unit_set_rate": None,
        }
    gold_total = sum(len(item["gold_units"]) for item in details)
    pred_total = sum(len(item["predicted_units"]) for item in details)
    matched_total = sum(item["matched_count"] for item in details)
    return {
        "samples": len(details),
        "gold_units": gold_total,
        "predicted_units": pred_total,
        "matched_units": matched_total,
        "answer_unit_precision_micro": matched_total / pred_total if pred_total else 0.0,
        "answer_unit_recall_micro": matched_total / gold_total if gold_total else 0.0,
        "answer_unit_precision_macro": sum(item["precision"] for item in details) / len(details),
        "answer_unit_recall_macro": sum(item["recall"] for item in details) / len(details),
        "complete_coverage_rate": sum(item["complete_coverage"] for item in details) / len(details),
        "omission_rate": sum(item["omitted_count"] for item in details) / gold_total if gold_total else 0.0,
        "exact_unit_set_rate": sum(item["exact_unit_set"] for item in details) / len(details),
        "mean_gold_unit_count": gold_total / len(details),
        "mean_predicted_unit_count": pred_total / len(details),
        "mean_prediction_tokens": sum(item["prediction_tokens"] for item in details) / len(details),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.predictions)
    details: list[dict[str, Any]] = []
    complex_rows = [row for row in rows if str(row.get("type")) in COMPLEX_TYPES]
    for row in complex_rows:
        gold = answer_units(row.get("gold", row.get("answer", "")))
        predicted = answer_units(row.get("prediction", ""))
        if len(gold) < 2:
            continue
        gold_set = set(gold)
        pred_set = set(predicted)
        matched = gold_set & pred_set
        precision = len(matched) / len(pred_set) if pred_set else 0.0
        recall = len(matched) / len(gold_set)
        details.append(
            {
                "image_id": row.get("image_id"),
                "qa_index": row.get("qa_index"),
                "type": row.get("type"),
                "gold": row.get("gold", row.get("answer", "")),
                "prediction": row.get("prediction", ""),
                "gold_units": gold,
                "predicted_units": predicted,
                "matched_units": sorted(matched),
                "matched_count": len(matched),
                "omitted_count": len(gold_set - pred_set),
                "extra_count": len(pred_set - gold_set),
                "precision": precision,
                "recall": recall,
                "complete_coverage": gold_set <= pred_set,
                "exact_unit_set": gold_set == pred_set,
                "prediction_tokens": token_length(row.get("prediction", "")),
            }
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in details:
        by_type[str(item["type"])].append(item)

    metrics = {
        "parser_version": PARSER_VERSION,
        "definition": {
            "scope": "Gold answers with >=2 deterministic units in Comprehensive Analysis or Object Situation Analysis",
            "precision": "matched predicted units / all predicted units",
            "recall": "matched gold units / all gold units",
            "complete_coverage_rate": "fraction of scoped answers containing every gold unit; extra units allowed",
            "omission_rate": "omitted gold units / all gold units = 1 - micro recall",
            "note": "Supplementary template-based concept metric; EM and Token-F1 remain primary.",
        },
        "source_predictions": str(args.predictions.resolve()),
        "all_samples": len(rows),
        "complex_category_samples": len(complex_rows),
        "multi_unit_samples": len(details),
        "overall": aggregate(details),
        "per_type": {
            qa_type: aggregate(items) for qa_type, items in sorted(by_type.items())
        },
        "answer_length": {
            "mean_prediction_tokens_all": sum(token_length(row.get("prediction", "")) for row in rows) / len(rows),
            "mean_gold_tokens_all": sum(token_length(row.get("gold", row.get("answer", ""))) for row in rows) / len(rows),
            "mean_prediction_tokens_complex": (
                sum(token_length(row.get("prediction", "")) for row in complex_rows) / len(complex_rows)
                if complex_rows else None
            ),
            "mean_gold_tokens_complex": (
                sum(token_length(row.get("gold", row.get("answer", ""))) for row in complex_rows) / len(complex_rows)
                if complex_rows else None
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "answer_unit_metrics.json"
    details_path = args.output_dir / "answer_unit_details.jsonl"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with details_path.open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Details: {details_path.resolve()}")


if __name__ == "__main__":
    main()
