"""Paired source-image bootstrap comparison for two evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mmfc.evaluation import (
    compute_classification_metrics,
    grouped_paired_bootstrap_differences,
)
from mmfc.training import load_config, read_jsonl, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare aligned predictions with paired cluster bootstrap"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qwen-material/prediction-comparison.json"),
    )
    return parser.parse_args()


def _indexed_predictions(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in indexed:
            raise ValueError(f"{path} contains a missing or duplicate sample_id")
        indexed[sample_id] = record
    return indexed


def exact_mcnemar_p_value(
    candidate_only_correct: int,
    reference_only_correct: int,
) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs."""

    first = int(candidate_only_correct)
    second = int(reference_only_correct)
    if first < 0 or second < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = first + second
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(first, second) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    classes = config.get("classes", [])
    class_names = tuple(
        item["name"] if isinstance(item, dict) else str(item)
        for item in classes
    )
    if not class_names:
        raise ValueError("config classes must not be empty")
    manifest_path = resolve_project_path(
        args.manifest
        if args.manifest is not None
        else config["data"]["test_manifest"]
    )
    reference_path = resolve_project_path(args.reference)
    candidate_path = resolve_project_path(args.candidate)
    reference = _indexed_predictions(reference_path)
    candidate = _indexed_predictions(candidate_path)
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate sample_id sets differ")

    manifest = read_jsonl(manifest_path)
    ordered_ids = [
        str(record["sample_id"])
        for record in manifest
        if str(record["sample_id"]) in reference
    ]
    if len(ordered_ids) != len(reference):
        raise ValueError("manifest does not cover every prediction sample")
    source_by_id = {
        str(record["sample_id"]): str(
            record.get("source_image_id", record["sample_id"])
        )
        for record in manifest
    }
    targets = np.asarray(
        [int(reference[sample_id]["target_id"]) for sample_id in ordered_ids],
        dtype=np.int64,
    )
    candidate_targets = np.asarray(
        [int(candidate[sample_id]["target_id"]) for sample_id in ordered_ids],
        dtype=np.int64,
    )
    if not np.array_equal(targets, candidate_targets):
        raise ValueError("reference and candidate target_id values differ")
    reference_probabilities = np.asarray(
        [reference[sample_id]["probabilities"] for sample_id in ordered_ids],
        dtype=np.float64,
    )
    candidate_probabilities = np.asarray(
        [candidate[sample_id]["probabilities"] for sample_id in ordered_ids],
        dtype=np.float64,
    )
    source_ids = [source_by_id[sample_id] for sample_id in ordered_ids]

    reference_metrics = compute_classification_metrics(
        targets,
        reference_probabilities,
        class_names,
    )
    candidate_metrics = compute_classification_metrics(
        targets,
        candidate_probabilities,
        class_names,
    )
    reference_predictions = reference_probabilities.argmax(axis=1)
    candidate_predictions = candidate_probabilities.argmax(axis=1)
    reference_correct = reference_predictions == targets
    candidate_correct = candidate_predictions == targets
    paired = grouped_paired_bootstrap_differences(
        targets,
        reference_probabilities,
        candidate_probabilities,
        source_ids,
        class_names,
        repeats=args.repeats,
        confidence=args.confidence,
        seed=args.seed,
    )
    per_class = []
    for reference_row, candidate_row in zip(
        reference_metrics["per_class"],
        candidate_metrics["per_class"],
    ):
        per_class.append(
            {
                "label_id": reference_row["label_id"],
                "label": reference_row["label"],
                "support": reference_row["support"],
                "reference_f1": reference_row["f1"],
                "candidate_f1": candidate_row["f1"],
                "f1_difference": (
                    candidate_row["f1"] - reference_row["f1"]
                ),
            }
        )
    outcome_changes = {
        "both_correct": int(np.sum(reference_correct & candidate_correct)),
        "candidate_only_correct": int(
            np.sum(~reference_correct & candidate_correct)
        ),
        "reference_only_correct": int(
            np.sum(reference_correct & ~candidate_correct)
        ),
        "both_wrong": int(np.sum(~reference_correct & ~candidate_correct)),
    }
    result = {
        "reference": {
            "name": args.reference_name,
            "path": str(reference_path),
            "metrics": reference_metrics["metrics"],
        },
        "candidate": {
            "name": args.candidate_name,
            "path": str(candidate_path),
            "metrics": candidate_metrics["metrics"],
        },
        "manifest": str(manifest_path),
        "outcome_changes": outcome_changes,
        "mcnemar_exact": {
            "discordant_pairs": (
                outcome_changes["candidate_only_correct"]
                + outcome_changes["reference_only_correct"]
            ),
            "p_value": exact_mcnemar_p_value(
                outcome_changes["candidate_only_correct"],
                outcome_changes["reference_only_correct"],
            ),
        },
        "paired_bootstrap": paired,
        "per_class": per_class,
    }
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "outcome_changes": result["outcome_changes"],
                "mcnemar_exact": result["mcnemar_exact"],
                "paired_bootstrap": paired,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
