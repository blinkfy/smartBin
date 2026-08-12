"""Audit clean reliability-ablation predictions under one evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNS = {
    42: {
        "full": "seed42/full/clean",
        "no_reliability": "seed42/no_reliability/clean",
    },
    43: {
        "full": "seed43/full/clean",
        "no_reliability": "seed43/no_reliability/clean",
    },
    44: {
        "full": "seed44/full/clean",
        "no_reliability": "seed44/no_reliability/clean",
    },
}
LEGACY_RUNS = {
    42: {
        "full": "reports/qwen-material/full_cross_modal_quality/robustness/seeds/seed-2026/clean",
        "no_reliability": "reports/qwen-material/confirmatory-ablations/v3-no-reliability-workers0/seed42/robustness/seeds/seed-2026/clean",
    },
    43: {
        "full": "reports/qwen-material/reliability-gate-diagnostic/training-seeds/seed43/raw/seeds/seed-2026/clean",
        "no_reliability": "reports/qwen-material/confirmatory-ablations/v3-no-reliability-workers0/seed43/robustness/seeds/seed-2026/clean",
    },
    44: {
        "full": "reports/qwen-material/reliability-gate-diagnostic/training-seeds/seed44/raw/seeds/seed-2026/clean",
        "no_reliability": "reports/qwen-material/confirmatory-ablations/v3-no-reliability-workers0/seed44/robustness/seeds/seed-2026/clean",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "reports"
        / "qwen-material"
        / "submission-revision"
        / "clean-protocol-audit",
    )
    parser.add_argument("--expected-batch-size", type=int, default=16)
    parser.add_argument("--expected-num-workers", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows[str(record["sample_id"])] = record
    return rows


def compare_predictions(canonical_path: Path, legacy_path: Path) -> dict[str, Any]:
    canonical = load_predictions(canonical_path)
    legacy = load_predictions(legacy_path)
    if canonical.keys() != legacy.keys():
        raise ValueError("canonical and legacy prediction sample IDs differ")
    absolute_differences: list[float] = []
    flipped_samples: list[dict[str, Any]] = []
    for sample_id in sorted(canonical):
        current = canonical[sample_id]
        previous = legacy[sample_id]
        current_probabilities = current["probabilities"]
        previous_probabilities = previous["probabilities"]
        if len(current_probabilities) != len(previous_probabilities):
            raise ValueError(f"probability width differs for {sample_id}")
        absolute_differences.extend(
            abs(float(left) - float(right))
            for left, right in zip(current_probabilities, previous_probabilities)
        )
        if int(current["predicted_id"]) != int(previous["predicted_id"]):
            flipped_samples.append(
                {
                    "sample_id": sample_id,
                    "target_id": int(current["target_id"]),
                    "canonical_predicted_id": int(current["predicted_id"]),
                    "legacy_predicted_id": int(previous["predicted_id"]),
                    "canonical_confidence": float(current["confidence"]),
                    "legacy_confidence": float(previous["confidence"]),
                }
            )
    return {
        "sample_count": len(canonical),
        "class_flip_count": len(flipped_samples),
        "class_flips": flipped_samples,
        "probability_absolute_difference_mean": mean(absolute_differences),
        "probability_absolute_difference_max": max(absolute_differences),
    }


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    rows: list[dict[str, Any]] = []
    legacy_effects: list[dict[str, float]] = []
    errors: list[str] = []
    checkpoint_hashes: dict[Path, str] = {}
    for seed, variants in RUNS.items():
        loaded: dict[str, dict[str, Any]] = {}
        legacy_loaded: dict[str, dict[str, Any]] = {}
        for variant, relative in variants.items():
            directory = root / relative
            evaluation_path = directory / "evaluation.json"
            predictions_path = directory / "predictions.jsonl"
            evaluation = load_json(evaluation_path)
            protocol = evaluation.get("evaluation_config", {})
            if int(protocol.get("batch_size", -1)) != args.expected_batch_size:
                errors.append(f"seed {seed} {variant}: unexpected batch size")
            if int(protocol.get("num_workers", -1)) != args.expected_num_workers:
                errors.append(f"seed {seed} {variant}: unexpected worker count")
            metrics = evaluation["metrics"]
            checkpoint_path = Path(evaluation["checkpoint"]["path"]).resolve()
            if checkpoint_path not in checkpoint_hashes:
                checkpoint_hashes[checkpoint_path] = sha256(checkpoint_path)
            loaded[variant] = {
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "nll": float(metrics["negative_log_likelihood"]),
                "brier": float(metrics["multiclass_brier_score"]),
                "ece": float(metrics["ece"]),
                "checkpoint": evaluation["checkpoint"],
                "checkpoint_sha256": checkpoint_hashes[checkpoint_path],
                "evaluation_path": str(evaluation_path),
                "predictions_path": str(predictions_path),
                "predictions_sha256": sha256(predictions_path),
            }
            legacy_directory = ROOT / LEGACY_RUNS[seed][variant]
            legacy_evaluation_path = legacy_directory / "evaluation.json"
            legacy_predictions_path = legacy_directory / "predictions.jsonl"
            legacy_evaluation = load_json(legacy_evaluation_path)
            legacy_metrics = legacy_evaluation["metrics"]
            legacy_checkpoint_path = Path(
                legacy_evaluation["checkpoint"]["path"]
            ).resolve()
            if legacy_checkpoint_path not in checkpoint_hashes:
                checkpoint_hashes[legacy_checkpoint_path] = sha256(
                    legacy_checkpoint_path
                )
            legacy_loaded[variant] = {
                "accuracy": float(legacy_metrics["accuracy"]),
                "macro_f1": float(legacy_metrics["macro_f1"]),
                "ece": float(legacy_metrics["ece"]),
                "evaluation_config": legacy_evaluation.get(
                    "evaluation_config", {}
                ),
                "evaluation_path": str(legacy_evaluation_path.resolve()),
                "predictions_path": str(legacy_predictions_path.resolve()),
                "predictions_sha256": sha256(legacy_predictions_path),
                "checkpoint": legacy_evaluation["checkpoint"],
                "checkpoint_sha256": checkpoint_hashes[legacy_checkpoint_path],
                "same_checkpoint_path": checkpoint_path == legacy_checkpoint_path,
                "same_checkpoint_sha256": (
                    checkpoint_hashes[checkpoint_path]
                    == checkpoint_hashes[legacy_checkpoint_path]
                ),
                "prediction_comparison": compare_predictions(
                    predictions_path,
                    legacy_predictions_path,
                ),
            }
        row = {
            "training_seed": seed,
            "full": loaded["full"],
            "no_reliability": loaded["no_reliability"],
            "effect_pp": {
                "accuracy": 100.0
                * (loaded["full"]["accuracy"] - loaded["no_reliability"]["accuracy"]),
                "macro_f1": 100.0
                * (loaded["full"]["macro_f1"] - loaded["no_reliability"]["macro_f1"]),
                "ece_improvement": 100.0
                * (loaded["no_reliability"]["ece"] - loaded["full"]["ece"]),
            },
            "legacy_runtime_trace": legacy_loaded,
        }
        rows.append(row)
        legacy_effects.append(
            {
                "training_seed": seed,
                "accuracy": 100.0
                * (
                    legacy_loaded["full"]["accuracy"]
                    - legacy_loaded["no_reliability"]["accuracy"]
                ),
                "macro_f1": 100.0
                * (
                    legacy_loaded["full"]["macro_f1"]
                    - legacy_loaded["no_reliability"]["macro_f1"]
                ),
                "ece_improvement": 100.0
                * (
                    legacy_loaded["no_reliability"]["ece"]
                    - legacy_loaded["full"]["ece"]
                ),
            }
        )

    summary: dict[str, dict[str, float]] = {}
    for metric in ("accuracy", "macro_f1", "ece_improvement"):
        values = [float(row["effect_pp"][metric]) for row in rows]
        summary[metric] = {"mean_pp": mean(values), "sample_std_pp": stdev(values)}
    legacy_summary: dict[str, dict[str, float]] = {}
    for metric in ("accuracy", "macro_f1", "ece_improvement"):
        values = [float(row[metric]) for row in legacy_effects]
        legacy_summary[metric] = {
            "mean_pp": mean(values),
            "sample_std_pp": stdev(values),
        }
    report = {
        "valid": not errors,
        "purpose": "Unify Table VI and the clean row of Table IX.",
        "protocol": {
            "batch_size": args.expected_batch_size,
            "num_workers": args.expected_num_workers,
            "split": "test",
            "sensor_condition": "clean",
        },
        "per_seed": rows,
        "summary": summary,
        "legacy_table_ix_summary": legacy_summary,
        "legacy_table_ix_per_seed": legacy_effects,
        "root_cause": (
            "The old robustness clean rows mixed evaluation runtimes: seed 42 "
            "used batch_size=16 and num_workers=0, whereas seed 43/44 used "
            "batch_size=64 and num_workers=4. The checkpoint files are identical; "
            "small CUDA probability differences changed one seed-43 ablation "
            "class decision. Both tables now use the canonical batch-16, "
            "zero-worker prediction archives."
        ),
        "statistics_script": str(
            (ROOT / "summarize_three_training_seed_robustness.py").resolve()
        ),
        "errors": errors,
    }
    output = root / "CLEAN_ABLATION_PROTOCOL_AUDIT.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
