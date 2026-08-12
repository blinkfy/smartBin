"""Summarize the three-seed CE+Focal control for the LMF baseline."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "negative_log_likelihood",
    "multiclass_brier_score",
    "ece",
)


def _read_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    metrics = report["metrics"]
    if int(metrics["sample_count"]) != 826:
        raise ValueError(f"unexpected sample count: {path}")
    return {name: float(metrics[name]) for name in METRICS}


def _summarize(runs: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        values = [run[metric] for run in runs]
        result[f"{metric}_mean"] = mean(values)
        result[f"{metric}_sd"] = stdev(values)
    return result


def main() -> None:
    output = (
        ROOT
        / "reports"
        / "qwen-material"
        / "standard-fusion-baselines-focal"
        / "lmf"
    )
    focal_runs: list[dict[str, float]] = []
    ce_runs: list[dict[str, float]] = []
    paired: list[dict[str, Any]] = []
    for seed in SEEDS:
        focal_path = output / f"seed{seed}" / "test" / "evaluation.json"
        ce_path = (
            ROOT
            / "reports"
            / "qwen-material"
            / "standard-fusion-baselines"
            / "lmf"
            / f"seed{seed}"
            / "test"
            / "evaluation.json"
        )
        focal = _read_metrics(focal_path)
        ce = _read_metrics(ce_path)
        focal_runs.append(focal)
        ce_runs.append(ce)
        paired.append(
            {
                "seed": seed,
                "accuracy_difference_pp": 100.0 * (focal["accuracy"] - ce["accuracy"]),
                "macro_f1_difference_pp": 100.0 * (focal["macro_f1"] - ce["macro_f1"]),
            }
        )

    focal_summary = _summarize(focal_runs)
    ce_summary = _summarize(ce_runs)
    accuracy_differences = [row["accuracy_difference_pp"] for row in paired]
    f1_differences = [row["macro_f1_difference_pp"] for row in paired]
    payload = {
        "valid": True,
        "seeds": list(SEEDS),
        "objective_control": "LMF cross-entropy plus unit-weight focal loss",
        "reference": "LMF cross-entropy only",
        "focal": focal_summary,
        "cross_entropy": ce_summary,
        "paired_by_seed": paired,
        "paired_summary": {
            "accuracy_difference_pp_mean": mean(accuracy_differences),
            "accuracy_difference_pp_sd": stdev(accuracy_differences),
            "macro_f1_difference_pp_mean": mean(f1_differences),
            "macro_f1_difference_pp_sd": stdev(f1_differences),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "SUMMARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# LMF Objective-Matched Control",
        "",
        "| Objective | Accuracy (%) | Macro F1 (%) | NLL | Brier | ECE (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (
        ("Cross-entropy", ce_summary),
        ("Cross-entropy + focal", focal_summary),
    ):
        lines.append(
            f"| {label} | {100 * summary['accuracy_mean']:.2f} +/- "
            f"{100 * summary['accuracy_sd']:.2f} | "
            f"{100 * summary['macro_f1_mean']:.2f} +/- "
            f"{100 * summary['macro_f1_sd']:.2f} | "
            f"{summary['negative_log_likelihood_mean']:.3f} | "
            f"{summary['multiclass_brier_score_mean']:.3f} | "
            f"{100 * summary['ece_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Paired CE+Focal minus CE differences: "
            f"Accuracy {mean(accuracy_differences):+.2f} +/- "
            f"{stdev(accuracy_differences):.2f} pp; Macro F1 "
            f"{mean(f1_differences):+.2f} +/- {stdev(f1_differences):.2f} pp.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired_summary"], indent=2))


if __name__ == "__main__":
    main()
