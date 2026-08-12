"""Summarize the three-seed predicted-grid stop-gradient control."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
METRICS = (
    "accuracy",
    "macro_f1",
    "negative_log_likelihood",
    "multiclass_brier_score",
    "ece",
)


def _read(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    metrics = payload["metrics"]
    if int(metrics["sample_count"]) != 826:
        raise ValueError(f"unexpected sample count: {path}")
    return {name: float(metrics[name]) for name in METRICS}


def _aggregate(runs: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        values = [run[metric] for run in runs]
        result[f"{metric}_mean"] = mean(values)
        result[f"{metric}_sd"] = stdev(values)
    return result


def main() -> None:
    report_root = (
        ROOT
        / "reports"
        / "qwen-material"
        / "confirmatory-ablations"
        / "no-stop-gradient"
    )
    no_stop_runs = [
        _read(report_root / f"seed{seed}" / "test" / "evaluation.json")
        for seed in SEEDS
    ]
    main_paths = {
        42: ROOT
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "test"
        / "evaluation.json",
        43: ROOT
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "seed-replication"
        / "seed43"
        / "test"
        / "evaluation.json",
        44: ROOT
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "seed-replication"
        / "seed44"
        / "test"
        / "evaluation.json",
    }
    stop_runs = [_read(main_paths[seed]) for seed in SEEDS]
    paired = []
    for seed, no_stop, stop in zip(SEEDS, no_stop_runs, stop_runs):
        paired.append(
            {
                "seed": seed,
                "no_stop_minus_stop_accuracy_pp": 100.0
                * (no_stop["accuracy"] - stop["accuracy"]),
                "no_stop_minus_stop_macro_f1_pp": 100.0
                * (no_stop["macro_f1"] - stop["macro_f1"]),
            }
        )
    accuracy_differences = [
        row["no_stop_minus_stop_accuracy_pp"] for row in paired
    ]
    f1_differences = [
        row["no_stop_minus_stop_macro_f1_pp"] for row in paired
    ]
    payload = {
        "valid": True,
        "seeds": list(SEEDS),
        "control": (
            "reliability evidence gradients propagate into the predicted-grid "
            "decoder"
        ),
        "reference": (
            "predicted grid is detached before reliability evidence computation"
        ),
        "no_stop_gradient": _aggregate(no_stop_runs),
        "stop_gradient": _aggregate(stop_runs),
        "paired_by_seed": paired,
        "paired_summary": {
            "accuracy_difference_pp_mean": mean(accuracy_differences),
            "accuracy_difference_pp_sd": stdev(accuracy_differences),
            "macro_f1_difference_pp_mean": mean(f1_differences),
            "macro_f1_difference_pp_sd": stdev(f1_differences),
        },
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "SUMMARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["paired_summary"], indent=2))


if __name__ == "__main__":
    main()
