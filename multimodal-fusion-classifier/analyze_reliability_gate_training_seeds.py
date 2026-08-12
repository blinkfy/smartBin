"""Aggregate gate diagnostics hierarchically across trained checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from mmfc.reliability_gate_diagnostics import (
    DEFAULT_GATE_SCENARIOS,
    compute_gate_diagnostics,
    load_gate_predictions,
    write_gate_figure,
)


ROOT = Path(__file__).resolve().parent
FAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)
INPUTS = {
    42: ROOT / "reports" / "qwen-material" / "reliability-gate-diagnostic" / "raw",
    43: ROOT / "reports" / "qwen-material" / "reliability-gate-diagnostic" / "training-seeds" / "seed43" / "raw",
    44: ROOT / "reports" / "qwen-material" / "reliability-gate-diagnostic" / "training-seeds" / "seed44" / "raw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports"
            / "qwen-material"
            / "reliability-gate-diagnostic"
            / "training-seeds"
            / "summary"
        ),
    )
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "std_across_training_seeds": stdev(values),
        "values_by_training_seed": {
            str(seed): value for seed, value in zip(INPUTS, values)
        },
    }


def main() -> None:
    args = parse_args()
    reports: dict[int, dict[str, Any]] = {}
    for training_seed, root in INPUTS.items():
        predictions = load_gate_predictions(
            root,
            seeds=FAULT_SEEDS,
            scenarios=DEFAULT_GATE_SCENARIOS,
        )
        reports[training_seed] = compute_gate_diagnostics(
            predictions,
            scenarios=DEFAULT_GATE_SCENARIOS,
        )

    distribution = []
    distribution_values: dict[str, list[float]] = {}
    for scenario in DEFAULT_GATE_SCENARIOS:
        rows = {
            seed: next(
                row
                for row in report["distribution_summary"]
                if row["scenario"] == scenario
            )
            for seed, report in reports.items()
        }
        distribution.append(
            {
                "scenario": scenario,
                "mean": mean(row["mean"] for row in rows.values()),
                "std_across_training_seeds": stdev(
                    row["mean"] for row in rows.values()
                ),
                "median_mean": mean(row["median"] for row in rows.values()),
                "q1_mean": mean(row["q1"] for row in rows.values()),
                "q3_mean": mean(row["q3"] for row in rows.values()),
                "training_seed_values": {
                    str(seed): row["mean"] for seed, row in rows.items()
                },
            }
        )
        distribution_values[scenario] = [
            value
            for report in reports.values()
            for value in report["distribution_values"][scenario]
        ]

    detection = []
    for scenario in DEFAULT_GATE_SCENARIOS[1:]:
        rows = {
            seed: next(
                row
                for row in report["fault_detection"]
                if row["scenario"] == scenario
            )
            for seed, report in reports.items()
        }
        detection.append(
            {
                "scenario": scenario,
                "auroc": mean_std([row["auroc"]["mean"] for row in rows.values()]),
                "auprc": mean_std([row["auprc"]["mean"] for row in rows.values()]),
                "effective_fault_fraction": mean_std(
                    [row["effective_fault_fraction"]["mean"] for row in rows.values()]
                ),
            }
        )

    macro_available = {
        metric: mean_std(
            [
                report["macro_detection_available_faults"][metric]
                for report in reports.values()
            ]
        )
        for metric in ("auroc", "auprc")
    }
    macro_all = {
        metric: mean_std(
            [report["macro_detection_all_faults"][metric] for report in reports.values()]
        )
        for metric in ("auroc", "auprc")
    }
    severity = {
        family: mean_std(
            [
                report["severity_correlations"][family]["mean"]
                for report in reports.values()
            ]
        )
        for family in ("noise", "bias")
    }
    report = {
        "valid": True,
        "training_seeds": list(INPUTS),
        "training_seed_count": len(INPUTS),
        "fault_seeds": list(FAULT_SEEDS),
        "fault_seed_count_per_training_seed": len(FAULT_SEEDS),
        "sample_count": 826,
        "scenario_order": list(DEFAULT_GATE_SCENARIOS),
        "distribution_summary": distribution,
        "distribution_values": distribution_values,
        "fault_detection": detection,
        "macro_detection_available_faults": macro_available,
        "macro_detection_all_faults": macro_all,
        "severity_correlations": severity,
        "per_training_seed": {
            str(seed): {
                "macro_available": value["macro_detection_available_faults"],
                "macro_all": value["macro_detection_all_faults"],
                "severity_correlations": value["severity_correlations"],
            }
            for seed, value in reports.items()
        },
        "protocol": {
            "hierarchy": (
                "metrics are first averaged over five fault realizations within "
                "each trained checkpoint, then summarized over three training seeds"
            ),
            "missing_boundary": "availability a=0 deterministically forces r=0",
            "abstract_metric": "macro detection over available faults only",
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = []
    for item in detection:
        rows.append(
            {
                "scenario": item["scenario"],
                "auroc_mean": item["auroc"]["mean"],
                "auroc_training_seed_std": item["auroc"]["std_across_training_seeds"],
                "auprc_mean": item["auprc"]["mean"],
                "auprc_training_seed_std": item["auprc"]["std_across_training_seeds"],
                "effective_fault_fraction_mean": item["effective_fault_fraction"]["mean"],
            }
        )
    with (args.output / "fault_detection.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_report = {
        "scenario_order": report["scenario_order"],
        "distribution_values": distribution_values,
        "fault_detection": [
            {
                "scenario": row["scenario"],
                "auroc": {
                    "mean": row["auroc"]["mean"],
                    "std": row["auroc"]["std_across_training_seeds"],
                },
                "auprc": {
                    "mean": row["auprc"]["mean"],
                    "std": row["auprc"]["std_across_training_seeds"],
                },
            }
            for row in detection
        ],
    }
    write_gate_figure(figure_report, args.output / "reliability_gate_training_seeds.svg")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
