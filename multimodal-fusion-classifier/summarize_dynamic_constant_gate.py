"""Summarize matched dynamic- and constant-gate checkpoints across three seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    ROOT / "reports" / "qwen-material" / "submission-revision" / "dynamic-vs-constant-gate"
)
SEEDS = (42, 43, 44)
FAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)
VARIANTS = ("dynamic", "constant")
SCENARIOS = ("clean", "missing_100", "noise_015", "bias_008", "stuck", "mixed")
DYNAMIC_CHECKPOINTS = {
    42: ROOT / "checkpoints" / "ablations" / "full_cross_modal_quality" / "best.pt",
    43: ROOT
    / "checkpoints"
    / "ablations"
    / "v3_seed_replicates"
    / "full_cross_modal_quality_seed43"
    / "best.pt",
    44: ROOT
    / "checkpoints"
    / "ablations"
    / "v3_seed_replicates"
    / "full_cross_modal_quality_seed44"
    / "best.pt",
}
CONSTANT_CHECKPOINTS = {
    seed: ROOT
    / "checkpoints"
    / "submission-revision"
    / "constant-gate-matched"
    / f"seed{seed}"
    / "best.pt"
    for seed in SEEDS
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    return parser.parse_args()


def scenario_probability_metrics(
    robustness_root: Path,
    scenario: str,
) -> dict[str, float]:
    nll_values = []
    brier_values = []
    for fault_seed in FAULT_SEEDS:
        evaluation = load_json(
            robustness_root
            / "seeds"
            / f"seed-{fault_seed}"
            / scenario
            / "evaluation.json"
        )
        nll_values.append(
            float(evaluation["metrics"]["negative_log_likelihood"])
        )
        brier_values.append(
            float(evaluation["metrics"]["multiclass_brier_score"])
        )
    return {
        "nll": mean(nll_values),
        "brier": mean(brier_values),
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            path = input_root / f"seed{seed}" / variant / "robustness" / "robustness_multiseed_summary.json"
            report = load_json(path)
            by_scenario = {row["scenario"]: row for row in report["summary"]}
            robustness_root = path.parent
            for scenario in SCENARIOS:
                source = by_scenario[scenario]
                probability_metrics = scenario_probability_metrics(
                    robustness_root,
                    scenario,
                )
                rows.append(
                    {
                        "training_seed": seed,
                        "variant": variant,
                        "scenario": scenario,
                        "accuracy": float(source["accuracy_mean"]),
                        "macro_f1": float(source["macro_f1_mean"]),
                        "nll": probability_metrics["nll"],
                        "brier": probability_metrics["brier"],
                        "ece": float(source["ece_mean"]),
                        "sensor_reliability": float(source["sensor_reliability_mean"]),
                    }
                )

    summary: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario and row["variant"] == variant
            ]
            result: dict[str, Any] = {
                "scenario": scenario,
                "variant": variant,
                "training_seed_count": len(selected),
            }
            for metric in (
                "accuracy",
                "macro_f1",
                "nll",
                "brier",
                "ece",
                "sensor_reliability",
            ):
                values = [float(row[metric]) for row in selected]
                result[f"{metric}_mean"] = mean(values)
                result[f"{metric}_sample_std"] = stdev(values)
            summary.append(result)

    contrasts: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        differences: dict[str, list[float]] = {
            "accuracy_pp": [],
            "macro_f1_pp": [],
            "nll_reduction": [],
            "brier_reduction": [],
            "ece_improvement_pp": [],
        }
        for seed in SEEDS:
            dynamic = next(
                row
                for row in rows
                if row["scenario"] == scenario
                and row["training_seed"] == seed
                and row["variant"] == "dynamic"
            )
            constant = next(
                row
                for row in rows
                if row["scenario"] == scenario
                and row["training_seed"] == seed
                and row["variant"] == "constant"
            )
            differences["accuracy_pp"].append(
                100.0 * (dynamic["accuracy"] - constant["accuracy"])
            )
            differences["macro_f1_pp"].append(
                100.0 * (dynamic["macro_f1"] - constant["macro_f1"])
            )
            differences["nll_reduction"].append(
                constant["nll"] - dynamic["nll"]
            )
            differences["brier_reduction"].append(
                constant["brier"] - dynamic["brier"]
            )
            differences["ece_improvement_pp"].append(
                100.0 * (constant["ece"] - dynamic["ece"])
            )
        contrast: dict[str, Any] = {"scenario": scenario}
        for metric, values in differences.items():
            contrast[f"{metric}_mean"] = mean(values)
            contrast[f"{metric}_sample_std"] = stdev(values)
        contrasts.append(contrast)

    checkpoint_provenance = []
    for seed in SEEDS:
        for variant, paths in (
            ("dynamic", DYNAMIC_CHECKPOINTS),
            ("constant", CONSTANT_CHECKPOINTS),
        ):
            checkpoint_path = paths[seed].resolve()
            clean_evaluation = load_json(
                input_root / f"seed{seed}" / variant / "test" / "evaluation.json"
            )
            checkpoint_provenance.append(
                {
                    "training_seed": seed,
                    "variant": variant,
                    "path": str(checkpoint_path),
                    "sha256": sha256(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "epoch": int(clean_evaluation["checkpoint"]["epoch"]),
                }
            )

    report = {
        "valid": True,
        "training_seeds": list(SEEDS),
        "fault_seeds": list(FAULT_SEEDS),
        "protocol": {
            "batch_size": 16,
            "evaluation_num_workers": 0,
            "training_num_workers": 4,
            "training_order_control": (
                "four-worker prefetch with the audited single-worker-order sampler"
            ),
            "training_execution": (
                "training seeds may run as independent concurrent processes; "
                "each seed retains its own unchanged configuration and RNG state"
            ),
            "matched_difference": (
                "sample-adaptive reliability versus one learned constant "
                "reliability value"
            ),
            "availability_rule": "r=0 under missingness for both variants",
            "detailed_artifact_root": str(input_root),
        },
        "summary": summary,
        "paired_dynamic_minus_constant": contrasts,
        "per_training_seed": rows,
        "checkpoint_provenance": checkpoint_provenance,
        "interpretation_rule": (
            "If dynamic gating does not improve classification, attribute primary "
            "robustness to fallback and use the dynamic score only as a fault-sensitive ordering."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output_root / "paired_dynamic_minus_constant.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    with (output_root / "per_training_seed.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
