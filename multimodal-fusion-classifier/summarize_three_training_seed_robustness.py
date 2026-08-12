"""Aggregate matched robustness effects across three trained model seeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parent
TRAINING_SEEDS = (42, 43, 44)
FAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)
SCENARIOS = (
    "clean",
    "missing_100",
    "noise_015",
    "bias_008",
    "stuck",
    "mixed",
)
METRICS = (
    ("accuracy_mean", "full_minus_ablation_accuracy_pp"),
    ("macro_f1_mean", "full_minus_ablation_macro_f1_pp"),
    ("ece_mean", "ablation_minus_full_ece_pp"),
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("seeds") != list(FAULT_SEEDS) or payload.get("seed_count") != 5:
        raise ValueError(f"unexpected fault-seed protocol: {path}")
    rows = {row["scenario"]: row for row in payload.get("summary", [])}
    missing = [scenario for scenario in SCENARIOS if scenario not in rows]
    if missing:
        raise ValueError(f"missing scenarios in {path}: {missing}")
    return rows


def _paths(seed: int) -> tuple[Path, Path]:
    if seed == 42:
        full = (
            ROOT
            / "reports"
            / "qwen-material"
            / "full_cross_modal_quality"
            / "robustness"
            / "robustness_multiseed_summary.json"
        )
    else:
        full = (
            ROOT
            / "reports"
            / "qwen-material"
            / "reliability-gate-diagnostic"
            / "training-seeds"
            / f"seed{seed}"
            / "raw"
            / "robustness_multiseed_summary.json"
        )
    ablation = (
        ROOT
        / "reports"
        / "qwen-material"
        / "confirmatory-ablations"
        / "v3-no-reliability-workers0"
        / f"seed{seed}"
        / "robustness"
        / "robustness_multiseed_summary.json"
    )
    return full, ablation


def _clean_protocol_rows(seed: int) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    root = (
        ROOT
        / "reports"
        / "qwen-material"
        / "submission-revision"
        / "clean-protocol-audit"
        / f"seed{seed}"
    )
    paths = {
        "full": root / "full" / "clean" / "evaluation.json",
        "ablation": root / "no_reliability" / "clean" / "evaluation.json",
    }
    values: dict[str, dict[str, float]] = {}
    for variant, path in paths.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        protocol = report.get("evaluation_config", {})
        if protocol.get("batch_size") != 16 or protocol.get("num_workers") != 0:
            raise ValueError(f"unexpected clean evaluation protocol: {path}")
        metrics = report["metrics"]
        values[variant] = {
            "accuracy_mean": float(metrics["accuracy"]),
            "macro_f1_mean": float(metrics["macro_f1"]),
            "ece_mean": float(metrics["ece"]),
        }
    return (
        values["full"],
        values["ablation"],
        {name: path.relative_to(ROOT).as_posix() for name, path in paths.items()},
    )


def _aggregate() -> dict[str, Any]:
    per_seed: dict[int, list[dict[str, Any]]] = {}
    source_paths: dict[str, dict[str, str]] = {}
    for seed in TRAINING_SEEDS:
        full_path, ablation_path = _paths(seed)
        full = _read(full_path)
        ablation = _read(ablation_path)
        source_paths[str(seed)] = {
            "full": full_path.relative_to(ROOT).as_posix(),
            "ablation": ablation_path.relative_to(ROOT).as_posix(),
        }
        clean_full, clean_ablation, clean_paths = _clean_protocol_rows(seed)
        source_paths[str(seed)]["clean_full"] = clean_paths["full"]
        source_paths[str(seed)]["clean_ablation"] = clean_paths["ablation"]
        rows: list[dict[str, Any]] = []
        for scenario in SCENARIOS:
            row: dict[str, Any] = {"scenario": scenario, "training_seed": seed}
            for source_metric, effect_metric in METRICS:
                if scenario == "clean":
                    full_value = float(clean_full[source_metric])
                    ablation_value = float(clean_ablation[source_metric])
                else:
                    full_value = float(full[scenario][source_metric])
                    ablation_value = float(ablation[scenario][source_metric])
                row[f"full_{source_metric}"] = full_value
                row[f"ablation_{source_metric}"] = ablation_value
                scale = 100.0
                if source_metric == "ece_mean":
                    effect = (ablation_value - full_value) * scale
                else:
                    effect = (full_value - ablation_value) * scale
                row[effect_metric] = effect
            rows.append(row)
        per_seed[seed] = rows

    aggregate_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        seed_rows = [per_seed[seed][scenario_index] for seed in TRAINING_SEEDS]
        row = {"scenario": scenario}
        numeric_keys = [key for key in seed_rows[0] if key not in {"scenario", "training_seed"}]
        for key in numeric_keys:
            values = [float(seed_row[key]) for seed_row in seed_rows]
            row[key] = mean(values)
            row[f"{key}_sd"] = stdev(values)
        aggregate_rows.append(row)

    return {
        "valid": True,
        "training_seeds": list(TRAINING_SEEDS),
        "training_seed_count": len(TRAINING_SEEDS),
        "fault_seeds": list(FAULT_SEEDS),
        "fault_seed_count_per_training_seed": len(FAULT_SEEDS),
        "aggregation": (
            "Each trained model is first averaged over matched fault seeds; "
            "the reported mean and sample SD are then computed across training seeds. "
            "The deterministic clean row is taken from the standard batch-16, "
            "zero-worker test predictions used by the matched retraining table."
        ),
        "rows": aggregate_rows,
        "per_training_seed": {str(seed): rows for seed, rows in per_seed.items()},
        "source_paths": source_paths,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = {
        "clean": "Clean",
        "missing_100": "Missing 100%",
        "noise_015": "Gaussian noise, sigma=0.15",
        "bias_008": "Directional bias, amplitude 0.08",
        "stuck": "Stuck grid",
        "mixed": "Mixed faults",
    }
    lines = [
        "# Three-training-seed robustness comparison",
        "",
        (
            "Each trained model is averaged over five matched fault seeds before "
            "the mean and sample SD are computed across training seeds."
        ),
        "",
        "| Condition | Accuracy effect (pp) | Macro F1 effect (pp) | ECE benefit (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {labels[row['scenario']]} | "
            f"{row['full_minus_ablation_accuracy_pp']:+.2f} +/- "
            f"{row['full_minus_ablation_accuracy_pp_sd']:.2f} | "
            f"{row['full_minus_ablation_macro_f1_pp']:+.2f} +/- "
            f"{row['full_minus_ablation_macro_f1_pp_sd']:.2f} | "
            f"{row['ablation_minus_full_ece_pp']:+.2f} +/- "
            f"{row['ablation_minus_full_ece_pp_sd']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "Positive Accuracy and Macro F1 effects favor the full model. "
                "Positive ECE benefit means lower ECE for the full model."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    payload = _aggregate()
    output = (
        ROOT
        / "reports"
        / "qwen-material"
        / "confirmatory-ablations"
        / "v3-no-reliability-workers0"
        / "THREE_TRAINING_SEED_ROBUSTNESS_COMPARISON.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output.with_suffix(".csv"), payload["rows"])
    _write_markdown(output.with_suffix(".md"), payload["rows"])
    print(
        json.dumps(
            {
                "valid": payload["valid"],
                "training_seed_count": payload["training_seed_count"],
                "scenario_count": len(payload["rows"]),
                "output": output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
