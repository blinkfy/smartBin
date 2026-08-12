"""Summarize mixed-fault path interventions without pseudoreplication."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_fine_grained_fault_queue import FAULT_VARIANTS, PERTURBATION_SEEDS
from run_fine_grained_intervention_queue import INTERVENTIONS, OUTPUT_ROOT, SEEDS


METRICS = {
    "accuracy": "accuracy",
    "macro_f1": "macro_f1",
    "nll": "negative_log_likelihood",
    "brier": "multiclass_brier_score",
    "ece": "ece",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _training_seed_means(variant: str, metric: str) -> np.ndarray:
    values = []
    source_name = METRICS[metric]
    for training_seed in SEEDS:
        perturbation_values = []
        for perturbation_seed in PERTURBATION_SEEDS:
            path = (
                OUTPUT_ROOT / "faults" / variant
                / f"training-seed{training_seed}" / "seeds"
                / f"seed-{perturbation_seed}" / "mixed" / "evaluation.json"
            )
            report = _read_json(path)
            if report.get("metrics", {}).get("sample_count") != 826:
                raise ValueError(f"incomplete mixed-fault evaluation: {path}")
            perturbation_values.append(float(report["metrics"][source_name]))
        values.append(float(np.mean(perturbation_values)))
    return np.asarray(values, dtype=np.float64)


def summarize(output_root: Path = OUTPUT_ROOT / "faults") -> dict[str, Any]:
    full = {
        metric: _training_seed_means("full", metric) for metric in METRICS
    }
    rows = []
    for variant in FAULT_VARIANTS:
        vectors = {
            metric: _training_seed_means(variant, metric) for metric in METRICS
        }
        row: dict[str, Any] = {
            "variant": variant,
            "label": (
                "Full model" if variant == "full"
                else INTERVENTIONS[variant]["label"]
            ),
            "training_seed_count": len(SEEDS),
            "perturbation_seed_count_per_training_seed": len(PERTURBATION_SEEDS),
        }
        for metric, values in vectors.items():
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd_across_training_seeds"] = float(
                values.std(ddof=1)
            )
            row[f"{metric}_training_seed_means"] = {
                str(seed): float(value) for seed, value in zip(SEEDS, values)
            }
        for metric in ("accuracy", "macro_f1"):
            delta = 100.0 * (full[metric] - vectors[metric])
            row[f"full_minus_variant_{metric}_pp_mean"] = float(delta.mean())
            row[f"full_minus_variant_{metric}_pp_sd"] = float(
                delta.std(ddof=1)
            )
        rows.append(row)

    result = {
        "schema_version": 1,
        "valid": True,
        "scenario": "mixed",
        "training_seeds": list(SEEDS),
        "perturbation_seeds": list(PERTURBATION_SEEDS),
        "aggregation": (
            "First average five matched perturbation seeds within each training "
            "seed, then report mean and sample SD across three training seeds."
        ),
        "rows": rows,
    }
    _atomic_text(
        output_root / "SUMMARY.json",
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )

    fields = [
        "variant",
        "label",
        "accuracy_mean",
        "accuracy_sd_across_training_seeds",
        "macro_f1_mean",
        "macro_f1_sd_across_training_seeds",
        "full_minus_variant_accuracy_pp_mean",
        "full_minus_variant_accuracy_pp_sd",
        "full_minus_variant_macro_f1_pp_mean",
        "full_minus_variant_macro_f1_pp_sd",
        "nll_mean",
        "brier_mean",
        "ece_mean",
    ]
    csv_path = output_root / "SUMMARY.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fields})
    temporary.replace(csv_path)

    lines = [
        "# Mixed-fault three-training-seed intervention summary",
        "",
        (
            "Each training seed is averaged over five matched perturbation "
            "seeds before the three-training-seed mean and SD are computed."
        ),
        "",
        "| Configuration | Accuracy (%) | Macro F1 (%) | Full - variant Acc. (pp) | Full - variant F1 (pp) | NLL | Brier | ECE (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {acc:.2f} +/- {acc_sd:.2f} | {f1:.2f} +/- {f1_sd:.2f} | "
            "{da:+.2f} | {df:+.2f} | {nll:.3f} | {brier:.3f} | {ece:.2f} |".format(
                label=row["label"],
                acc=100.0 * row["accuracy_mean"],
                acc_sd=100.0 * row["accuracy_sd_across_training_seeds"],
                f1=100.0 * row["macro_f1_mean"],
                f1_sd=100.0 * row["macro_f1_sd_across_training_seeds"],
                da=row["full_minus_variant_accuracy_pp_mean"],
                df=row["full_minus_variant_macro_f1_pp_mean"],
                nll=row["nll_mean"],
                brier=row["brier_mean"],
                ece=100.0 * row["ece_mean"],
            )
        )
    _atomic_text(output_root / "SUMMARY.md", "\n".join(lines) + "\n")
    return result


def main() -> None:
    result = summarize()
    print(
        json.dumps(
            {"valid": result["valid"], "rows": len(result["rows"])},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
