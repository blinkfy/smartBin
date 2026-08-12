"""Aggregate the independently trained three-seed modality matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mmfc.training import PROJECT_ROOT
from prepare_modality_baselines import SEEDS, VARIANTS


OUTPUT_ROOT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "modality-baselines"
)
FULL_REPORTS = {
    42: PROJECT_ROOT / "reports" / "qwen-material"
    / "full_cross_modal_quality" / "test" / "evaluation.json",
    43: PROJECT_ROOT / "reports" / "qwen-material"
    / "full_cross_modal_quality" / "seed-replication" / "seed43"
    / "test" / "evaluation.json",
    44: PROJECT_ROOT / "reports" / "qwen-material"
    / "full_cross_modal_quality" / "seed-replication" / "seed44"
    / "test" / "evaluation.json",
}
METRICS = {
    "accuracy": "accuracy",
    "macro_precision": "macro_precision",
    "macro_recall": "macro_recall",
    "macro_f1": "macro_f1",
    "weighted_f1": "weighted_f1",
    "nll": "negative_log_likelihood",
    "brier": "multiclass_brier_score",
    "ece": "ece",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    if value.get("metrics", {}).get("sample_count") != 826:
        raise ValueError(f"incomplete evaluation: {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _paths(output_root: Path, variant: str) -> dict[int, Path]:
    return {
        seed: output_root / variant / f"seed{seed}" / "test" / "evaluation.json"
        for seed in SEEDS
    }


def _row(
    name: str,
    label: str,
    modalities: list[str],
    paths: Mapping[int, Path],
) -> dict[str, Any]:
    reports = {seed: _read(path) for seed, path in paths.items()}
    row: dict[str, Any] = {
        "variant": name,
        "label": label,
        "modalities": modalities,
        "seeds": list(SEEDS),
    }
    for short_name, source_name in METRICS.items():
        values = np.asarray(
            [float(reports[seed]["metrics"][source_name]) for seed in SEEDS],
            dtype=np.float64,
        )
        row[f"{short_name}_mean"] = float(values.mean())
        row[f"{short_name}_sd"] = float(values.std(ddof=1))
        row[f"{short_name}_by_seed"] = {
            str(seed): float(value) for seed, value in zip(SEEDS, values)
        }
    performance = reports[42].get("performance", {})
    row["active_parameter_count"] = performance.get("active_parameter_count")
    row["active_trainable_parameter_count"] = performance.get(
        "active_trainable_parameter_count"
    )
    return row


def summarize(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    rows = [
        _row(
            "full",
            "Full visual + text + sensor model",
            ["visual", "text", "sensor"],
            FULL_REPORTS,
        )
    ]
    for variant, specification in VARIANTS.items():
        rows.append(
            _row(
                variant,
                str(specification["label"]),
                list(specification["modalities"]),
                _paths(output_root, variant),
            )
        )
    full = rows[0]
    for row in rows:
        for metric in ("accuracy", "macro_f1"):
            row[f"full_minus_variant_{metric}_pp"] = 100.0 * (
                float(full[f"{metric}_mean"])
                - float(row[f"{metric}_mean"])
            )
    result = {
        "schema_version": 1,
        "valid": True,
        "protocol": "uniform_three_seed_independently_trained_modality_matrix_v1",
        "seeds": list(SEEDS),
        "sample_count_per_seed": 826,
        "text_only_exclusion": (
            "The class-description bank has no sample-specific observation and "
            "is therefore evaluated only when conditioned on visual or sensor "
            "evidence."
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
        "accuracy_sd",
        "macro_precision_mean",
        "macro_recall_mean",
        "macro_f1_mean",
        "macro_f1_sd",
        "weighted_f1_mean",
        "nll_mean",
        "brier_mean",
        "ece_mean",
        "full_minus_variant_accuracy_pp",
        "full_minus_variant_macro_f1_pp",
        "active_parameter_count",
        "active_trainable_parameter_count",
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
        "# Independently trained modality matrix",
        "",
        "All entries are mean +/- sample SD over seeds 42, 43, and 44.",
        "",
        "| Configuration | Modalities | Accuracy (%) | Macro F1 (%) | NLL | Brier | ECE (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {modalities} | {acc:.2f} +/- {acc_sd:.2f} | "
            "{f1:.2f} +/- {f1_sd:.2f} | {nll:.3f} | {brier:.3f} | "
            "{ece:.2f} |".format(
                label=row["label"],
                modalities=" + ".join(row["modalities"]),
                acc=100.0 * row["accuracy_mean"],
                acc_sd=100.0 * row["accuracy_sd"],
                f1=100.0 * row["macro_f1_mean"],
                f1_sd=100.0 * row["macro_f1_sd"],
                nll=row["nll_mean"],
                brier=row["brier_mean"],
                ece=100.0 * row["ece_mean"],
            )
        )
    _atomic_text(output_root / "SUMMARY.md", "\n".join(lines) + "\n")
    return result


def main() -> None:
    result = summarize()
    print(json.dumps({"valid": result["valid"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
