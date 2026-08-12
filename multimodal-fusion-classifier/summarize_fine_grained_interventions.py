"""Summarize three-checkpoint fine-grained intervention evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from run_fine_grained_intervention_queue import (
    INTERVENTIONS,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    SEEDS,
)


BASELINES = {
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


def _metric_vector(paths: Mapping[int, Path], metric: str) -> np.ndarray:
    values = []
    source_name = METRICS[metric]
    for seed in SEEDS:
        report = _read_json(paths[seed])
        if report.get("metrics", {}).get("sample_count") != 826:
            raise ValueError(f"incomplete evaluation: {paths[seed]}")
        values.append(float(report["metrics"][source_name]))
    return np.asarray(values, dtype=np.float64)


def _mean_sd(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def summarize(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    baseline_vectors = {
        metric: _metric_vector(BASELINES, metric) for metric in METRICS
    }
    rows: list[dict[str, Any]] = []
    baseline_row: dict[str, Any] = {
        "variant": "full",
        "label": "Full model",
        "group": "reference",
        "protocol": "independently_trained_checkpoints",
    }
    for metric, values in baseline_vectors.items():
        mean, sd = _mean_sd(values)
        baseline_row[f"{metric}_mean"] = mean
        baseline_row[f"{metric}_sd"] = sd
        baseline_row[f"{metric}_by_seed"] = {
            str(seed): float(value) for seed, value in zip(SEEDS, values)
        }
    baseline_row["full_minus_variant_accuracy_pp_mean"] = 0.0
    baseline_row["full_minus_variant_accuracy_pp_sd"] = 0.0
    baseline_row["full_minus_variant_macro_f1_pp_mean"] = 0.0
    baseline_row["full_minus_variant_macro_f1_pp_sd"] = 0.0
    rows.append(baseline_row)

    for variant, specification in INTERVENTIONS.items():
        paths = {
            seed: output_root / variant / f"seed{seed}" / "test"
            / "evaluation.json"
            for seed in SEEDS
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing intervention reports: " + ", ".join(missing))
        row: dict[str, Any] = {
            "variant": variant,
            "label": specification["label"],
            "group": specification["group"],
            "protocol": "inference_path_intervention_no_retraining",
            "cli_args": list(specification["args"]),
        }
        vectors = {
            metric: _metric_vector(paths, metric) for metric in METRICS
        }
        for metric, values in vectors.items():
            mean, sd = _mean_sd(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_by_seed"] = {
                str(seed): float(value) for seed, value in zip(SEEDS, values)
            }
        for metric in ("accuracy", "macro_f1"):
            delta = 100.0 * (baseline_vectors[metric] - vectors[metric])
            mean, sd = _mean_sd(delta)
            row[f"full_minus_variant_{metric}_pp_mean"] = mean
            row[f"full_minus_variant_{metric}_pp_sd"] = sd
            row[f"full_minus_variant_{metric}_pp_by_seed"] = {
                str(seed): float(value) for seed, value in zip(SEEDS, delta)
            }
        rows.append(row)

    result = {
        "schema_version": 1,
        "valid": True,
        "protocol": "fine_grained_inference_path_intervention",
        "boundary": (
            "Each row masks a path in the same three independently trained full "
            "checkpoints. It measures active path reliance, not compensated "
            "performance after retraining."
        ),
        "seeds": list(SEEDS),
        "sample_count_per_seed": 826,
        "rows": rows,
    }
    _atomic_text(
        output_root / "SUMMARY.json",
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )

    csv_fields = [
        "variant",
        "label",
        "group",
        "accuracy_mean",
        "accuracy_sd",
        "macro_f1_mean",
        "macro_f1_sd",
        "nll_mean",
        "brier_mean",
        "ece_mean",
        "full_minus_variant_accuracy_pp_mean",
        "full_minus_variant_accuracy_pp_sd",
        "full_minus_variant_macro_f1_pp_mean",
        "full_minus_variant_macro_f1_pp_sd",
    ]
    csv_path = output_root / "SUMMARY.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in csv_fields})
    temporary.replace(csv_path)

    lines = [
        "# Fine-grained three-checkpoint intervention summary",
        "",
        (
            "All interventions are applied to the independently trained seed "
            "42/43/44 full checkpoints without retraining. Positive deltas mean "
            "the intact full path performs better."
        ),
        "",
        "| Configuration | Group | Accuracy (%) | Macro F1 (%) | Full - variant Acc. (pp) | Full - variant F1 (pp) | NLL | Brier | ECE (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {group} | {acc:.2f} +/- {acc_sd:.2f} | "
            "{f1:.2f} +/- {f1_sd:.2f} | {da:+.2f} | {df:+.2f} | "
            "{nll:.3f} | {brier:.3f} | {ece:.2f} |".format(
                label=row["label"],
                group=row["group"],
                acc=100.0 * row["accuracy_mean"],
                acc_sd=100.0 * row["accuracy_sd"],
                f1=100.0 * row["macro_f1_mean"],
                f1_sd=100.0 * row["macro_f1_sd"],
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
            {
                "valid": result["valid"],
                "rows": len(result["rows"]),
                "output": str(OUTPUT_ROOT / "SUMMARY.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
