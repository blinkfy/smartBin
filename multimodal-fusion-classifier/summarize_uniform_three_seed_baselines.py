"""Validate and summarize the uniform three-seed main comparison."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from prepare_standard_fusion_baselines import SEEDS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "qwen-material"
    / "uniform-three-seed-main-comparison"
)
METHODS = (
    "concat_mlp",
    "gmu",
    "lmf",
    "tmc",
    "qmf",
    "proposed",
)
DISPLAY_NAMES = {
    "concat_mlp": "Concat-MLP",
    "gmu": "GMU",
    "lmf": "LMF",
    "tmc": "TMC",
    "qmf": "QMF",
    "proposed": "Proposed",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the matched seed42/43/44 main comparison"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _evaluation_path(method: str, seed: int) -> Path:
    report_root = PROJECT_ROOT / "reports" / "qwen-material"
    if method in {"concat_mlp", "gmu", "lmf"}:
        return (
            report_root / "standard-fusion-baselines" / method
            / f"seed{seed}" / "test" / "evaluation.json"
        )
    if method in {"tmc", "qmf"}:
        return (
            report_root / "recent-robust-baselines" / method
            / f"seed{seed}" / "test" / "evaluation.json"
        )
    if seed == 42:
        return report_root / "full_cross_modal_quality" / "test" / "evaluation.json"
    return (
        report_root / "full_cross_modal_quality" / "seed-replication"
        / f"seed{seed}" / "test" / "evaluation.json"
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _format_percent(summary: Mapping[str, float], key: str) -> str:
    return f"{100 * summary[key + '_mean']:.2f} ± {100 * summary[key + '_sd']:.2f}"


def build_summary(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    method_rows: list[dict[str, Any]] = []
    for method in METHODS:
        runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            path = _evaluation_path(method, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            report = _read_json(path)
            metrics = report.get("metrics", {})
            missing = [key for key in METRICS if key not in metrics]
            if missing:
                raise ValueError(f"missing metrics {missing}: {path}")
            if int(metrics.get("sample_count", 0)) != 826:
                raise ValueError(f"unexpected test sample count: {path}")
            runs.append(
                {
                    "seed": seed,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "metrics": {key: float(metrics[key]) for key in METRICS},
                }
            )
        aggregate: dict[str, float] = {}
        for key in METRICS:
            values = [run["metrics"][key] for run in runs]
            aggregate[f"{key}_mean"] = mean(values)
            aggregate[f"{key}_sd"] = stdev(values)
        method_rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "seeds": list(SEEDS),
                "runs": runs,
                "aggregate": aggregate,
            }
        )

    payload: dict[str, Any] = {
        "valid": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "runs_per_method": len(SEEDS),
            "test_samples_per_run": 826,
            "dispersion": "sample_standard_deviation_across_training_seeds",
        },
        "methods": method_rows,
    }
    _atomic_write(output_dir / "SUMMARY.json", payload)

    csv_path = output_dir / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["method"] + [
            f"{key}_{suffix}" for key in METRICS for suffix in ("mean", "sd")
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in method_rows:
            writer.writerow({"method": row["display_name"], **row["aggregate"]})

    markdown = [
        "| Method | Accuracy (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | Weighted F1 (%) | NLL | Brier | ECE (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        aggregate = row["aggregate"]
        markdown.append(
            "| {name} | {acc} | {mp} | {mr} | {mf1} | {wf1} | {nll:.3f} ± {nll_sd:.3f} | {brier:.3f} ± {brier_sd:.3f} | {ece} |".format(
                name=row["display_name"],
                acc=_format_percent(aggregate, "accuracy"),
                mp=_format_percent(aggregate, "macro_precision"),
                mr=_format_percent(aggregate, "macro_recall"),
                mf1=_format_percent(aggregate, "macro_f1"),
                wf1=_format_percent(aggregate, "weighted_f1"),
                nll=aggregate["negative_log_likelihood_mean"],
                nll_sd=aggregate["negative_log_likelihood_sd"],
                brier=aggregate["multiclass_brier_score_mean"],
                brier_sd=aggregate["multiclass_brier_score_sd"],
                ece=_format_percent(aggregate, "ece"),
            )
        )
    (output_dir / "TABLE.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    args = parse_args()
    result = build_summary(args.output)
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "methods": result["protocol"]["methods"],
                "seeds": result["protocol"]["seeds"],
                "runs_per_method": result["protocol"]["runs_per_method"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
