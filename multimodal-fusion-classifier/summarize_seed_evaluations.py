"""Aggregate repeated training-seed evaluations into paper-ready statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mmfc.data import write_json
from mmfc.training import resolve_project_path


DEFAULT_METRICS = (
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
        description="Summarize evaluation metrics across training seeds"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=EVALUATION_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_named_paths(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw_path.strip():
            raise ValueError(f"run must use NAME=PATH: {value}")
        if name in seen:
            raise ValueError(f"duplicate run name: {name}")
        seen.add(name)
        result.append((name, resolve_project_path(Path(raw_path.strip()))))
    if len(result) < 2:
        raise ValueError("at least two runs are required")
    return result


def summarize_runs(
    runs: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    metric_names: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    individual: list[dict[str, Any]] = []
    sample_counts: set[int] = set()
    class_names: list[list[str]] = []
    for name, document in runs:
        metrics = document.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"run {name} has no metrics object")
        sample_count = int(metrics.get("sample_count", -1))
        if sample_count <= 0:
            raise ValueError(f"run {name} has invalid sample_count")
        sample_counts.add(sample_count)
        names = [str(item) for item in document.get("class_names", [])]
        if names:
            class_names.append(names)
        values: dict[str, float] = {}
        for metric_name in metric_names:
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"run {name} has invalid metric: {metric_name}"
                )
            values[metric_name] = float(value)
        individual.append(
            {
                "name": name,
                "sample_count": sample_count,
                "checkpoint_epoch": document.get("checkpoint", {}).get("epoch"),
                "metrics": values,
            }
        )
    if len(sample_counts) != 1:
        raise ValueError("runs do not use the same sample count")
    if class_names and any(names != class_names[0] for names in class_names[1:]):
        raise ValueError("runs do not use the same class order")

    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.asarray(
            [item["metrics"][metric_name] for item in individual],
            dtype=np.float64,
        )
        aggregate[metric_name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "valid": True,
        "run_count": len(individual),
        "sample_count_per_run": sample_counts.pop(),
        "class_names": class_names[0] if class_names else [],
        "individual": individual,
        "aggregate": aggregate,
    }


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(report["aggregate"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "checkpoint_epoch", *metric_names],
        )
        writer.writeheader()
        for item in report["individual"]:
            writer.writerow(
                {
                    "name": item["name"],
                    "checkpoint_epoch": item["checkpoint_epoch"],
                    **item["metrics"],
                }
            )
        for statistic in ("mean", "sample_std", "min", "max"):
            writer.writerow(
                {
                    "name": statistic,
                    "checkpoint_epoch": "",
                    **{
                        metric_name: report["aggregate"][metric_name][
                            statistic
                        ]
                        for metric_name in metric_names
                    },
                }
            )


def main() -> None:
    args = parse_args()
    named_paths = parse_named_paths(args.run)
    runs: list[tuple[str, Mapping[str, Any]]] = []
    for name, path in named_paths:
        with path.open("r", encoding="utf-8") as handle:
            runs.append((name, json.load(handle)))
    report = summarize_runs(runs)
    report["inputs"] = {
        name: str(path) for name, path in named_paths
    }
    output = resolve_project_path(args.output)
    csv_output = output.with_suffix(".csv")
    write_json(output, report)
    _write_csv(csv_output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "csv": str(csv_output),
                **report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
