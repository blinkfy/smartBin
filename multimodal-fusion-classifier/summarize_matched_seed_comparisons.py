"""Aggregate matched-training-seed paired prediction comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mmfc.training import resolve_project_path


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
        description="Summarize paired comparisons over matched training seeds"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=COMPARISON_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_named_paths(values: Sequence[str]) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw_path.strip():
            raise ValueError(f"run must use NAME=PATH: {value}")
        if name in seen:
            raise ValueError(f"duplicate run name: {name}")
        seen.add(name)
        result.append(
            (name, resolve_project_path(Path(raw_path.strip())))
        )
    if len(result) < 2:
        raise ValueError("at least two matched training seeds are required")
    return result


def summarize_comparisons(
    runs: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("at least two matched training seeds are required")
    individual = []
    for name, document in runs:
        paired = document.get("paired_bootstrap", {})
        differences = paired.get("differences", {})
        if (
            paired.get("repeats") != 5000
            or paired.get("sample_count") != 826
            or paired.get("cluster_count") != 570
        ):
            raise ValueError(f"run {name} paired-bootstrap identity mismatch")
        values = {}
        for metric in METRICS:
            value = differences.get(metric, {}).get("point")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"run {name} has invalid difference: {metric}")
            values[metric] = float(value)
        individual.append(
            {
                "name": name,
                "reference_name": document.get("reference", {}).get("name"),
                "candidate_name": document.get("candidate", {}).get("name"),
                "reference_metrics": document.get("reference", {}).get(
                    "metrics"
                ),
                "candidate_metrics": document.get("candidate", {}).get(
                    "metrics"
                ),
                "mcnemar_exact": document.get("mcnemar_exact"),
                "differences": values,
            }
        )
    aggregate = {}
    for metric in METRICS:
        values = np.asarray(
            [item["differences"][metric] for item in individual],
            dtype=np.float64,
        )
        aggregate[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_seed_count": int((values > 0).sum()),
            "negative_seed_count": int((values < 0).sum()),
            "zero_seed_count": int((values == 0).sum()),
        }
    return {
        "valid": True,
        "run_count": len(individual),
        "interpretation": (
            "candidate minus reference for every metric; lower-is-better "
            "metrics are favorable to the candidate when negative"
        ),
        "individual": individual,
        "aggregate": aggregate,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("name", *METRICS),
        )
        writer.writeheader()
        for item in report["individual"]:
            writer.writerow({"name": item["name"], **item["differences"]})
        for statistic in ("mean", "sample_std", "min", "max"):
            writer.writerow(
                {
                    "name": statistic,
                    **{
                        metric: report["aggregate"][metric][statistic]
                        for metric in METRICS
                    },
                }
            )


def main() -> None:
    args = parse_args()
    named_paths = parse_named_paths(args.run)
    runs = []
    for name, path in named_paths:
        runs.append(
            (
                name,
                json.loads(path.read_text(encoding="utf-8-sig")),
            )
        )
    report = summarize_comparisons(runs)
    report["inputs"] = {
        name: path.as_posix() for name, path in named_paths
    }
    output = resolve_project_path(args.output)
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _write_csv(output.with_suffix(".csv"), report)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "run_count": report["run_count"],
                "output": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
