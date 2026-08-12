"""Plot train-set class imbalance beside per-class model F1."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mmfc.data import write_json
from mmfc.training import load_config, read_jsonl, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot class frequency and per-class F1"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="NAME=EVALUATION_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_named_evaluations(
    values: Sequence[str],
) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw_path.strip():
            raise ValueError(f"evaluation must use NAME=PATH: {value}")
        if name in seen:
            raise ValueError(f"duplicate evaluation name: {name}")
        seen.add(name)
        result.append((name, resolve_project_path(Path(raw_path.strip()))))
    return result


def build_class_rows(
    class_names: Sequence[str],
    train_records: Sequence[Mapping[str, Any]],
    evaluations: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    counts = Counter(str(record["label"]) for record in train_records)
    if set(counts) != set(class_names):
        missing = sorted(set(class_names) - set(counts))
        extra = sorted(set(counts) - set(class_names))
        raise ValueError(f"train class mismatch; missing={missing}, extra={extra}")

    per_model: dict[str, dict[str, float]] = {}
    for model_name, document in evaluations:
        per_class = document.get("per_class")
        if not isinstance(per_class, list):
            raise ValueError(f"evaluation {model_name} has no per_class list")
        values = {str(row["label"]): float(row["f1"]) for row in per_class}
        if set(values) != set(class_names):
            raise ValueError(f"evaluation {model_name} class labels do not match")
        per_model[model_name] = values

    return sorted(
        [
            {
                "label": label,
                "train_count": int(counts[label]),
                "f1": {
                    model_name: values[label]
                    for model_name, values in per_model.items()
                },
            }
            for label in class_names
        ],
        key=lambda row: (-row["train_count"], row["label"]),
    )


def plot_class_rows(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(rows[0]["f1"])
    colors = ["#1967A3", "#D1495B", "#E9A23B", "#2A9D8F", "#7A5195"]
    if len(model_names) > len(colors):
        raise ValueError("at most five evaluations can be plotted")
    indices = np.arange(len(rows), dtype=np.float64)
    labels = [row["label"] for row in rows]
    counts = np.asarray([row["train_count"] for row in rows])
    figure, (count_axis, score_axis) = plt.subplots(
        1,
        2,
        figsize=(15.0, 10.0),
        sharey=True,
        gridspec_kw={"width_ratios": [0.8, 1.45]},
    )

    count_axis.barh(indices, counts, color="#6C8EAD", height=0.72)
    count_axis.set_xscale("log")
    count_axis.set_xlabel("Training samples (log scale)")
    count_axis.set_title("Long-tailed training distribution")
    count_axis.set_yticks(indices, labels)
    count_axis.invert_yaxis()
    count_axis.grid(axis="x", alpha=0.2)
    for index, count in enumerate(counts):
        count_axis.text(
            float(count) * 1.04,
            index,
            str(int(count)),
            va="center",
            ha="left",
            fontsize=8,
        )
    count_axis.set_xlim(max(10.0, counts.min() * 0.65), counts.max() * 1.45)

    group_height = 0.76
    bar_height = group_height / max(len(model_names), 1)
    for model_index, model_name in enumerate(model_names):
        offset = (
            model_index - (len(model_names) - 1) / 2.0
        ) * bar_height
        values = 100.0 * np.asarray(
            [row["f1"][model_name] for row in rows],
            dtype=np.float64,
        )
        score_axis.barh(
            indices + offset,
            values,
            height=bar_height * 0.9,
            color=colors[model_index],
            label=model_name,
        )
    score_axis.axvline(80.0, color="#455A64", linewidth=1.0, linestyle=":")
    score_axis.set_xlim(0.0, 105.0)
    score_axis.set_xlabel("Per-class F1 (%)")
    score_axis.set_title("Fixed held-out split performance")
    score_axis.grid(axis="x", alpha=0.2)
    score_axis.legend(loc="lower right", fontsize=8)

    figure.suptitle(
        "Class imbalance and fine-grained classification performance",
        fontsize=14,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {
        "class_count": len(rows),
        "models": model_names,
        "max_train_count": int(counts.max()),
        "min_train_count": int(counts.min()),
        "imbalance_ratio": float(counts.max() / counts.min()),
        "output": str(output),
        "pdf": str(pdf_path),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    class_names = [
        item["name"] if isinstance(item, dict) else str(item)
        for item in config["classes"]
    ]
    train_records = read_jsonl(
        resolve_project_path(config["data"]["train_manifest"])
    )
    evaluation_paths = parse_named_evaluations(args.evaluation)
    evaluations: list[tuple[str, Mapping[str, Any]]] = []
    for name, path in evaluation_paths:
        with path.open("r", encoding="utf-8") as handle:
            evaluations.append((name, json.load(handle)))
    rows = build_class_rows(class_names, train_records, evaluations)
    output = resolve_project_path(args.output)
    report = plot_class_rows(rows, output)
    report["inputs"] = {
        name: str(path) for name, path in evaluation_paths
    }
    report_path = output.with_suffix(".json")
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                **{key: value for key, value in report.items() if key != "rows"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
