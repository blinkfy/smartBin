"""Plot paper-result figures from the strictly audited CSV table bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURE_FILENAMES = (
    "formal_ablation_effects.png",
    "yolo_segmentation_metrics.png",
    "scene_pipeline_comparison.png",
    "scene_strata_diagnostics.png",
    "sensor_source_comparison.png",
)
COLORS = ("#2474A6", "#2E8B57", "#D9822B", "#B54A4A", "#6C5B9A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final paper figures from generated CSV tables"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "paper-tables"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(value: Any) -> str:
    text = str(value).replace("_", " ")
    return "\n".join(
        textwrap.wrap(
            text,
            width=18,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(
        temporary,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    temporary.replace(path)


def _plot_formal_ablations(
    rows: Sequence[Mapping[str, Any]],
    statistics: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    candidates = [
        row for row in rows if str(row.get("experiment")) != "full"
    ]
    statistics_by_name = {
        str(row.get("experiment")): row for row in statistics
    }
    names = [str(row.get("experiment")) for row in candidates]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    specs = (
        (
            "accuracy_delta_pp",
            "accuracy_ci_lower_pp",
            "accuracy_ci_upper_pp",
            "Accuracy delta (pp)",
            COLORS[0],
        ),
        (
            "macro_f1_delta_pp",
            "macro_f1_ci_lower_pp",
            "macro_f1_ci_upper_pp",
            "Macro F1 delta (pp)",
            COLORS[1],
        ),
    )
    for axis, (point_key, low_key, high_key, ylabel, color) in zip(
        axes,
        specs,
    ):
        points = np.array([_number(row.get(point_key)) for row in candidates])
        lower = []
        upper = []
        for name, point in zip(names, points):
            stat = statistics_by_name.get(name, {})
            low = _number(stat.get(low_key), point)
            high = _number(stat.get(high_key), point)
            lower.append(max(0.0, point - low))
            upper.append(max(0.0, high - point))
        axis.bar(x, points, color=color, width=0.68)
        axis.errorbar(
            x,
            points,
            yerr=np.array([lower, upper]),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1.0,
        )
        axis.axhline(0.0, color="#444444", linewidth=0.9)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axes[0].set_title(
        "Formal retrained ablations (legacy full reference)"
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([_label(name) for name in names], fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def _plot_yolo(row: Mapping[str, Any], path: Path) -> None:
    labels = (
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
    )
    box = [
        _number(row.get(key))
        for key in (
            "box_precision",
            "box_recall",
            "box_map50",
            "box_map50_95",
        )
    ]
    mask = [
        _number(row.get(key))
        for key in (
            "mask_precision",
            "mask_recall",
            "mask_map50",
            "mask_map50_95",
        )
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(x - width / 2, box, width, label="Box", color=COLORS[0])
    axis.bar(x + width / 2, mask, width, label="Mask", color=COLORS[2])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Metric value")
    axis.set_title("Target-domain YOLO11s-seg test metrics")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.legend(frameon=False, ncols=2)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    _save(fig, path)


def _plot_scene_pipeline(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    sources = sorted({str(row.get("prediction_source")) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True)
    for source, color in zip(sources, COLORS):
        source_rows = sorted(
            (
                row
                for row in rows
                if str(row.get("prediction_source")) == source
            ),
            key=lambda row: _number(row.get("confidence_threshold")),
        )
        thresholds = [
            _number(row.get("confidence_threshold")) for row in source_rows
        ]
        axes[0].plot(
            thresholds,
            [_number(row.get("end_to_end_recall")) for row in source_rows],
            marker="o",
            label=source,
            color=color,
        )
        axes[1].plot(
            thresholds,
            [_number(row.get("macro_f1")) for row in source_rows],
            marker="o",
            label=source,
            color=color,
        )
    for axis, title in zip(
        axes,
        ("End-to-end recall", "Scene-level Macro F1"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Detector confidence threshold")
        axis.set_ylim(0.0, 1.0)
        axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Two-stage scene pipeline")
    fig.tight_layout()
    _save(fig, path)


def _plot_scene_strata(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    thresholds = sorted(
        {_number(row.get("confidence_threshold")) for row in rows}
    )
    selected_threshold = thresholds[0] if thresholds else 0.0
    selected = [
        row
        for row in rows
        if _number(row.get("confidence_threshold")) == selected_threshold
    ]
    dimensions = sorted({str(row.get("dimension")) for row in selected})
    column_count = max(1, len(dimensions))
    fig, axes_value = plt.subplots(
        1,
        column_count,
        figsize=(5.0 * column_count, 4.8),
        squeeze=False,
        sharey=True,
    )
    axes = axes_value[0]
    for axis, dimension in zip(axes, dimensions):
        dimension_rows = [
            row for row in selected if str(row.get("dimension")) == dimension
        ]
        strata = sorted({str(row.get("stratum")) for row in dimension_rows})
        sources = sorted(
            {str(row.get("prediction_source")) for row in dimension_rows}
        )
        x = np.arange(len(strata))
        width = 0.8 / max(1, len(sources))
        for index, (source, color) in enumerate(zip(sources, COLORS)):
            lookup = {
                str(row.get("stratum")): row
                for row in dimension_rows
                if str(row.get("prediction_source")) == source
            }
            values = [
                _number(lookup.get(stratum, {}).get("end_to_end_recall"))
                for stratum in strata
            ]
            offset = (index - (len(sources) - 1) / 2) * width
            axis.bar(
                x + offset,
                values,
                width,
                label=source,
                color=color,
            )
        axis.set_title(_label(dimension))
        axis.set_xticks(x)
        axis.set_xticklabels([_label(value) for value in strata], fontsize=8)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel("End-to-end recall")
    if dimensions:
        axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Scene strata diagnostics (confidence={selected_threshold:g})"
    )
    fig.tight_layout()
    _save(fig, path)


def _plot_sensor_sources(
    runtime_rows: Sequence[Mapping[str, Any]],
    retraining_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharey=True)
    groups = (
        (
            axes[0],
            runtime_rows,
            "sensor_source",
            "Same-checkpoint source transfer",
        ),
        (
            axes[1],
            retraining_rows,
            "training_sensor_source",
            "Independent source retraining",
        ),
    )
    for axis, rows, label_key, title in groups:
        labels = [str(row.get(label_key)) for row in rows]
        x = np.arange(len(labels))
        width = 0.36
        axis.bar(
            x - width / 2,
            [_number(row.get("accuracy")) for row in rows],
            width,
            label="Accuracy",
            color=COLORS[0],
        )
        axis.bar(
            x + width / 2,
            [_number(row.get("macro_f1")) for row in rows],
            width,
            label="Macro F1",
            color=COLORS[1],
        )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([_label(value) for value in labels])
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(frameon=False, ncols=2)
    fig.tight_layout()
    _save(fig, path)


def build_paper_figures(
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    outputs = {
        filename: output_dir / filename for filename in FIGURE_FILENAMES
    }
    _plot_formal_ablations(
        tables["formal_ablations.csv"],
        tables["formal_ablation_statistics.csv"],
        outputs["formal_ablation_effects.png"],
    )
    _plot_yolo(
        tables["yolo_segmentation.csv"][0],
        outputs["yolo_segmentation_metrics.png"],
    )
    _plot_scene_pipeline(
        tables["scene_pipeline.csv"],
        outputs["scene_pipeline_comparison.png"],
    )
    _plot_scene_strata(
        tables["scene_strata.csv"],
        outputs["scene_strata_diagnostics.png"],
    )
    _plot_sensor_sources(
        tables["sensor_source_transfer.csv"],
        tables["sensor_source_retraining.csv"],
        outputs["sensor_source_comparison.png"],
    )
    return outputs


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = (
        args.output.resolve() if args.output is not None else input_dir
    )
    required = (
        "formal_ablations.csv",
        "formal_ablation_statistics.csv",
        "yolo_segmentation.csv",
        "scene_pipeline.csv",
        "scene_strata.csv",
        "sensor_source_transfer.csv",
        "sensor_source_retraining.csv",
    )
    tables = {name: _read_csv(input_dir / name) for name in required}
    outputs = build_paper_figures(
        tables=tables,
        output_dir=output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}\t{path}")


if __name__ == "__main__":
    main()
