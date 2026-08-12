"""Render paper-ready training curves from an atomic history document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mmfc.data import write_json
from mmfc.training import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multimodal training history")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def load_history(path: Path) -> tuple[dict[str, Any], list[dict[str, float]]]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    rows = document.get("history") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise ValueError("history document must contain a non-empty history list")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != sorted(set(epochs)):
        raise ValueError("history epochs must be unique and increasing")
    return document if isinstance(document, dict) else {}, rows


def _series(
    rows: Sequence[dict[str, Any]],
    name: str,
) -> np.ndarray | None:
    if not all(name in row and row[name] is not None for row in rows):
        return None
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def resolve_best_epoch(
    document: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> tuple[int | None, str | None, bool]:
    recorded_epoch = document.get("best_checkpoint_epoch")
    selection_metric = document.get("selection_metric")
    if recorded_epoch is not None:
        return int(recorded_epoch), selection_metric, False

    metric_name = (
        "val_macro_f1"
        if str(selection_metric).lower() == "macro_f1"
        else "val_accuracy"
    )
    eligible = [row for row in rows if row.get(metric_name) is not None]
    if not eligible:
        return None, selection_metric, False
    best_row = max(eligible, key=lambda row: float(row[metric_name]))
    return int(best_row["epoch"]), selection_metric or "accuracy", True


def plot_history(
    document: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    output: Path,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=np.int64)
    best_epoch, selection_metric, best_epoch_inferred = resolve_best_epoch(
        document,
        rows,
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), sharex=True)
    palette = {
        "train": "#1967A3",
        "val": "#D1495B",
        "secondary": "#2A9D8F",
        "accent": "#E9A23B",
        "muted": "#6C757D",
        "violet": "#7A5195",
    }

    score_axis = axes[0, 0]
    score_specs = (
        ("train_accuracy", "Train accuracy", palette["train"], "-"),
        ("val_accuracy", "Validation accuracy", palette["val"], "-"),
        ("train_macro_f1", "Train macro F1", palette["train"], "--"),
        ("val_macro_f1", "Validation macro F1", palette["val"], "--"),
    )
    for key, label, color, style in score_specs:
        values = _series(rows, key)
        if values is not None:
            score_axis.plot(
                epochs,
                100.0 * values,
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.7,
            )
    score_axis.set_ylabel("Score (%)")
    score_axis.set_title("Classification")
    score_axis.set_ylim(0.0, 100.0)
    score_axis.legend(loc="lower right", fontsize=8)

    loss_axis = axes[0, 1]
    loss_specs = (
        ("train_loss", "Train total", palette["train"], "-"),
        ("val_loss", "Validation total", palette["val"], "-"),
        ("train_cross_entropy", "Train CE", palette["secondary"], "--"),
        ("val_cross_entropy", "Validation CE", palette["accent"], "--"),
        ("train_focal", "Train focal", palette["muted"], ":"),
        ("val_focal", "Validation focal", palette["violet"], ":"),
    )
    for key, label, color, style in loss_specs:
        values = _series(rows, key)
        if values is not None:
            loss_axis.plot(
                epochs,
                values,
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.5,
            )
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Optimization losses")
    loss_axis.legend(loc="upper right", fontsize=7, ncol=2)

    auxiliary_axis = axes[1, 0]
    latent_nll = _series(rows, "val_sensor_reconstruction")
    auxiliary_handles = []
    auxiliary_labels = []
    if latent_nll is not None:
        latent_line = auxiliary_axis.plot(
            epochs,
            latent_nll,
            label="Latent NLL",
            color=palette["train"],
            linewidth=1.6,
        )
        auxiliary_handles += latent_line
        auxiliary_labels.append("Latent NLL")
    component_axis = auxiliary_axis.twinx()
    auxiliary_specs = (
        ("val_reliability_calibration", "Calibration", palette["val"]),
        ("val_sensor_variance_prior", "Variance prior", palette["secondary"]),
        ("val_sensor_grid_reconstruction", "Grid reconstruction", palette["accent"]),
        ("val_reliability_ranking", "Pairwise ranking", palette["violet"]),
    )
    for key, label, color in auxiliary_specs:
        values = _series(rows, key)
        if values is not None:
            lines = component_axis.plot(
                epochs,
                values,
                label=label,
                color=color,
                linewidth=1.5,
            )
            auxiliary_handles += lines
            auxiliary_labels.append(label)
    auxiliary_axis.set_xlabel("Epoch")
    auxiliary_axis.set_ylabel("Latent NLL")
    component_axis.set_ylabel("Other validation component")
    auxiliary_axis.set_title("Auxiliary objectives")
    auxiliary_axis.legend(
        auxiliary_handles,
        auxiliary_labels,
        loc="best",
        fontsize=7,
    )

    reliability_axis = axes[1, 1]
    reliability_specs = (
        ("train_sensor_reliability", "Train reliability", palette["train"]),
        ("val_sensor_reliability", "Validation reliability", palette["val"]),
        ("train_sensor_consistency", "Train consistency", palette["secondary"]),
        ("val_sensor_consistency", "Validation consistency", palette["accent"]),
    )
    for key, label, color in reliability_specs:
        values = _series(rows, key)
        if values is not None:
            reliability_axis.plot(
                epochs,
                values,
                label=label,
                color=color,
                linewidth=1.5,
            )
    reliability_axis.set_xlabel("Epoch")
    reliability_axis.set_ylabel("Mean score")
    reliability_axis.set_ylim(0.0, 1.0)
    reliability_axis.set_title("Reliability diagnostics")
    learning_rate = _series(rows, "learning_rate")
    handles, labels = reliability_axis.get_legend_handles_labels()
    if learning_rate is not None:
        rate_axis = reliability_axis.twinx()
        rate_axis.plot(
            epochs,
            learning_rate,
            label="Learning rate",
            color=palette["muted"],
            linewidth=1.2,
            linestyle=":",
        )
        rate_axis.set_ylabel("Learning rate")
        rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
        handles += rate_handles
        labels += rate_labels
    reliability_axis.legend(handles, labels, loc="best", fontsize=7)

    for axis in axes.reshape(-1):
        axis.grid(axis="both", alpha=0.18)
        if best_epoch is not None:
            axis.axvline(
                int(best_epoch),
                color="#202124",
                linewidth=1.0,
                linestyle="--",
                alpha=0.7,
            )
    figure.suptitle(title or "Multimodal classifier training history", fontsize=14)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return {
        "history_rows": len(rows),
        "first_epoch": int(epochs[0]),
        "last_epoch": int(epochs[-1]),
        "best_checkpoint_epoch": (
            int(best_epoch) if best_epoch is not None else None
        ),
        "best_checkpoint_epoch_inferred": best_epoch_inferred,
        "selection_metric": selection_metric,
        "best_selection_score": document.get("best_selection_score"),
        "output": str(output),
    }


def main() -> None:
    args = parse_args()
    history_path = resolve_project_path(args.history)
    output = resolve_project_path(args.output)
    document, rows = load_history(history_path)
    report = plot_history(document, rows, output, title=args.title)
    report["history"] = str(history_path)
    report_path = output.with_suffix(".json")
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
