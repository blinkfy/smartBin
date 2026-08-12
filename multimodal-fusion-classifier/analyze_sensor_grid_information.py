"""Characterize spatial information in the recorded 4x4 sensor arrays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize dynamic range and spatial variation of 4x4 grids"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "qwen-material" / "test.jsonl",
    )
    parser.add_argument(
        "--sensor-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "qwen-material" / "sensors.jsonl",
    )
    parser.add_argument(
        "--display-names",
        type=Path,
        default=(
            PROJECT_ROOT
            / "paper_rewriting_output"
            / "aann2026_submission"
            / "class_display_names.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "sensor-grid-information"
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            records.append(value)
    return records


def extract_grid(record: dict[str, Any]) -> np.ndarray:
    for key in (
        "normalized_grid",
        "sensor_grid",
        "grid",
        "sensor_array",
        "array",
        "readings",
        "values",
        "data",
    ):
        if key in record:
            value = np.asarray(record[key], dtype=np.float64)
            if value.shape == (1, 4, 4):
                value = value[0]
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError(
                    f"invalid 4x4 grid for sample {record.get('sample_id')}"
                )
            return value
    raise ValueError(f"no grid for sample {record.get('sample_id')}")


def sample_statistics(grid: np.ndarray) -> dict[str, float]:
    horizontal = np.abs(np.diff(grid, axis=1))
    vertical = np.abs(np.diff(grid, axis=0))
    return {
        "mean": float(grid.mean()),
        "std": float(grid.std(ddof=0)),
        "variance": float(grid.var(ddof=0)),
        "dynamic_range": float(np.ptp(grid)),
        "horizontal_gradient": float(horizontal.mean()),
        "vertical_gradient": float(vertical.mean()),
        "mean_gradient": float(0.5 * (horizontal.mean() + vertical.mean())),
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "maximum": float(values.max()),
    }


def write_figure(
    rows: list[dict[str, Any]],
    grids: dict[str, np.ndarray],
    class_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "svg.fonttype": "none",
            "svg.hashsalt": "mmfc-grid-information-2026",
        }
    )
    ranges = np.asarray([row["dynamic_range"] for row in rows])
    ordered = sorted(rows, key=lambda row: row["dynamic_range"])
    example_indices = (0, len(ordered) // 3, 2 * len(ordered) // 3, -1)
    examples = [ordered[index] for index in example_indices]

    figure = plt.figure(figsize=(11.2, 5.0), constrained_layout=True)
    layout = figure.add_gridspec(2, 4, height_ratios=(1.0, 1.25))
    example_titles = ("Exact flat", "Lower range", "Higher range", "Maximum range")
    for column, (row, title) in enumerate(zip(examples, example_titles)):
        axis = figure.add_subplot(layout[0, column])
        image = axis.imshow(
            grids[row["sample_id"]],
            vmin=0.0,
            vmax=1.0,
            cmap="RdYlBu_r",
            interpolation="nearest",
        )
        axis.set_title(
            f"{title}\nrange={row['dynamic_range']:.3f}",
            fontweight="bold",
        )
        axis.set_xticks(range(4))
        axis.set_yticks(range(4))
        axis.set_xticklabels(range(1, 5))
        axis.set_yticklabels(range(1, 5))
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#4a4f55")
        if column == 3:
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    histogram = figure.add_subplot(layout[1, :2])
    histogram.hist(
        ranges,
        bins=np.linspace(0.0, 1.0, 31),
        color="#567a9c",
        edgecolor="white",
        linewidth=0.5,
    )
    histogram.axvline(0.02, color="#a34a47", linestyle="--", linewidth=1.1)
    histogram.axvline(0.05, color="#3f7758", linestyle="--", linewidth=1.1)
    histogram.set_xlabel("Per-sample dynamic range")
    histogram.set_ylabel("Object records")
    histogram.set_title("(a) Test-grid dynamic range", loc="left", fontweight="bold")
    histogram.grid(axis="y", color="#d9dde2", linewidth=0.6)
    histogram.spines[["top", "right"]].set_visible(False)

    class_axis = figure.add_subplot(layout[1, 2:])
    sorted_classes = sorted(
        class_rows,
        key=lambda row: row["median_dynamic_range"],
    )
    names = [row["display_name"] for row in sorted_classes]
    medians = [row["median_dynamic_range"] for row in sorted_classes]
    q1 = [row["q1_dynamic_range"] for row in sorted_classes]
    q3 = [row["q3_dynamic_range"] for row in sorted_classes]
    positions = np.arange(len(names))
    class_axis.errorbar(
        medians,
        positions,
        xerr=(np.asarray(medians) - q1, np.asarray(q3) - medians),
        fmt="o",
        color="#3f7758",
        ecolor="#88a894",
        capsize=2,
        markersize=3.8,
        linewidth=1.0,
    )
    class_axis.set_yticks(positions)
    class_axis.set_yticklabels(names)
    class_axis.set_xlabel("Dynamic range, median [IQR]")
    class_axis.set_title("(b) Variation by waste category", loc="left", fontweight="bold")
    class_axis.grid(axis="x", color="#d9dde2", linewidth=0.6)
    class_axis.spines[["top", "right"]].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    sensor_records = read_jsonl(args.sensor_manifest)
    sensors = {str(record["sample_id"]): record for record in sensor_records}
    display_names = json.loads(args.display_names.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    grids: dict[str, np.ndarray] = {}
    for item in manifest:
        sample_id = str(item["sample_id"])
        if sample_id not in sensors:
            raise ValueError(f"missing sensor record: {sample_id}")
        grid = extract_grid(sensors[sample_id])
        grids[sample_id] = grid
        rows.append(
            {
                "sample_id": sample_id,
                "source_image_id": str(item["source_image_id"]),
                "label": str(item["label"]),
                "display_name": display_names.get(str(item["label"]), str(item["label"])),
                **sample_statistics(grid),
            }
        )

    class_rows: list[dict[str, Any]] = []
    for label in sorted({row["label"] for row in rows}):
        selected = [row for row in rows if row["label"] == label]
        ranges = np.asarray([row["dynamic_range"] for row in selected])
        variances = np.asarray([row["variance"] for row in selected])
        class_rows.append(
            {
                "label": label,
                "display_name": display_names.get(label, label),
                "count": len(selected),
                "exact_flat_count": int(np.count_nonzero(ranges <= 1e-12)),
                "exact_flat_fraction": float(np.mean(ranges <= 1e-12)),
                "mean_dynamic_range": float(ranges.mean()),
                "q1_dynamic_range": float(np.quantile(ranges, 0.25)),
                "median_dynamic_range": float(np.median(ranges)),
                "q3_dynamic_range": float(np.quantile(ranges, 0.75)),
                "mean_variance": float(variances.mean()),
                "median_variance": float(np.median(variances)),
            }
        )

    dynamic_range = np.asarray([row["dynamic_range"] for row in rows])
    variance = np.asarray([row["variance"] for row in rows])
    gradients = np.asarray([row["mean_gradient"] for row in rows])
    exact_flat = dynamic_range <= 1e-12
    summary = {
        "valid": True,
        "sample_count": len(rows),
        "source_image_count": len({row["source_image_id"] for row in rows}),
        "grid_shape": [1, 4, 4],
        "grid_field": "normalized_grid",
        "exact_flat": {
            "threshold": 1e-12,
            "count": int(exact_flat.sum()),
            "fraction": float(exact_flat.mean()),
        },
        "low_dynamic_range": {
            "range_le_0.02": {
                "count": int(np.count_nonzero(dynamic_range <= 0.02)),
                "fraction": float(np.mean(dynamic_range <= 0.02)),
            },
            "range_le_0.05": {
                "count": int(np.count_nonzero(dynamic_range <= 0.05)),
                "fraction": float(np.mean(dynamic_range <= 0.05)),
            },
        },
        "dynamic_range": distribution(dynamic_range),
        "spatial_variance": distribution(variance),
        "mean_neighbor_gradient": distribution(gradients),
        "classes": class_rows,
        "interpretation_boundary": (
            "Exact flatness means no within-grid spatial contrast after stored "
            "normalization; it does not imply absence of an absolute response."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, values in (("samples.csv", rows), ("classes.csv", class_rows)):
        with (args.output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write_figure(rows, grids, class_rows, args.output / "sensor_grid_information.svg")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
