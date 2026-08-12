"""Render the RGB-only deployable joint multimodal Segment-head architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
    / "joint_head_architecture.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the joint multimodal segmentation architecture."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def draw_joint_architecture(output: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output = resolve_project_path(output)
    colors = {
        "rgb": "#DCEAF5",
        "yolo": "#D8E7DB",
        "text": "#F4E2A7",
        "sensor": "#F5D2C5",
        "fusion": "#DDD7ED",
        "head": "#CEE3E6",
        "loss": "#ECEFF1",
        "border": "#263238",
        "line": "#455A64",
        "training": "#B23A48",
    }
    figure, axis = plt.subplots(figsize=(17.2, 8.8))
    axis.set_xlim(0, 17.2)
    axis.set_ylim(0, 8.8)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color: str,
        *,
        fontsize: float = 8.6,
        linewidth: float = 1.2,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.025,rounding_size=0.05",
                facecolor=color,
                edgecolor=colors["border"],
                linewidth=linewidth,
                zorder=3,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.25,
            zorder=4,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        dashed: bool = False,
        color: str | None = None,
        width: float = 1.25,
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=width,
                linestyle="--" if dashed else "-",
                color=color or colors["line"],
                connectionstyle="arc3,rad=0.0",
                zorder=2,
            )
        )

    def route(
        points: list[tuple[float, float]],
        *,
        dashed: bool = False,
        color: str | None = None,
        width: float = 1.25,
    ) -> None:
        line_color = color or colors["line"]
        style = "--" if dashed else "-"
        for start, end in zip(points[:-2], points[1:-1]):
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=line_color,
                linestyle=style,
                linewidth=width,
                zorder=2,
            )
        arrow(
            points[-2],
            points[-1],
            dashed=dashed,
            color=line_color,
            width=width,
        )

    axis.text(
        8.6,
        8.47,
        "RGB-only deployable text and predicted-soft-sensor YOLO11s-seg head",
        ha="center",
        va="center",
        fontsize=15.5,
        weight="bold",
    )
    axis.text(
        8.6,
        8.08,
        "Training supervision is separated from the metric forward path",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#546E7A",
    )

    box(0.35, 4.55, 1.65, 1.0, "RGB image\n640 x 640", colors["rgb"], fontsize=9.2)
    box(
        2.45,
        4.25,
        2.15,
        1.6,
        "Frozen target-domain\nYOLO11s-seg\nbackbone + neck",
        colors["yolo"],
        fontsize=9.0,
    )
    box(
        5.1,
        4.15,
        1.65,
        1.8,
        "P3  80 x 80\nP4  40 x 40\nP5  20 x 20",
        colors["yolo"],
        fontsize=9.1,
    )

    box(
        4.75,
        6.55,
        2.2,
        0.95,
        "Frozen CLIP class prototypes\n19 x 512",
        colors["text"],
        fontsize=8.7,
    )
    box(
        7.45,
        6.35,
        2.45,
        1.35,
        "Projection + per-scale\nspatial semantic attention\ntext residual  ΔT_i",
        colors["text"],
        fontsize=8.7,
    )

    box(
        7.35,
        2.6,
        2.7,
        1.6,
        "Global pooled P3/P4/P5\nRGB-to-4 x 4 predictor\nmean + uncertainty",
        colors["sensor"],
        fontsize=8.7,
    )
    box(
        10.55,
        2.6,
        2.25,
        1.6,
        "Reliability exp(-u)\n4 x 4 grid encoder\nsensor residual  ΔS_i",
        colors["sensor"],
        fontsize=8.6,
    )

    box(
        10.35,
        4.55,
        2.8,
        1.25,
        "Per-scale residual conditioning\nP_i + g_T ΔT_i + g_S r ΔS_i",
        colors["fusion"],
        fontsize=9.0,
        linewidth=1.5,
    )
    box(
        13.65,
        4.45,
        1.7,
        1.45,
        "Native\nSegment head",
        colors["head"],
        fontsize=9.4,
        linewidth=1.5,
    )
    box(
        15.75,
        4.25,
        1.15,
        1.85,
        "bbox\nclass\nmask\nconfidence",
        colors["head"],
        fontsize=8.7,
    )

    box(
        0.65,
        0.75,
        2.55,
        1.05,
        "Qwen/rule image-conditioned\n4 x 4 soft target",
        colors["loss"],
        fontsize=8.5,
    )
    box(
        4.0,
        0.6,
        2.7,
        1.35,
        "Training only\nweighted grid reconstruction\n+ uncertainty calibration",
        colors["loss"],
        fontsize=8.5,
    )
    box(
        10.1,
        0.65,
        3.2,
        1.15,
        "Optional future observed 4 x 4 sensor\n(interface present; not used in formal metrics)",
        "#F4F5F5",
        fontsize=8.2,
    )

    arrow((2.0, 5.05), (2.45, 5.05))
    arrow((4.6, 5.05), (5.1, 5.05))
    arrow((6.75, 5.05), (10.35, 5.05))
    arrow((13.15, 5.17), (13.65, 5.17))
    arrow((15.35, 5.17), (15.75, 5.17))

    arrow((6.95, 7.02), (7.45, 7.02))
    route([(8.68, 6.35), (8.68, 5.72), (10.35, 5.72)])

    route([(5.93, 4.15), (5.93, 3.4), (7.35, 3.4)])
    arrow((10.05, 3.4), (10.55, 3.4))
    route([(11.68, 4.2), (11.68, 4.55)])

    arrow(
        (3.2, 1.28),
        (4.0, 1.28),
        dashed=True,
        color=colors["training"],
    )
    route(
        [(2.0, 0.75), (2.0, 0.35), (8.68, 0.35), (8.68, 2.6)],
        dashed=True,
        color=colors["training"],
    )
    route(
        [(6.7, 1.28), (8.2, 1.28), (8.2, 2.6)],
        dashed=True,
        color=colors["training"],
    )
    arrow(
        (11.7, 1.8),
        (11.7, 2.6),
        dashed=True,
        color="#78909C",
    )

    axis.plot(
        [0.75, 1.35],
        [7.25, 7.25],
        color=colors["line"],
        linewidth=1.5,
    )
    axis.text(1.48, 7.25, "deployment / metric forward", va="center", fontsize=8.3)
    axis.plot(
        [0.75, 1.35],
        [6.85, 6.85],
        color=colors["training"],
        linewidth=1.5,
        linestyle="--",
    )
    axis.text(1.48, 6.85, "training-only supervision", va="center", fontsize=8.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    report = {
        "schema_version": 1,
        "valid": output.is_file() and pdf_path.is_file(),
        "png": output.resolve().as_posix(),
        "pdf": pdf_path.resolve().as_posix(),
        "claim_boundaries": {
            "metric_forward_inputs": ["RGB"],
            "qwen_rule_target_training_only": True,
            "hardware_sensor_used_in_formal_metrics": False,
            "rgb_d_input": False,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(
        json.dumps(
            draw_joint_architecture(parse_args().output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
