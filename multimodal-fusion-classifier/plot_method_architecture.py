"""Render the implemented multimodal reliability architecture as a paper figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmfc.data import write_json
from mmfc.training import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the implemented multimodal method architecture"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def draw_method_architecture(output: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    colors = {
        "visual": "#D8EAF7",
        "text": "#DDF1E4",
        "sensor": "#FCE8C8",
        "quality": "#F7D7D9",
        "fusion": "#E8DDF1",
        "loss": "#ECEFF1",
        "border": "#263238",
        "muted": "#607D8B",
    }
    figure, axis = plt.subplots(figsize=(17.5, 8.5))
    axis.set_xlim(0.0, 17.5)
    axis.set_ylim(0.0, 8.5)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color: str,
        *,
        fontsize: float = 9.0,
        linewidth: float = 1.2,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.05",
            facecolor=color,
            edgecolor=colors["border"],
            linewidth=linewidth,
            zorder=3,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True,
            zorder=4,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = "#455A64",
        dashed: bool = False,
        width: float = 1.2,
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=width,
                linestyle="--" if dashed else "-",
                color=color,
                connectionstyle="arc3,rad=0.0",
                zorder=2,
            )
        )

    def routed_arrow(
        points: list[tuple[float, float]],
        *,
        color: str = "#455A64",
        dashed: bool = False,
        width: float = 1.2,
    ) -> None:
        style = "--" if dashed else "-"
        xs = [point[0] for point in points[:-1]]
        ys = [point[1] for point in points[:-1]]
        axis.plot(
            xs,
            ys,
            color=color,
            linestyle=style,
            linewidth=width,
            zorder=2,
        )
        arrow(
            points[-2],
            points[-1],
            color=color,
            dashed=dashed,
            width=width,
        )

    axis.text(
        8.75,
        8.22,
        "Cross-modal consistency-aware multimodal waste classifier",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )

    box(
        0.4,
        6.2,
        3.0,
        1.25,
        "RGB crop, 480 x 480\nPreprocess + YOLO11s-seg\n3-scale features (freeze first 5 layers)",
        colors["visual"],
        fontsize=8.6,
    )
    box(
        0.4,
        4.55,
        3.0,
        1.1,
        "19 class descriptions\nCLIP ViT-B/16 prototypes\nprototype encoder (freeze first 3 layers)",
        colors["text"],
        fontsize=8.5,
    )
    box(
        0.4,
        2.85,
        3.0,
        1.15,
        "4 x 4 Qwen/rule soft-sensor grid\nSensor encoder: 4 x 4 -> 16 x 16\n1 x 1 conv -> 512-D",
        colors["sensor"],
        fontsize=8.3,
    )
    box(
        0.4,
        1.65,
        1.35,
        0.7,
        "Availability a",
        colors["quality"],
        fontsize=8.3,
    )
    box(
        1.95,
        1.65,
        1.45,
        0.7,
        "Quality target q\n(training only)",
        colors["loss"],
        fontsize=7.8,
    )

    box(4.1, 6.38, 1.7, 0.85, "Dynamic visual\nvector v", colors["visual"])
    box(4.1, 4.68, 1.7, 0.85, "CLIP prototype\nbank E", colors["text"])
    box(
        4.1,
        3.0,
        1.7,
        0.85,
        "Observed sensor\nlatent s",
        colors["sensor"],
    )
    box(
        6.6,
        5.68,
        2.4,
        0.95,
        "Visual prototype query\nt_v = Query(v, E)",
        colors["text"],
        fontsize=8.3,
    )

    box(
        6.6,
        3.45,
        2.4,
        1.5,
        "Visual-text predictor\nmu_s, log variance_s\nand clean-grid decoder",
        colors["fusion"],
        fontsize=8.4,
    )
    box(
        9.55,
        3.45,
        2.3,
        1.5,
        "Quality evidence\nlatent discrepancy + cosine\nuncertainty + raw/cross-grid\n+ availability",
        colors["quality"],
        fontsize=8.0,
    )
    box(
        12.4,
        3.45,
        2.0,
        1.5,
        "Reliability head\nr in [0,1]\ncalibration + ranking",
        colors["quality"],
        fontsize=8.3,
    )
    box(
        9.55,
        5.68,
        2.3,
        0.95,
        "Fusion prototype query\nt = Query(v + s_eff, E)",
        colors["text"],
        fontsize=8.0,
    )
    box(
        12.4,
        1.75,
        2.0,
        1.0,
        "Reliable sensor feature\nmu_s + r (s - mu_s)",
        colors["sensor"],
        fontsize=8.1,
    )
    box(
        12.4,
        5.65,
        2.0,
        1.55,
        "Independent adapters\nFiLM-style modality affine\n8-head modality attention\n+ residual",
        colors["fusion"],
        fontsize=8.1,
    )
    box(
        15.35,
        5.75,
        1.75,
        1.35,
        "19-class logits\nobject-level\nprediction",
        "#D7E8D4",
        fontsize=8.5,
    )

    box(
        6.15,
        0.25,
        2.35,
        0.75,
        "Clean-latent heteroscedastic NLL\n+ variance prior",
        colors["loss"],
        fontsize=7.8,
    )
    box(
        8.85,
        0.25,
        2.2,
        0.75,
        "Clean-grid Smooth L1",
        colors["loss"],
        fontsize=8.0,
    )
    box(
        11.4,
        0.25,
        2.35,
        0.75,
        "Reliability BCE\n+ pairwise ranking",
        colors["loss"],
        fontsize=7.8,
    )
    box(
        14.1,
        0.25,
        2.15,
        0.75,
        "CE + Focal",
        colors["loss"],
        fontsize=8.0,
    )

    arrow((3.4, 6.83), (4.1, 6.83))
    arrow((3.4, 5.1), (4.1, 5.1))
    arrow((3.4, 3.43), (4.1, 3.43))

    routed_arrow([(5.8, 7.0), (11.95, 7.0), (12.4, 6.95)])
    arrow((11.85, 6.15), (12.4, 6.15))
    arrow((14.4, 6.43), (15.35, 6.43))

    arrow((5.8, 6.7), (6.6, 6.18))
    arrow((5.8, 5.1), (6.6, 5.9))
    arrow((5.8, 6.55), (6.6, 4.63))
    arrow((7.8, 5.68), (7.8, 4.95))
    arrow((9.0, 4.2), (9.55, 4.2))
    arrow((5.8, 3.43), (9.55, 3.82))
    routed_arrow(
        [(1.75, 2.15), (1.85, 2.55), (8.95, 2.55), (9.55, 3.62)],
        color=colors["muted"],
    )
    arrow((11.85, 4.2), (12.4, 4.2))

    arrow((5.8, 6.98), (9.55, 6.38))
    routed_arrow([(5.8, 5.0), (6.05, 5.48), (9.2, 5.48), (9.55, 5.92)])
    routed_arrow([(7.8, 3.45), (7.8, 2.25), (12.4, 2.25)])
    routed_arrow([(5.8, 3.25), (6.2, 1.45), (12.0, 1.45), (12.4, 2.05)])
    arrow((13.4, 3.45), (13.4, 2.75))
    routed_arrow([(13.05, 2.75), (12.05, 2.95), (12.05, 5.9), (11.85, 5.9)])
    routed_arrow([(14.4, 2.25), (14.8, 2.25), (14.8, 5.9), (14.4, 5.9)])

    arrow((7.35, 3.45), (7.35, 1.0), dashed=True, color=colors["muted"])
    arrow((8.3, 3.45), (9.95, 1.0), dashed=True, color=colors["muted"])
    arrow((13.4, 3.45), (12.58, 1.0), dashed=True, color=colors["muted"])
    routed_arrow(
        [(3.4, 1.85), (4.0, 1.15), (12.58, 1.15), (12.58, 1.0)],
        dashed=True,
        color=colors["muted"],
    )
    arrow((16.2, 5.75), (15.2, 1.0), dashed=True, color=colors["muted"])

    axis.text(
        0.4,
        7.75,
        "Inputs and encoders",
        fontsize=10.5,
        weight="bold",
        color=colors["border"],
    )
    axis.text(
        4.1,
        7.75,
        "Representations",
        fontsize=10.5,
        weight="bold",
        color=colors["border"],
    )
    axis.text(
        6.6,
        5.28,
        "Cross-modal quality and fallback",
        fontsize=9.8,
        weight="bold",
        color=colors["border"],
    )
    axis.text(
        12.4,
        7.75,
        "Reliability-aware fusion",
        fontsize=10.5,
        weight="bold",
        color=colors["border"],
    )
    axis.text(
        0.4,
        0.62,
        "Soft-sensor sidecar is image-conditioned; it is not a hardware measurement.",
        fontsize=8.0,
        color=colors["muted"],
    )

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {
        "output": str(output),
        "pdf": str(pdf_path),
        "scope": "implemented object-level classifier",
        "query_paths": {
            "predictor_context": "t_v = Query(v, E)",
            "fusion_context": "t = Query(v + s_eff, E)",
        },
        "soft_sensor_note": (
            "The current grid is image-conditioned Qwen/rule sidecar data, "
            "not independent hardware measurement."
        ),
    }


def main() -> None:
    args = parse_args()
    output = resolve_project_path(args.output)
    report = draw_method_architecture(output)
    report_path = output.with_suffix(".json")
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
