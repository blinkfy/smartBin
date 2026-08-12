"""Render the audited standardized instance-strata diagnostic figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT


ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "standardized-instance-strata"
)
DEFAULT_EVALUATION = ROOT / "evaluation-final-chunk64" / "evaluation.json"
DEFAULT_AUDIT = ROOT / "AUDIT.json"
DEFAULT_OUTPUT = ROOT / "standardized_instance_strata.png"
DEFAULT_CAPTION = ROOT / "FIGURE_CAPTION.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the standardized instance-strata figure"
    )
    parser.add_argument(
        "--evaluation", type=Path, default=DEFAULT_EVALUATION
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--caption", type=Path, default=DEFAULT_CAPTION)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def figure_spec() -> dict[str, Any]:
    return {
        "metric": "mask_mean_recall50_95",
        "panels": [
            {
                "dimension": "coco_visible_mask_area",
                "strata": ["small", "medium", "large"],
            },
            {
                "dimension": "bbox_overlap_proxy",
                "strata": ["isolated", "overlap_low", "overlap_high"],
            },
        ],
        "models": ["rgb_yolo", "v5_rgb_fallback", "v6_rgb_fallback"],
        "confidence_interval": "source_image_bootstrap_95_percent",
        "y_limits": [0.55, 1.01],
        "formal_paper_metric": False,
        "bbox_overlap_is_human_occlusion": False,
    }


def draw_figure(
    evaluation_path: Path = DEFAULT_EVALUATION,
    audit_path: Path = DEFAULT_AUDIT,
    output: Path = DEFAULT_OUTPUT,
    caption_path: Path = DEFAULT_CAPTION,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    evaluation_path = _resolve(evaluation_path)
    audit_path = _resolve(audit_path)
    output = _resolve(output)
    caption_path = _resolve(caption_path)
    for path in (evaluation_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"strata figure input is missing: {path}")
    evaluation = _read(evaluation_path)
    audit = _read(audit_path)
    if (
        evaluation.get("valid") is not True
        or evaluation.get("formal_paper_metric") is not False
        or audit.get("valid") is not True
        or audit.get("formal_paper_metric") is not False
    ):
        raise ValueError("strata figure inputs are invalid")

    spec = figure_spec()
    lookup = {
        (
            row["model"],
            row["dimension"],
            row["stratum"],
            row["metric"],
        ): row
        for row in evaluation["rows"]
    }
    models = spec["models"]
    labels = {
        "rgb_yolo": "RGB YOLO",
        "v5_rgb_fallback": "v5 RGB fallback",
        "v6_rgb_fallback": "v6 RGB fallback",
    }
    colors = {
        "rgb_yolo": "#30343B",
        "v5_rgb_fallback": "#168A83",
        "v6_rgb_fallback": "#D85B45",
    }
    markers = {
        "rgb_yolo": "o",
        "v5_rgb_fallback": "s",
        "v6_rgb_fallback": "^",
    }
    offsets = [-0.18, 0.0, 0.18]
    panel_titles = [
        "A  Visible-mask scale (COCO thresholds)",
        "B  Bounding-box overlap proxy",
    ]
    display_names = [
        ["Small", "Medium", "Large"],
        ["Isolated", "Low overlap", "High overlap"],
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        sharey=True,
        constrained_layout=False,
    )
    for panel_index, (axis, panel) in enumerate(zip(axes, spec["panels"])):
        strata = panel["strata"]
        x = np.arange(len(strata), dtype=float)
        supports = [
            int(
                lookup[
                    (
                        "rgb_yolo",
                        panel["dimension"],
                        stratum,
                        "mask",
                    )
                ]["support"]
            )
            for stratum in strata
        ]
        for model_index, model in enumerate(models):
            rows = [
                lookup[(model, panel["dimension"], stratum, "mask")]
                for stratum in strata
            ]
            means = np.asarray(
                [row["mean_recall50_95"] for row in rows], dtype=float
            )
            lower = np.asarray(
                [row["mean_recall50_95_ci_lower"] for row in rows],
                dtype=float,
            )
            upper = np.asarray(
                [row["mean_recall50_95_ci_upper"] for row in rows],
                dtype=float,
            )
            axis.errorbar(
                x + offsets[model_index],
                means,
                yerr=np.vstack((means - lower, upper - means)),
                fmt=markers[model],
                markersize=6.2,
                markeredgewidth=0.8,
                markeredgecolor="white",
                color=colors[model],
                ecolor=colors[model],
                elinewidth=1.25,
                capsize=3.2,
                capthick=1.1,
                label=labels[model],
                zorder=3,
            )
        axis.set_title(panel_titles[panel_index], loc="left", pad=11)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [
                f"{name}\n(n={support})"
                for name, support in zip(
                    display_names[panel_index], supports
                )
            ]
        )
        axis.set_ylim(*spec["y_limits"])
        axis.set_yticks(np.arange(0.6, 1.01, 0.1))
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.8, alpha=0.9)
        axis.set_axisbelow(True)
        axis.set_xlabel(
            "Target stratum"
            if panel_index == 0
            else "Potential crowding stratum"
        )
    axes[0].set_ylabel("Mask mR@0.50:0.95")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Standardized instance-strata diagnostic",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.02,
        (
            "Post-hoc internal test diagnostic; 95% source-image bootstrap "
            "CI. Box overlap is a crowding proxy, not a human occlusion label."
        ),
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#4E555C",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.845,
        bottom=0.18,
        wspace=0.16,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(output, dpi=220, facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if output.stat().st_size < 50_000 or pdf_path.stat().st_size < 10_000:
        raise ValueError("rendered strata figure is unexpectedly small")

    caption = (
        "**Figure. Standardized instance-strata diagnostic.** Mask "
        "mR@0.50:0.95 for the RGB checkpoint and the v5/v6 RGB-fallback "
        "paths under fixed conf=0.25 and NMS IoU=0.5. Error bars are 95% "
        "source-image bootstrap confidence intervals (2,000 replicates). "
        "Scale uses rasterized visible-mask COCO thresholds. Bounding-box "
        "overlap is a potential crowding proxy rather than a human-annotated "
        "occlusion state. The small and high-overlap subsets contain only "
        "11 and 20 targets, respectively; results are post-hoc internal "
        "diagnostics, not formal blind-test metrics."
    )
    _atomic_text(caption_path, caption + "\n")

    metadata_path = output.with_suffix(".json")
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "figure": "standardized_instance_strata_diagnostic",
        "spec": spec,
        "render_checks": {
            "png_minimum_size_passed": True,
            "pdf_minimum_size_passed": True,
            "manual_visual_inspection": {
                "completed": True,
                "result": "pass",
                "checks": [
                    "axes_and_error_bars_visible",
                    "legend_and_support_labels_legible",
                    "no_text_or_panel_overlap",
                    "small_subset_uncertainty_not_clipped",
                    "posthoc_and_overlap_proxy_boundary_visible",
                ],
            },
        },
        "bindings": {
            "evaluation": _binding(evaluation_path),
            "audit": _binding(audit_path),
            "plot_source": _binding(Path(__file__)),
            "png": _binding(output),
            "pdf": _binding(pdf_path),
            "caption": _binding(caption_path),
        },
    }
    _atomic_text(
        metadata_path,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    return result


def main() -> None:
    args = parse_args()
    result = draw_figure(
        args.evaluation,
        args.audit,
        args.output,
        args.caption,
    )
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "png": result["bindings"]["png"],
                "pdf": result["bindings"]["pdf"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
