"""Compare visual-only, LMF, and proposed models under deployment stress."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_recent_baseline import _validate_checkpoint as validate_baseline
from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.evaluation import compute_classification_metrics
from mmfc.model import build_production_model
from mmfc.recent_baselines import build_recent_baseline_model
from mmfc.training import ManifestDataset, load_config, resolve_project_path


ROOT = Path(__file__).resolve().parent
CONDITIONS = (
    "clean",
    "sensor_missing",
    "sensor_noise_high",
    "sensor_bias_high",
    "sensor_stuck",
    "visual_blur",
    "visual_low_light",
    "visual_occlusion",
)
MODEL_RUNS = {
    "visual_only": {
        42: (
            "configs/modality-baselines/visual_only_seed42.json",
            "checkpoints/ablations/vision_only/best.pt",
        ),
        43: (
            "configs/modality-baselines/visual_only_seed43.json",
            "checkpoints/modality-baselines/visual_only/seed43/best.pt",
        ),
        44: (
            "configs/modality-baselines/visual_only_seed44.json",
            "checkpoints/modality-baselines/visual_only/seed44/best.pt",
        ),
    },
    "lmf": {
        seed: (
            f"configs/standard-fusion-baselines/lmf_seed{seed}.json",
            f"checkpoints/standard-fusion-baselines/lmf/seed{seed}/best.pt",
        )
        for seed in (42, 43, 44)
    },
    "proposed": {
        42: (
            "configs/ablations/full_cross_modal_quality.json",
            "checkpoints/ablations/full_cross_modal_quality/best.pt",
        ),
        43: (
            "configs/replicates/full_cross_modal_quality_seed43.json",
            "checkpoints/ablations/v3_seed_replicates/full_cross_modal_quality_seed43/best.pt",
        ),
        44: (
            "configs/replicates/full_cross_modal_quality_seed44.json",
            "checkpoints/ablations/v3_seed_replicates/full_cross_modal_quality_seed44/best.pt",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "qwen-material" / "multimodal-utility",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--perturbation-seed", type=int, default=20260810)
    return parser.parse_args()


def sensor_condition(
    name: str,
    sensor: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    availability = torch.ones(sensor.shape[0], 1, dtype=sensor.dtype)
    if name == "sensor_missing":
        return torch.zeros_like(sensor), torch.zeros_like(availability)
    if name == "sensor_noise_high":
        noise = torch.randn(sensor.shape, generator=generator, dtype=sensor.dtype)
        return (sensor + 0.15 * noise).clamp(0.0, 1.0), availability
    if name == "sensor_bias_high":
        rows = torch.linspace(-1.0, 1.0, 4, dtype=sensor.dtype).view(1, 1, 4, 1)
        columns = torch.linspace(-1.0, 1.0, 4, dtype=sensor.dtype).view(1, 1, 1, 4)
        direction = torch.rand((sensor.shape[0], 1, 1, 1), generator=generator)
        pattern = torch.where(direction < 0.5, rows, columns)
        return (sensor + 0.15 * pattern).clamp(0.0, 1.0), availability
    if name == "sensor_stuck":
        return sensor[:, :, :1, :1].expand_as(sensor).clone(), availability
    return sensor, availability


def visual_condition(name: str, images: torch.Tensor) -> torch.Tensor:
    if name == "visual_blur":
        return F.avg_pool2d(images, kernel_size=9, stride=1, padding=4)
    if name == "visual_low_light":
        return (images * 0.35).clamp(0.0, 1.0)
    if name == "visual_occlusion":
        result = images.clone()
        height, width = result.shape[-2:]
        half_h = int(round(0.20 * height))
        half_w = int(round(0.20 * width))
        center_h, center_w = height // 2, width // 2
        result[
            :,
            :,
            center_h - half_h : center_h + half_h,
            center_w - half_w : center_w + half_w,
        ] = 0.0
        return result
    return images


def build_model(
    family: str,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, tuple[str, ...]]:
    if family == "lmf":
        class_names = validate_baseline(checkpoint, config)
        model = build_recent_baseline_model(config).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        class_names = validated_checkpoint_class_names(checkpoint, config)
        model = build_production_model(config).to(device)
        load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    return model, tuple(class_names)


def evaluate_run(
    family: str,
    seed: int,
    config_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    perturbation_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, class_names = build_model(family, config, checkpoint, device)
    data = config["data"]
    dataset = ManifestDataset(
        data["test_manifest"],
        class_names=class_names,
        image_size=int(data.get("image_size", 480)),
        sensor_manifest=data["sensor_manifest"],
        preprocess=config.get("preprocess", {}),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    generators = {
        condition: torch.Generator().manual_seed(perturbation_seed)
        for condition in CONDITIONS
    }
    collectors = {
        condition: {
            "targets": [],
            "probabilities": [],
            "sample_ids": [],
            "source_image_ids": [],
        }
        for condition in CONDITIONS
    }
    with torch.inference_mode():
        for batch in loader:
            clean_images = batch["image"]
            clean_sensor = batch["sensor"]
            targets = batch["label"].numpy()
            for condition in CONDITIONS:
                images = visual_condition(condition, clean_images).to(
                    device, non_blocking=True
                )
                sensor, availability = sensor_condition(
                    condition,
                    clean_sensor,
                    generator=generators[condition],
                )
                logits = model(
                    images,
                    sensor.to(device, non_blocking=True),
                    sensor_available=availability.to(device, non_blocking=True),
                )
                if isinstance(logits, tuple):
                    logits = logits[0]
                collector = collectors[condition]
                collector["targets"].append(targets.copy())
                collector["probabilities"].append(
                    torch.softmax(logits, dim=1).cpu().numpy()
                )
                collector["sample_ids"].extend(str(value) for value in batch["sample_id"])
                collector["source_image_ids"].extend(
                    str(value) for value in batch["source_image_id"]
                )

    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, Any]] = {}
    for condition, collector in collectors.items():
        targets = np.concatenate(collector["targets"])
        probabilities = np.concatenate(collector["probabilities"])
        computed = compute_classification_metrics(
            targets,
            probabilities,
            class_names,
            ece_bins=15,
        )
        metrics = computed["metrics"]
        rows.append(
            {
                "family": family,
                "training_seed": seed,
                "condition": condition,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "nll": float(metrics["negative_log_likelihood"]),
                "brier": float(metrics["multiclass_brier_score"]),
                "ece": float(metrics["ece"]),
            }
        )
        arrays[condition] = {
            "targets": targets,
            "probabilities": probabilities,
            "sample_ids": collector["sample_ids"],
            "source_image_ids": collector["source_image_ids"],
            "class_names": class_names,
        }
    del model, checkpoint, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, arrays


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for family in MODEL_RUNS:
        for condition in CONDITIONS:
            selected = [
                row
                for row in rows
                if row["family"] == family and row["condition"] == condition
            ]
            output: dict[str, Any] = {
                "family": family,
                "condition": condition,
                "training_seed_count": len(selected),
            }
            for metric in ("accuracy", "macro_f1", "nll", "brier", "ece"):
                values = [float(row[metric]) for row in selected]
                output[f"{metric}_mean"] = mean(values)
                output[f"{metric}_std"] = stdev(values)
            result.append(output)
    return result


def low_confidence_subset(
    arrays: dict[tuple[str, int], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    visual_runs = [arrays[("visual_only", seed)]["clean"] for seed in (42, 43, 44)]
    sample_ids = visual_runs[0]["sample_ids"]
    if any(run["sample_ids"] != sample_ids for run in visual_runs[1:]):
        raise RuntimeError("visual-only sample order differs across seeds")
    target = visual_runs[0]["targets"]
    confidence = np.mean(
        [run["probabilities"].max(axis=1) for run in visual_runs],
        axis=0,
    )
    cutoff = float(np.quantile(confidence, 0.25))
    selected = confidence <= cutoff
    rows = []
    for family in MODEL_RUNS:
        for seed in (42, 43, 44):
            run = arrays[(family, seed)]["clean"]
            computed = compute_classification_metrics(
                target[selected],
                run["probabilities"][selected],
                run["class_names"],
                ece_bins=15,
            )
            metrics = computed["metrics"]
            rows.append(
                {
                    "family": family,
                    "training_seed": seed,
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "nll": float(metrics["negative_log_likelihood"]),
                }
            )
    summary = []
    for family in MODEL_RUNS:
        family_rows = [row for row in rows if row["family"] == family]
        summary.append(
            {
                "family": family,
                "accuracy_mean": mean(row["accuracy"] for row in family_rows),
                "accuracy_std": stdev(row["accuracy"] for row in family_rows),
                "macro_f1_mean": mean(row["macro_f1"] for row in family_rows),
                "macro_f1_std": stdev(row["macro_f1"] for row in family_rows),
                "nll_mean": mean(row["nll"] for row in family_rows),
                "nll_std": stdev(row["nll"] for row in family_rows),
            }
        )
    return {
        "selection_model": "mean confidence of three visual-only checkpoints",
        "selection_fraction": 0.25,
        "confidence_cutoff": cutoff,
        "sample_count": int(selected.sum()),
        "source_image_count": len(
            {
                visual_runs[0]["source_image_ids"][index]
                for index in np.flatnonzero(selected)
            }
        ),
        "summary": summary,
        "per_seed": rows,
    }


def write_figure(summary: list[dict[str, Any]], output: Path) -> None:
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
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "svg.hashsalt": "mmfc-multimodal-utility-2026",
        }
    )
    labels = {
        "sensor_missing": "Sensor\nmissing",
        "sensor_noise_high": "Sensor\nnoise",
        "sensor_bias_high": "Sensor\nbias",
        "sensor_stuck": "Sensor\nstuck",
        "visual_blur": "Visual\nblur",
        "visual_low_light": "Low\nlight",
        "visual_occlusion": "Visual\nocclusion",
    }
    stress_conditions = tuple(value for value in CONDITIONS if value != "clean")
    colors = {"proposed": "#3f7758", "lmf": "#756f86"}
    names = {"proposed": "Proposed", "lmf": "LMF"}
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 3.75), constrained_layout=True)
    x = np.arange(len(stress_conditions))
    width = 0.34
    for axis, metric, title in zip(
        axes,
        ("accuracy_mean", "macro_f1_mean"),
        (
            "(a) Accuracy change relative to Visual-only",
            "(b) Macro F1 change relative to Visual-only",
        ),
    ):
        visual = {
            condition: next(
                row[metric]
                for row in summary
                if row["family"] == "visual_only"
                and row["condition"] == condition
            )
            for condition in stress_conditions
        }
        for family, offset in (("proposed", -width / 2), ("lmf", width / 2)):
            deltas = [
                100.0
                * (
                    next(
                        row[metric]
                        for row in summary
                        if row["family"] == family
                        and row["condition"] == condition
                    )
                    - visual[condition]
                )
                for condition in stress_conditions
            ]
            axis.bar(
                x + offset,
                deltas,
                width=width,
                color=colors[family],
                label=names[family],
                edgecolor="white",
                linewidth=0.5,
            )
        axis.axhline(0.0, color="#30343b", linewidth=0.9)
        axis.axvspan(-0.5, 3.5, color="#f3f5f7", zorder=-2)
        axis.axvline(3.5, color="#c9cdd2", linewidth=0.7)
        axis.set_xticks(x)
        axis.set_xticklabels([labels[value] for value in stress_conditions])
        axis.set_ylabel("Difference (percentage points)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#d9dde2", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid loader settings")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_rows: list[dict[str, Any]] = []
    all_arrays: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for family, runs in MODEL_RUNS.items():
        for seed, (config_value, checkpoint_value) in runs.items():
            rows, arrays = evaluate_run(
                family,
                seed,
                resolve_project_path(config_value),
                resolve_project_path(checkpoint_value),
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                perturbation_seed=args.perturbation_seed,
            )
            all_rows.extend(rows)
            all_arrays[(family, seed)] = arrays
            print(json.dumps(rows, ensure_ascii=False), flush=True)

    summary_rows = aggregate(all_rows)
    report = {
        "valid": True,
        "training_seeds": [42, 43, 44],
        "perturbation_seed": args.perturbation_seed,
        "sample_count": 826,
        "condition_definitions": {
            "sensor_noise_high": "Gaussian cell noise, sigma=0.15, clipped to [0,1]",
            "sensor_bias_high": "row- or column-directional amplitude 0.15",
            "sensor_stuck": "top-left response replicated to all 16 sites",
            "visual_blur": "9x9 average filter after common preprocessing",
            "visual_low_light": "RGB intensity multiplied by 0.35",
            "visual_occlusion": "center 40% by 40% region set to zero",
        },
        "summary": summary_rows,
        "low_visual_confidence_subset": low_confidence_subset(all_arrays),
        "per_seed": all_rows,
        "claim_boundary": (
            "Deterministic software stress tests and an exploratory subset on "
            "the reused internal benchmark; not external deployment validation."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_figure(summary_rows, args.output / "multimodal_utility.svg")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
