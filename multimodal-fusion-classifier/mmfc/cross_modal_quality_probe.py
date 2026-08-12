"""Cross-modal raw-grid reconstruction and sensor-quality probe."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import (
    SENSOR_CROSS_MODAL_QUALITY_DIM,
    SENSOR_GRID_QUALITY_DIM,
    cross_modal_sensor_grid_quality_features,
)
from .quality_probe import quality_regression_metrics
from .robustness import default_sensor_scenarios
from .training import degrade_sensor_grid, set_seed


CROSS_MODAL_QUALITY_FEATURE_NAMES = (
    *(f"observed_{index}" for index in range(SENSOR_GRID_QUALITY_DIM)),
    *(f"predicted_{index}" for index in range(SENSOR_GRID_QUALITY_DIM)),
    "mean_absolute_difference",
    "root_mean_square_difference",
    "maximum_absolute_difference",
    "absolute_mean_bias",
    "cosine_similarity",
    "horizontal_gradient_difference",
    "vertical_gradient_difference",
    "availability",
)
CROSS_MODAL_QUALITY_DIM = len(CROSS_MODAL_QUALITY_FEATURE_NAMES)
if CROSS_MODAL_QUALITY_DIM != SENSOR_CROSS_MODAL_QUALITY_DIM:
    raise RuntimeError("cross-modal quality feature names do not match model features")


class SensorGridDecoderProbe(nn.Sequential):
    """Decode a clean 4x4 grid from a visual-text sensor latent."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        hidden_dim = max(4, int(hidden_dim))
        super().__init__(
            nn.Linear(int(latent_dim), hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 16),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, latent: Tensor) -> Tensor:
        return torch.sigmoid(super().forward(latent)).reshape(-1, 1, 4, 4)


class CrossModalQualityProbe(nn.Sequential):
    """Estimate observation quality from predicted and observed raw grids."""

    def __init__(self, hidden_dim: int = 32) -> None:
        hidden_dim = max(4, int(hidden_dim))
        super().__init__(
            nn.Linear(CROSS_MODAL_QUALITY_DIM, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)


def cross_modal_grid_quality_features(
    predicted_grid: Tensor,
    observed_grid: Tensor,
    availability: Tensor,
) -> Tensor:
    """Build raw-grid and cross-modal discrepancy features."""

    return cross_modal_sensor_grid_quality_features(
        predicted_grid,
        observed_grid,
        availability,
    )


def _pairwise_ranking_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    margin: float,
) -> Tensor:
    values = logits.reshape(-1)
    target_values = targets.reshape(-1)
    target_difference = target_values[:, None] - target_values[None, :]
    logit_difference = values[:, None] - values[None, :]
    mask = torch.triu(
        torch.ones_like(target_difference, dtype=torch.bool),
        diagonal=1,
    ) & (target_difference.abs() >= float(margin))
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    rank_targets = (target_difference[mask] > 0.0).to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logit_difference[mask],
        rank_targets,
    )


def fit_cross_modal_quality_probe(
    train_latents: Tensor,
    train_grids: Tensor,
    degradation_config: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int = 42,
    decoder_epochs: int = 50,
    quality_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    decoder_hidden_dim: int = 64,
    quality_hidden_dim: int = 32,
    ranking_weight: float = 0.2,
    ranking_margin: float = 0.05,
    clean_weight: float = 3.0,
) -> tuple[
    SensorGridDecoderProbe,
    CrossModalQualityProbe,
    list[dict[str, float | int]],
]:
    """Fit a clean-grid decoder, then a paired quality estimator."""

    if train_latents.ndim != 2 or train_grids.ndim != 4:
        raise ValueError("train_latents/train_grids must have shapes [N,D]/[N,1,4,4]")
    if train_latents.shape[0] != train_grids.shape[0] or tuple(
        train_grids.shape[1:]
    ) != (1, 4, 4):
        raise ValueError("train latent and grid sample counts/shapes are incompatible")
    if decoder_epochs <= 0 or quality_epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be greater than 0")
    if learning_rate <= 0.0 or ranking_weight < 0.0:
        raise ValueError("learning_rate must be positive and ranking_weight non-negative")
    if ranking_margin < 0.0 or clean_weight < 1.0:
        raise ValueError("ranking_margin must be non-negative and clean_weight >= 1")

    set_seed(seed)
    latents = train_latents.detach().cpu().float()
    clean_grids = train_grids.detach().cpu().float()
    decoder = SensorGridDecoderProbe(
        latents.shape[1],
        hidden_dim=decoder_hidden_dim,
    ).to(device)
    decoder_optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=learning_rate,
    )
    history: list[dict[str, float | int]] = []

    for epoch in range(decoder_epochs):
        order = torch.randperm(
            latents.shape[0],
            generator=torch.Generator().manual_seed(seed * 1000 + epoch),
        )
        decoder.train()
        total = 0.0
        count = 0
        for start in range(0, latents.shape[0], batch_size):
            indices = order[start : start + batch_size]
            batch_latents = latents[indices].to(device)
            batch_grids = clean_grids[indices].to(device)
            decoder_optimizer.zero_grad(set_to_none=True)
            prediction = decoder(batch_latents)
            loss = F.smooth_l1_loss(prediction, batch_grids)
            loss.backward()
            decoder_optimizer.step()
            total += float(loss.detach()) * indices.numel()
            count += indices.numel()
        history.append(
            {
                "stage": "decoder",
                "epoch": epoch + 1,
                "loss": total / max(count, 1),
            }
        )

    decoder.eval()
    with torch.inference_mode():
        predicted_clean = torch.cat(
            [
                decoder(latents[start : start + batch_size].to(device)).cpu()
                for start in range(0, latents.shape[0], batch_size)
            ],
            dim=0,
        )

    quality_probe = CrossModalQualityProbe(
        hidden_dim=quality_hidden_dim
    ).to(device)
    quality_optimizer = torch.optim.AdamW(
        quality_probe.parameters(),
        lr=learning_rate,
    )
    for epoch in range(quality_epochs):
        degradation = degrade_sensor_grid(
            clean_grids,
            dict(degradation_config),
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        features = cross_modal_grid_quality_features(
            predicted_clean,
            degradation["degraded_grid"],
            degradation["availability"],
        )
        targets = degradation["quality_target"]
        order = torch.randperm(
            latents.shape[0],
            generator=torch.Generator().manual_seed(seed * 2000 + epoch),
        )
        quality_probe.train()
        total = 0.0
        calibration_total = 0.0
        ranking_total = 0.0
        count = 0
        for start in range(0, latents.shape[0], batch_size):
            indices = order[start : start + batch_size]
            batch_features = features[indices].to(device)
            batch_targets = targets[indices].to(device)
            quality_optimizer.zero_grad(set_to_none=True)
            logits = quality_probe(batch_features)
            per_value = F.binary_cross_entropy_with_logits(
                logits,
                batch_targets,
                reduction="none",
            )
            weights = torch.where(
                batch_targets >= 1.0 - 1e-6,
                torch.full_like(per_value, float(clean_weight)),
                torch.ones_like(per_value),
            )
            calibration = (per_value * weights).sum() / weights.sum()
            ranking = _pairwise_ranking_loss(
                logits,
                batch_targets,
                margin=ranking_margin,
            )
            loss = calibration + float(ranking_weight) * ranking
            loss.backward()
            quality_optimizer.step()
            sample_count = indices.numel()
            total += float(loss.detach()) * sample_count
            calibration_total += float(calibration.detach()) * sample_count
            ranking_total += float(ranking.detach()) * sample_count
            count += sample_count
        history.append(
            {
                "stage": "quality",
                "epoch": epoch + 1,
                "loss": total / max(count, 1),
                "calibration_loss": calibration_total / max(count, 1),
                "ranking_loss": ranking_total / max(count, 1),
            }
        )
    return decoder, quality_probe, history


def evaluate_cross_modal_quality_probe(
    decoder: nn.Module,
    quality_probe: nn.Module,
    test_latents: Tensor,
    test_grids: Tensor,
    *,
    device: torch.device,
    seed: int = 2026,
    batch_size: int = 256,
    scenarios: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate clean-grid reconstruction and seeded quality calibration."""

    if test_latents.shape[0] != test_grids.shape[0]:
        raise ValueError("test latent and grid sample counts must match")
    decoder.eval()
    quality_probe.eval()
    with torch.inference_mode():
        predicted_clean = torch.cat(
            [
                decoder(test_latents[start : start + batch_size].to(device)).cpu()
                for start in range(0, test_latents.shape[0], batch_size)
            ],
            dim=0,
        )
    clean_error = predicted_clean - test_grids.cpu()
    decoder_metrics = {
        "mae": float(clean_error.abs().mean()),
        "rmse": float(clean_error.square().mean().sqrt()),
    }

    rows: list[dict[str, Any]] = []
    clean = test_grids.detach().cpu().float()
    for scenario_name, scenario_config in (
        scenarios or default_sensor_scenarios()
    ).items():
        degradation = degrade_sensor_grid(
            clean,
            scenario_config or {"enabled": False},
            generator=torch.Generator().manual_seed(seed),
        )
        features = cross_modal_grid_quality_features(
            predicted_clean,
            degradation["degraded_grid"],
            degradation["availability"],
        )
        with torch.inference_mode():
            raw_predictions = torch.cat(
                [
                    torch.sigmoid(
                        quality_probe(
                            features[start : start + batch_size].to(device)
                        )
                    ).cpu()
                    for start in range(0, features.shape[0], batch_size)
                ],
                dim=0,
            )
        predictions = (
            raw_predictions * degradation["availability"]
        ).numpy().reshape(-1)
        targets = degradation["quality_target"].numpy().reshape(-1)
        availability_predictions = degradation["availability"].numpy().reshape(-1)
        metrics = quality_regression_metrics(targets, predictions)
        baseline = quality_regression_metrics(
            targets,
            availability_predictions,
        )
        rows.append(
            {
                "scenario": scenario_name,
                "sample_count": int(clean.shape[0]),
                **metrics,
                "availability_baseline_mae": baseline["mae"],
                "availability_baseline_rmse": baseline["rmse"],
                "availability_baseline_pearson": baseline["pearson"],
                "mae_improvement": float(baseline["mae"])
                - float(metrics["mae"]),
            }
        )
    return rows, decoder_metrics


def aggregate_cross_modal_quality_runs(
    runs: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate per-seed quality metrics with sample standard deviation."""

    if not runs:
        raise ValueError("runs must not be empty")
    seeds = list(runs)
    scenario_order = [str(row["scenario"]) for row in runs[seeds[0]]]
    for seed, rows in runs.items():
        if [str(row["scenario"]) for row in rows] != scenario_order:
            raise ValueError(f"scenario order mismatch for seed {seed}")
    aggregated: list[dict[str, Any]] = []
    ignored = {"scenario", "sample_count"}
    for scenario_index, scenario in enumerate(scenario_order):
        source_rows = [runs[seed][scenario_index] for seed in seeds]
        row: dict[str, Any] = {
            "scenario": scenario,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
            "sample_count": int(source_rows[0]["sample_count"]),
        }
        keys = [key for key in source_rows[0] if key not in ignored]
        for key in keys:
            values = [
                float(source[key])
                for source in source_rows
                if source.get(key) is not None
                and math.isfinite(float(source[key]))
            ]
            row[f"{key}_mean"] = (
                float(np.mean(values)) if values else None
            )
            row[f"{key}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ) if values else None
            row[f"{key}_valid_seed_count"] = len(values)
        aggregated.append(row)
    return aggregated


def write_cross_modal_quality_probe_artifacts(
    decoder: nn.Module,
    quality_probe: nn.Module,
    history: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    metadata: Mapping[str, Any],
    decoder_metrics: Mapping[str, float],
) -> dict[str, str]:
    """Write UTF-8 tables, checkpoint, and a compact comparison plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cross_modal_quality_summary.csv"
    fields = list(rows[0]) if rows else ["scenario"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "cross_modal_quality_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metadata": dict(metadata),
                "decoder_metrics": dict(decoder_metrics),
                "history": list(history),
                "summary": list(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = output_dir / "cross_modal_quality_probe.pt"
    torch.save(
        {
            "decoder_state": decoder.state_dict(),
            "quality_probe_state": quality_probe.state_dict(),
            "feature_names": list(CROSS_MODAL_QUALITY_FEATURE_NAMES),
            "metadata": dict(metadata),
            "decoder_metrics": dict(decoder_metrics),
        },
        checkpoint_path,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["scenario"]) for row in rows]
    indices = np.arange(len(rows))
    figure, axis = plt.subplots(
        figsize=(max(10.0, len(rows) * 1.05), 5.5)
    )
    axis.plot(
        indices,
        [float(row["mae"]) for row in rows],
        marker="o",
        linewidth=2,
        label="Cross-modal probe",
    )
    axis.plot(
        indices,
        [float(row["availability_baseline_mae"]) for row in rows],
        marker="s",
        linewidth=2,
        label="Availability baseline",
    )
    axis.set_xticks(indices, labels, rotation=35, ha="right")
    axis.set_ylabel("Quality MAE")
    axis.set_title("Cross-modal sensor quality probe")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / "cross_modal_quality_probe.png"
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return {
        "summary_csv": str(csv_path.resolve()),
        "summary_json": str(summary_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "plot_png": str(plot_path.resolve()),
    }


def write_cross_modal_quality_multiseed_artifacts(
    runs: Mapping[int, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    *,
    metadata: Mapping[str, Any],
    decoder_metrics: Mapping[str, float],
) -> dict[str, str]:
    """Write aggregate cross-modal probe tables and error-bar plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate_cross_modal_quality_runs(runs)
    csv_path = output_dir / "cross_modal_quality_multiseed_summary.csv"
    fields = list(rows[0]) if rows else ["scenario"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "cross_modal_quality_multiseed_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "metadata": dict(metadata),
                "decoder_metrics": dict(decoder_metrics),
                "summary": rows,
                "per_seed": {
                    str(seed): list(seed_rows)
                    for seed, seed_rows in runs.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["scenario"]) for row in rows]
    indices = np.arange(len(rows))
    figure, axis = plt.subplots(
        figsize=(max(10.0, len(rows) * 1.05), 5.5)
    )
    axis.errorbar(
        indices,
        [float(row["mae_mean"]) for row in rows],
        yerr=[float(row["mae_std"]) for row in rows],
        marker="o",
        linewidth=2,
        capsize=3,
        label="Cross-modal probe",
    )
    axis.errorbar(
        indices,
        [float(row["availability_baseline_mae_mean"]) for row in rows],
        yerr=[float(row["availability_baseline_mae_std"]) for row in rows],
        marker="s",
        linewidth=2,
        capsize=3,
        label="Availability baseline",
    )
    axis.set_xticks(indices, labels, rotation=35, ha="right")
    axis.set_ylabel("Quality MAE (mean +/- sample std)")
    axis.set_title("Cross-modal sensor quality probe across seeds")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / "cross_modal_quality_multiseed.png"
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return {
        "summary_csv": str(csv_path.resolve()),
        "summary_json": str(json_path.resolve()),
        "plot_png": str(plot_path.resolve()),
    }
