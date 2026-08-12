"""Lightweight probe for raw sensor-grid quality features."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import SENSOR_GRID_QUALITY_DIM, sensor_grid_quality_features
from .robustness import default_sensor_scenarios
from .training import (
    _sample_key,
    degrade_sensor_grid,
    load_sensor_grid,
    read_jsonl,
    resolve_project_path,
    set_seed,
)


QUALITY_FEATURE_NAMES = (
    "mean",
    "std",
    "dynamic_range",
    "horizontal_gradient",
    "vertical_gradient",
    "saturation_fraction",
    "flatness",
)


class SensorQualityProbe(nn.Sequential):
    def __init__(self, hidden_dim: int = 16) -> None:
        hidden_dim = max(2, int(hidden_dim))
        super().__init__(
            nn.Linear(SENSOR_GRID_QUALITY_DIM, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)


def load_split_sensor_grids(
    manifest_path: str | Path,
    sensor_manifest: str | Path,
) -> Tensor:
    """Load split-aligned 4x4 grids without reading any source images."""

    resolved_manifest = resolve_project_path(manifest_path)
    resolved_sensor_manifest = resolve_project_path(sensor_manifest)
    sidecar: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(resolved_sensor_manifest):
        sample_id = _sample_key(record)
        if not sample_id:
            continue
        if sample_id in sidecar:
            raise ValueError(f"duplicate sensor sample id: {sample_id}")
        sidecar[sample_id] = record

    grids: list[Tensor] = []
    for record in read_jsonl(resolved_manifest):
        grids.append(
            load_sensor_grid(
                record,
                manifest_dir=resolved_manifest.parent,
                sidecar=sidecar,
            )
        )
    if not grids:
        raise ValueError(f"manifest contains no sensor samples: {resolved_manifest}")
    return torch.stack(grids, dim=0)


def fit_quality_probe(
    train_grids: Tensor,
    degradation_config: Mapping[str, Any],
    *,
    device: torch.device,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-3,
    hidden_dim: int = 16,
) -> tuple[SensorQualityProbe, list[dict[str, float | int]]]:
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than 0")
    set_seed(seed)
    clean = train_grids.detach().cpu().float()
    model = SensorQualityProbe(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        degradation = degrade_sensor_grid(
            clean,
            dict(degradation_config),
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        features = sensor_grid_quality_features(
            degradation["degraded_grid"]
        )
        availability = degradation["availability"]
        targets = degradation["quality_target"]
        order = torch.randperm(
            clean.shape[0],
            generator=torch.Generator().manual_seed(seed * 1000 + epoch),
        )
        model.train()
        total_loss = 0.0
        sample_count = 0
        for start in range(0, clean.shape[0], batch_size):
            indices = order[start : start + batch_size]
            batch_features = features[indices].to(device)
            batch_availability = availability[indices].to(device)
            batch_targets = targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_prediction = torch.sigmoid(model(batch_features))
            prediction = raw_prediction * batch_availability
            loss = F.binary_cross_entropy(
                prediction.clamp(1e-6, 1.0 - 1e-6),
                batch_targets,
            )
            loss.backward()
            optimizer.step()
            count = indices.numel()
            total_loss += float(loss.detach()) * count
            sample_count += count
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / max(sample_count, 1),
            }
        )
    return model, history


def quality_regression_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float | None]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("quality targets and predictions must share a non-empty shape")
    error = predictions - targets
    pearson = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if np.std(predictions) > 1e-12 and np.std(targets) > 1e-12
        else None
    )
    return {
        "target_mean": float(targets.mean()),
        "prediction_mean": float(predictions.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_error": float(error.mean()),
        "pearson": pearson,
    }


def evaluate_quality_probe(
    model: nn.Module,
    test_grids: Tensor,
    *,
    device: torch.device,
    seed: int = 2026,
    scenarios: Mapping[str, Mapping[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    scenarios = scenarios or default_sensor_scenarios()
    clean = test_grids.detach().cpu().float()
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for scenario_name, scenario_config in scenarios.items():
            degradation = degrade_sensor_grid(
                clean,
                scenario_config or {"enabled": False},
                generator=torch.Generator().manual_seed(seed),
            )
            features = sensor_grid_quality_features(
                degradation["degraded_grid"]
            ).to(device)
            availability = degradation["availability"]
            targets = degradation["quality_target"].numpy().reshape(-1)
            raw_prediction = torch.sigmoid(model(features)).cpu()
            predictions = (
                raw_prediction * availability
            ).numpy().reshape(-1)
            availability_predictions = availability.numpy().reshape(-1)
            probe_metrics = quality_regression_metrics(targets, predictions)
            baseline_metrics = quality_regression_metrics(
                targets, availability_predictions
            )
            rows.append(
                {
                    "scenario": scenario_name,
                    "sample_count": int(clean.shape[0]),
                    **probe_metrics,
                    "availability_baseline_mae": baseline_metrics["mae"],
                    "availability_baseline_rmse": baseline_metrics["rmse"],
                    "availability_baseline_pearson": baseline_metrics[
                        "pearson"
                    ],
                    "mae_improvement": float(baseline_metrics["mae"])
                    - float(probe_metrics["mae"]),
                }
            )
    return rows


def _write_quality_probe_plot(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["scenario"]) for row in rows]
    indices = np.arange(len(rows))
    figure, axes = plt.subplots(
        2, 1, figsize=(max(10.0, len(rows) * 1.05), 8.5)
    )
    axes[0].plot(
        indices,
        [float(row["target_mean"]) for row in rows],
        marker="o",
        linewidth=2,
        label="Quality target",
    )
    axes[0].plot(
        indices,
        [float(row["prediction_mean"]) for row in rows],
        marker="s",
        linewidth=2,
        label="Raw-grid probe",
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Mean quality")
    axes[0].set_xticks(indices)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left")
    axes[0].set_title("Raw sensor-grid quality probe")

    width = 0.36
    axes[1].bar(
        indices - width / 2,
        [float(row["mae"]) for row in rows],
        width,
        label="Probe MAE",
    )
    axes[1].bar(
        indices + width / 2,
        [float(row["availability_baseline_mae"]) for row in rows],
        width,
        label="Availability-only MAE",
    )
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(indices, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_quality_probe_artifacts(
    model: nn.Module,
    history: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quality_probe_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "quality_probe_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "metadata": dict(metadata),
                "feature_names": list(QUALITY_FEATURE_NAMES),
                "training_history": list(history),
                "summary": list(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = output_dir / "quality_probe.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": list(QUALITY_FEATURE_NAMES),
            "metadata": dict(metadata),
        },
        checkpoint_path,
    )
    plot_path = output_dir / "quality_probe.png"
    _write_quality_probe_plot(plot_path, rows)
    return {
        "summary_csv": str(csv_path.resolve()),
        "summary_json": str(json_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "plot_png": str(plot_path.resolve()),
    }
