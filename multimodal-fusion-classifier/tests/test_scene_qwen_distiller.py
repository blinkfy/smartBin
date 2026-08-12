from __future__ import annotations

import numpy as np
import pytest
import torch

from mmfc.scene_distillation import (
    SceneQwenDistiller,
    correlation,
    distillation_loss,
    prediction_metrics,
)


def test_scene_distiller_contract_and_backward() -> None:
    model = SceneQwenDistiller()
    images = torch.randn(3, 3, 128, 128)
    teacher = torch.rand(3, 1, 4, 4)
    coverage = torch.rand(3, 1, 4, 4)

    prediction = model(images)
    loss, components = distillation_loss(
        prediction, teacher, coverage
    )
    loss.backward()

    assert prediction.shape == (3, 1, 4, 4)
    assert torch.all((0.0 <= prediction) & (prediction <= 1.0))
    assert set(components) == {"weighted_mse", "l1", "gradient"}
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_scene_distillation_metrics_compare_frozen_baselines() -> None:
    teacher = np.asarray(
        [[[[0.0, 0.2, 0.0, 0.0]] * 4]], dtype=np.float64
    )
    prediction = teacher.copy()
    coverage = (teacher > 0).astype(np.float64)
    metrics = prediction_metrics(
        prediction,
        teacher,
        coverage,
        train_mean_grid=np.zeros((4, 4)),
    )

    assert metrics["mse"] == 0.0
    assert metrics["positive_iou_at_0_05"] == 1.0
    assert metrics["zero_baseline_mse"] > metrics["mse"]


def test_rank_correlation_handles_ties() -> None:
    value = correlation(
        np.asarray([0.0, 0.0, 1.0, 2.0]),
        np.asarray([0.0, 0.0, 1.0, 2.0]),
        rank=True,
    )

    assert value == pytest.approx(1.0)
