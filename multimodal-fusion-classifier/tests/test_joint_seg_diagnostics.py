from __future__ import annotations

import numpy as np
import pytest

from evaluate_joint_multimodal_seg import sensor_prediction_metrics


def test_sensor_prediction_metrics_separate_covered_cells() -> None:
    target = np.zeros((2, 1, 4, 4), dtype=np.float32)
    target[:, :, 0, 0] = 1.0
    predicted = target.copy()
    predicted[:, :, 0, 0] = 0.75
    coverage = np.zeros_like(target)
    coverage[:, :, 0, 0] = 1.0
    uncertainty = np.full_like(target, 0.2)

    metrics = sensor_prediction_metrics(
        predicted, target, coverage, uncertainty
    )

    assert metrics["sample_count"] == 2
    assert metrics["cell_count"] == 32
    assert metrics["covered_cell_fraction"] == pytest.approx(1 / 16)
    assert metrics["covered_cell_mae"] == pytest.approx(0.25)
    assert metrics["uncovered_cell_mae"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.25 / 16)


def test_sensor_prediction_metrics_reject_shape_drift() -> None:
    values = np.zeros((2, 1, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="identical shape"):
        sensor_prediction_metrics(
            values,
            values[:1],
            values,
            values,
        )
