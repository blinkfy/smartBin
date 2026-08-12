from __future__ import annotations

import torch

from mmfc.joint_seg_spatial import (
    SpatialReliabilityAwareSensorTextFeatureFusion,
)


def test_spatial_fusion_predicts_four_by_four_grid() -> None:
    fusion = SpatialReliabilityAwareSensorTextFeatureFusion(
        (8, 16, 32),
        torch.randn(3, 512),
        semantic_dim=8,
        sensor_hidden_dim=16,
        spatial_hidden_dim=16,
    )
    features = [
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]
    observed = torch.rand(2, 1, 4, 4)

    output = fusion(features, observed, torch.ones(2, 1))

    assert len(output) == 3
    assert fusion.last_aux["predicted_sensor_grid"].shape == (2, 1, 4, 4)
    assert fusion.last_aux["sensor_reliability"].shape == (2, 1)
