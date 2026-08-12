from __future__ import annotations

import torch

from mmfc.joint_seg_reliable import (
    ReliabilityAwareSensorTextFeatureFusion,
    pack_reliable_sensor_channels,
    split_reliable_sensor_channels,
)


def test_reliable_transport_preserves_grid_and_availability() -> None:
    images = torch.zeros(2, 3, 16, 16, dtype=torch.uint8)
    grids = torch.linspace(0.0, 1.0, 32).reshape(2, 1, 4, 4)
    availability = torch.tensor([[1.0], [0.0]])

    packed = pack_reliable_sensor_channels(images, grids, availability)
    rgb, restored, restored_availability = split_reliable_sensor_channels(
        packed.float() / 255.0
    )

    assert packed.shape == (2, 5, 16, 16)
    assert rgb.shape == (2, 3, 16, 16)
    assert torch.allclose(restored, (grids * 255).round() / 255)
    assert torch.equal(restored_availability, availability)


def test_reliable_fusion_uses_predicted_grid_when_sensor_missing() -> None:
    fusion = ReliabilityAwareSensorTextFeatureFusion(
        (8, 16, 32),
        torch.randn(3, 512),
        semantic_dim=4,
        sensor_hidden_dim=16,
    )
    fusion.eval()
    features = [
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 16, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]
    observed = torch.rand(2, 1, 4, 4)
    availability = torch.zeros(2, 1)

    output = fusion(features, observed, availability)
    auxiliary = fusion.last_aux

    assert len(output) == 3
    assert torch.equal(
        auxiliary["effective_sensor_grid"],
        auxiliary["predicted_sensor_grid"],
    )
    assert torch.equal(
        auxiliary["sensor_reliability"],
        torch.zeros(2, 1),
    )
    assert auxiliary["sensor_cross_modal_quality_features"].shape == (2, 22)
