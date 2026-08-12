from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from mmfc.joint_seg import SensorTextFeatureFusion


def test_joint_fusion_is_shape_stable_and_identity_initialized() -> None:
    torch.manual_seed(7)
    features = [
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]
    prototypes = torch.randn(3, 512)
    fusion = SensorTextFeatureFusion(
        [8, 16, 32],
        prototypes,
        semantic_dim=8,
        sensor_hidden_dim=16,
    )

    outputs = fusion(features)

    assert [tuple(item.shape) for item in outputs] == [
        tuple(item.shape) for item in features
    ]
    for output, feature in zip(outputs, features):
        torch.testing.assert_close(output, feature)
    for module in (*fusion.text_output, *fusion.sensor_output):
        assert torch.count_nonzero(module.weight) == 0
    assert fusion.last_aux["predicted_sensor_grid"].shape == (2, 1, 4, 4)
    assert fusion.last_aux["sensor_uncertainty"].shape == (2, 1, 4, 4)
    assert fusion.last_aux["sensor_reliability"].shape == (2, 1)


def test_joint_fusion_accepts_future_external_sensor_without_changing_shape() -> None:
    torch.manual_seed(11)
    features = [
        torch.randn(1, 8, 8, 8),
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]
    fusion = SensorTextFeatureFusion(
        [8, 16, 32],
        torch.randn(4, 512),
        semantic_dim=8,
        sensor_hidden_dim=16,
    )
    fusion.sensor_scale.data.fill_(0.5)
    observed = torch.full((1, 1, 4, 4), 0.75)

    outputs = fusion(features, observed)

    assert [tuple(item.shape) for item in outputs] == [
        tuple(item.shape) for item in features
    ]
    torch.testing.assert_close(
        fusion.last_aux["effective_sensor_grid"], observed
    )
    assert fusion.last_aux["external_sensor_used"].item() == 1.0


@pytest.mark.parametrize(
    ("text_enabled", "sensor_enabled"),
    ((True, False), (False, True)),
)
def test_joint_fusion_supports_independent_training_modes(
    text_enabled: bool,
    sensor_enabled: bool,
) -> None:
    torch.manual_seed(13)
    features = [
        torch.randn(2, 8, 8, 8),
        torch.randn(2, 16, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]
    fusion = SensorTextFeatureFusion(
        [8, 16, 32],
        torch.randn(4, 512),
        semantic_dim=8,
        sensor_hidden_dim=16,
        text_enabled=text_enabled,
        sensor_enabled=sensor_enabled,
    )

    outputs = fusion(features)

    for output, feature in zip(outputs, features):
        torch.testing.assert_close(output, feature)
    assert bool(fusion.last_aux["text_enabled"][0, 0]) is text_enabled
    assert bool(fusion.last_aux["sensor_enabled"][0, 0]) is sensor_enabled
    if not sensor_enabled:
        assert torch.count_nonzero(
            fusion.last_aux["predicted_sensor_grid"]
        ) == 0
        with pytest.raises(ValueError, match="sensor branch is disabled"):
            fusion(features, torch.zeros(2, 1, 4, 4))


def test_joint_fusion_rejects_no_active_branch() -> None:
    with pytest.raises(ValueError, match="at least one fusion branch"):
        SensorTextFeatureFusion(
            [8, 16, 32],
            torch.randn(4, 512),
            text_enabled=False,
            sensor_enabled=False,
        )


def test_joint_fusion_defaults_legacy_checkpoint_to_full_mode() -> None:
    fusion = SensorTextFeatureFusion(
        [8, 16, 32],
        torch.randn(4, 512),
        semantic_dim=8,
        sensor_hidden_dim=16,
    )
    del fusion.text_enabled
    del fusion.sensor_enabled
    features = [
        torch.randn(1, 8, 8, 8),
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]

    fusion(features)

    assert fusion.last_aux["text_enabled"].item() == 1.0
    assert fusion.last_aux["sensor_enabled"].item() == 1.0
