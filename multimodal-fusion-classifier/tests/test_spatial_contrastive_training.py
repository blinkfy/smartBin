from __future__ import annotations

from train_spatial_contrastive_external_sensor_joint_seg import (
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_NAME,
    PROTOCOL_NAME,
    parse_args,
)


def test_spatial_training_defaults_to_v4_parent(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train"])

    args = parse_args()

    assert args.base_weights == DEFAULT_BASE_WEIGHTS
    assert args.name == DEFAULT_NAME
    assert PROTOCOL_NAME.endswith("_v5")
