from __future__ import annotations

from train_patent_aligned_external_sensor_joint_seg import (
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_NAME,
    PROTOCOL_NAME,
    parse_args,
)


def test_patent_training_defaults_to_v5_parent(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train"])

    args = parse_args()

    assert args.base_weights == DEFAULT_BASE_WEIGHTS
    assert args.name == DEFAULT_NAME
    assert PROTOCOL_NAME.endswith("_v6")
    assert args.workers == 4
