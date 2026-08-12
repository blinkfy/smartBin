from __future__ import annotations

from train_contrastive_reliable_external_sensor_joint_seg import parse_args


def test_contrastive_training_defaults_are_development_only(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["train"])

    args = parse_args()

    assert args.mismatch_probability == 0.35
    assert args.epochs == 5
    assert args.workers == 4
    assert "contrastive" in args.name
