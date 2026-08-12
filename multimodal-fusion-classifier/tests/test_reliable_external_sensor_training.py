from __future__ import annotations

import argparse

import pytest

from train_reliable_external_sensor_joint_seg import _degradation_config


def test_reliable_training_degradation_contract() -> None:
    args = argparse.Namespace(
        missing_probability=0.1,
        noise_probability=0.5,
        noise_std_low=0.02,
        noise_std_high=0.15,
        bias_probability=0.25,
        bias_strength=0.15,
        stuck_probability=0.15,
        quality_scale=0.25,
    )

    config = _degradation_config(args)

    assert config["enabled"] is True
    assert config["noise_std_range"] == [0.02, 0.15]
    assert config["missing_probability"] == 0.1


def test_reliable_training_rejects_invalid_probability() -> None:
    args = argparse.Namespace(
        missing_probability=1.1,
        noise_probability=0.5,
        noise_std_low=0.02,
        noise_std_high=0.15,
        bias_probability=0.25,
        bias_strength=0.15,
        stuck_probability=0.15,
        quality_scale=0.25,
    )

    with pytest.raises(ValueError, match="probabilities"):
        _degradation_config(args)
