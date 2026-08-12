from __future__ import annotations

import sys
from pathlib import Path

from prepare_modality_baselines import build_config
from run_modality_baseline_queue import build_train_command


def test_modality_baseline_configs_use_uniform_three_seed_protocol() -> None:
    sensor = build_config("sensor_only", 43)
    assert sensor["model"]["enabled_modalities"] == ["sensor"]
    assert sensor["model"]["fusion_mode"] == "mean"
    assert sensor["model"]["sensor_reliability_enabled"] is False
    assert sensor["loss"]["sensor_grid_reconstruction_weight"] == 0.0
    assert sensor["robustness"]["sensor_degradation"]["enabled"] is False
    assert sensor["training"]["epochs"] == 100
    assert sensor["training"]["seed"] == 43
    assert sensor["training"]["checkpoint_dir"].endswith(
        "modality-baselines/sensor_only/seed43"
    )


def test_seed42_visual_baseline_reuses_verified_formal_checkpoint() -> None:
    config = build_config("visual_sensor", 42)
    assert config["training"]["checkpoint_dir"] == (
        "checkpoints/ablations/vision_sensor"
    )


def test_training_command_resumes_only_when_last_checkpoint_exists(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    checkpoint_dir = tmp_path / "checkpoint"
    assert build_train_command(config, checkpoint_dir) == [
        sys.executable,
        "-u",
        "train.py",
        "--config",
        str(config),
    ]
    checkpoint_dir.mkdir()
    (checkpoint_dir / "last.pt").write_bytes(b"checkpoint")
    assert build_train_command(config, checkpoint_dir)[-1] == "--resume"
