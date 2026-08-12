from __future__ import annotations

from run_joint_seg_retrained_ablations import build_command


def test_retrained_ablation_command_binds_mode_and_seed() -> None:
    run = {
        "run_name": "example-seed43",
        "fusion_mode": "text_only",
        "seed": 43,
    }
    policy = {
        "epochs": 20,
        "patience": 5,
        "batch_size": 32,
        "image_size": 640,
        "workers": 0,
        "freeze_layers": 24,
        "amp": True,
        "learning_rate": 0.0001,
        "weight_decay": 0.0005,
        "semantic_dim": 64,
        "sensor_hidden_dim": 256,
        "text_temperature": 0.1,
        "sensor_reconstruction_weight": 0.25,
        "sensor_calibration_weight": 0.05,
        "cache": False,
    }

    command = build_command(run, policy)

    assert command[0]
    assert command[1] == "train_joint_multimodal_seg.py"
    assert command[command.index("--fusion-mode") + 1] == "text_only"
    assert command[command.index("--seed") + 1] == "43"
    assert command[command.index("--cache") + 1] == "false"
    assert "--no-amp" not in command
