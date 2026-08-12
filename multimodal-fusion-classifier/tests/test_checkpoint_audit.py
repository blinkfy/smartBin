from __future__ import annotations

import json

import torch

from audit_checkpoints import audit_checkpoint_directory
from mmfc.training import _training_config_fingerprint


def _write_checkpoint_artifacts(
    tmp_path,
    *,
    best_epoch=2,
    selection_metric="accuracy",
    selection_mode="max",
):
    config = {
        "classes": [{"name": "a"}, {"name": "b"}],
        "training": {
            "epochs": 2,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
        },
    }
    fingerprint = _training_config_fingerprint(config)
    history = [
        {
            "epoch": 1,
            "val_accuracy": 0.6,
            "val_macro_f1": 0.5,
            "val_sensor_grid_reconstruction": 0.2,
        },
        {
            "epoch": 2,
            "val_accuracy": 0.7,
            "val_macro_f1": 0.6,
            "val_sensor_grid_reconstruction": 0.3,
        },
    ]
    common = {
        "class_names": ["a", "b"],
        "config_fingerprint": fingerprint,
        "best_val_accuracy": 0.7,
        "model_state": {"weight": torch.ones(2, 3)},
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
    }
    torch.save(
        {
            **common,
            "epoch": best_epoch,
            "val_accuracy": 0.7,
            "checkpoint_role": "best_evaluation",
            "history": history[:best_epoch],
        },
        tmp_path / "best.pt",
    )
    torch.save(
        {
            **common,
            "epoch": 2,
            "val_accuracy": 0.7,
            "checkpoint_role": "last_resume",
            "history": history,
            "optimizer_state": {},
            "scheduler_state": {},
            "data_loader_generator_state": torch.zeros(1, dtype=torch.uint8),
            "torch_rng_state": torch.zeros(1, dtype=torch.uint8),
            "cuda_rng_state_all": [],
        },
        tmp_path / "last.pt",
    )
    (tmp_path / "history.json").write_text(
        json.dumps(
            {
                "config_fingerprint": fingerprint,
                "history": history,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "training_summary.json").write_text(
        json.dumps(
            {
                "config_fingerprint": fingerprint,
                "completed_epochs": 2,
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
            }
        ),
        encoding="utf-8",
    )
    return config


def test_checkpoint_audit_accepts_consistent_best_and_last(tmp_path):
    config = _write_checkpoint_artifacts(tmp_path)

    report = audit_checkpoint_directory(config, tmp_path)

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["best"]["epoch"] == 2
    assert report["last"]["checkpoint_role"] == "last_resume"
    assert report["best"]["state"]["schema_sha256"] == report["last"]["state"][
        "schema_sha256"
    ]


def test_checkpoint_audit_rejects_wrong_best_epoch(tmp_path):
    config = _write_checkpoint_artifacts(tmp_path, best_epoch=1)

    report = audit_checkpoint_directory(config, tmp_path)

    assert report["valid"] is False
    assert any("argmax epoch" in error for error in report["errors"])


def test_checkpoint_audit_supports_minimized_selection_metric(tmp_path):
    config = _write_checkpoint_artifacts(
        tmp_path,
        best_epoch=1,
        selection_metric="sensor_grid_reconstruction",
        selection_mode="min",
    )

    report = audit_checkpoint_directory(config, tmp_path)

    assert report["valid"] is True
    assert report["expected"]["selection_mode"] == "min"
