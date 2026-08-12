from __future__ import annotations

from prepare_v3_reliability_ablation import (
    LOSS_REMOVALS,
    MODEL_REMOVALS,
    build_ablation_config,
    normalized_reliability_ablation_config,
)


def _base() -> dict:
    return {
        "model": {
            "feature_dim": 512,
            "sensor_reliability_enabled": True,
            "sensor_grid_quality_enabled": True,
            "sensor_cross_modal_quality_enabled": True,
        },
        "loss": {
            "ce_weight": 1.0,
            "reconstruction_weight": 0.1,
            "calibration_weight": 0.3,
            "variance_prior_weight": 0.1,
            "sensor_grid_reconstruction_weight": 0.5,
            "calibration_ranking_weight": 0.05,
        },
        "training": {
            "seed": 42,
            "epochs": 100,
            "num_workers": 0,
            "checkpoint_dir": "checkpoints/base",
        },
        "experiment": {
            "name": "full_cross_modal_quality",
            "group": "enhancement",
            "description": "v3",
        },
    }


def test_v3_reliability_ablation_changes_only_declared_subsystem():
    base = _base()

    value = build_ablation_config(
        base,
        seed=43,
        performance_evidence="performance.json",
    )

    assert all(value["model"][name] is False for name in MODEL_REMOVALS)
    assert all(value["loss"][name] == 0.0 for name in LOSS_REMOVALS)
    assert value["training"]["seed"] == 43
    assert value["training"]["num_workers"] == 4
    assert normalized_reliability_ablation_config(value) == (
        normalized_reliability_ablation_config(base)
    )


def test_v3_reliability_ablation_preserves_degradation_training():
    base = _base()
    base["robustness"] = {"enabled": True, "missing_probability": 0.25}

    value = build_ablation_config(
        base,
        seed=42,
        performance_evidence="performance.json",
    )

    assert value["robustness"] == base["robustness"]
