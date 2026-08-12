from __future__ import annotations

import copy

import pytest

from audit_v3_loss_component_ablations import validate_protocol
from prepare_v3_loss_component_ablations import (
    SEEDS,
    VARIANTS,
    build_component_config,
    normalized_component_config,
)


def _base() -> dict:
    return {
        "classes": [{"name": "a", "description": "a"}],
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
        "robustness": {
            "sensor_degradation": {
                "enabled": True,
                "missing_probability": 0.1,
            }
        },
        "training": {
            "seed": 42,
            "batch_size": 16,
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


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("seed", list(SEEDS))
def test_component_config_changes_only_declared_loss_and_identity(
    variant: str,
    seed: int,
):
    base = _base()
    value = build_component_config(base, variant=variant, seed=seed)
    field = VARIANTS[variant]["loss_field"]

    assert value["loss"][field] == 0.0
    for other_variant, specification in VARIANTS.items():
        other_field = specification["loss_field"]
        if other_variant != variant:
            assert value["loss"][other_field] == base["loss"][other_field]
    assert value["training"]["seed"] == seed
    assert value["training"]["num_workers"] == 0
    assert value["training"]["batch_size"] == 16
    assert value["training"]["epochs"] == 100
    assert value["robustness"] == base["robustness"]
    assert normalized_component_config(
        value,
        loss_field=field,
    ) == normalized_component_config(base, loss_field=field)


def test_component_config_rejects_unknown_variant_and_seed():
    base = _base()

    with pytest.raises(ValueError, match="unsupported component"):
        build_component_config(base, variant="unknown", seed=42)
    with pytest.raises(ValueError, match="unsupported matched seed"):
        build_component_config(
            base,
            variant="v3_no_ranking",
            seed=41,
        )


def test_component_normalization_still_detects_unrelated_change():
    base = _base()
    value = build_component_config(
        base,
        variant="v3_no_ranking",
        seed=43,
    )
    changed = copy.deepcopy(value)
    changed["robustness"]["sensor_degradation"]["missing_probability"] = 0.2

    assert normalized_component_config(
        changed,
        loss_field="calibration_ranking_weight",
    ) != normalized_component_config(
        base,
        loss_field="calibration_ranking_weight",
    )


def test_protocol_validation_rejects_worker_drift_before_binding_check():
    protocol = {
        "state": "frozen_before_component_training",
        "valid": True,
        "scope": "matched_seed_v3_auxiliary_loss_component_ablation",
        "training": {
            "matched_seeds": list(SEEDS),
            "epochs": 100,
            "batch_size": 16,
            "num_workers": 4,
            "resume": False,
        },
        "interventions": {name: {} for name in VARIANTS},
        "source_bindings": {},
    }

    errors = validate_protocol(protocol)

    assert "training contract mismatch" in errors
    assert "source bindings missing" in errors
