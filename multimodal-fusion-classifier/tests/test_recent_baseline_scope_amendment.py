from __future__ import annotations

import pytest

from audit_recent_robust_baselines import AMENDED_METHOD_TRAINING_SEEDS
from build_recent_robust_baseline_scope_amendment import _validate_original
from evaluate_unseen_sensor_conditions import CONDITIONS
from prepare_recent_robust_baselines import METHODS, SEEDS


def _original_protocol():
    return {
        "valid": True,
        "state": "frozen_before_baseline_training",
        "methods": list(METHODS),
        "training_seeds": list(SEEDS),
        "run_count": 6,
        "conditions": CONDITIONS,
        "primary_paired_conditions": [
            "clean",
            "noise_030_out_of_range",
            "spatial_permutation",
            "foreign_sample",
            "missing_false_available",
        ],
        "bootstrap_repeats": 2000,
    }


def test_scope_amendment_keeps_tmc_single_seed_explicit():
    assert AMENDED_METHOD_TRAINING_SEEDS == {
        "qmf": (42, 43, 44),
        "tmc": (42,),
    }
    _validate_original(_original_protocol())


def test_scope_amendment_rejects_rewritten_original_plan():
    protocol = _original_protocol()
    protocol["run_count"] = 4

    with pytest.raises(ValueError, match="original"):
        _validate_original(protocol)
