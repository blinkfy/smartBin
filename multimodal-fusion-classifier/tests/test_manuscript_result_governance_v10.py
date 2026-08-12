from __future__ import annotations

import pytest

from build_manuscript_result_governance_v10 import validate_sources


def _governance():
    return {
        "valid": True,
        "version": 9,
        "decision": {"confirmed_option": "A"},
    }


def _protocol():
    return {
        "valid": True,
        "state": "frozen_before_baseline_training",
        "methods": ["qmf", "tmc"],
        "training_seeds": [42, 43, 44],
        "run_count": 6,
    }


def test_validate_sources_accepts_frozen_option_a_baselines():
    validate_sources(_governance(), _protocol())


def test_validate_sources_rejects_missing_baseline_run():
    protocol = _protocol()
    protocol["run_count"] = 5

    with pytest.raises(ValueError, match="disagree"):
        validate_sources(_governance(), protocol)
