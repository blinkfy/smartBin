from __future__ import annotations

import pytest

from build_manuscript_result_governance_v11 import (
    validate_execution_state,
)
from build_manuscript_result_governance_v12 import validate_transition


def _governance():
    return {
        "valid": True,
        "version": 10,
        "decision": {"confirmed_option": "A"},
        "recent_baseline_extension": {"result_available": False},
    }


def _statuses():
    return {
        "component": {"state": "running", "stage": "seed42_training"},
        "unseen": {"state": "waiting", "stage": "component_training_queue"},
        "recent_baseline": {
            "state": "waiting",
            "stage": "unseen_sensor_condition_evaluation",
        },
        "manuscript_refresh": {
            "state": "waiting",
            "stage": "recent_robust_baseline_outcome",
        },
    }


def test_accepts_queued_option_a_manuscript_refresh():
    validate_execution_state(_governance(), _statuses())


def test_rejects_premature_or_failed_queue_state():
    statuses = _statuses()
    statuses["manuscript_refresh"]["state"] = "failed"

    with pytest.raises(ValueError, match="inadmissible"):
        validate_execution_state(_governance(), statuses)


def test_rejects_wrong_waiting_dependency():
    statuses = _statuses()
    statuses["manuscript_refresh"]["stage"] = "component_training_queue"

    with pytest.raises(ValueError, match="waiting-stage"):
        validate_execution_state(_governance(), statuses)


def test_v12_accepts_only_the_audited_reporting_transition():
    governance = {
        "valid": True,
        "version": 11,
        "decision": {"confirmed_option": "A"},
    }
    statuses = _statuses()

    validate_transition(
        governance,
        statuses,
        ["confirmatory_builder"],
    )
    with pytest.raises(ValueError, match="transition scope"):
        validate_transition(
            governance,
            statuses,
            ["confirmatory_builder", "manuscript_updater"],
        )
