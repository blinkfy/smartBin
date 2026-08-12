from __future__ import annotations

import pytest

from build_manuscript_result_governance_v13 import validate_transition
from build_manuscript_result_governance_v14 import (
    validate_visual_release_transition,
)


def _governance():
    return {
        "valid": True,
        "version": 12,
        "decision": {"confirmed_option": "A"},
    }


def _amendment():
    return {
        "valid": True,
        "state": "amended_after_user_scope_reduction",
        "method_training_seeds": {
            "qmf": [42, 43, 44],
            "tmc": [42],
        },
        "run_count": 4,
        "scope_reduction": {
            "single_run_reported_as_replication": False,
            "partial_run_used_for_results": False,
        },
    }


def _outcome():
    return {
        "valid": True,
        "state": "complete",
        "scope": (
            "matched_same_input_qmf_three_seed_tmc_single_seed_reference"
        ),
        "method_training_seeds": {
            "qmf": [42, 43, 44],
            "tmc": [42],
        },
        "run_count": 4,
        "summary": [
            {
                "method": "tmc",
                "seed_count": 1,
                "accuracy_sample_std": None,
                "macro_f1_sample_std": None,
            }
        ],
    }


def _governance_v13():
    return {
        "valid": True,
        "version": 13,
        "decision": {"confirmed_option": "A"},
        "baseline_reporting_contract": {
            "method_training_seeds": {
                "qmf": [42, 43, 44],
                "tmc": [42],
            }
        },
    }


def _confirmatory_outcome():
    return {"valid": True, "state": "complete"}


def _automated_audit():
    return {
        "valid": True,
        "state": "automated_build_complete_visual_qa_pending",
        "visual_qa_complete": False,
    }


def _visual_qa():
    return {
        "valid": True,
        "state": "complete",
        "visual_qa_complete": True,
        "issue_count": 0,
        "page_count": 13,
    }


def _manuscript_status():
    return {"state": "awaiting_visual_qa"}


def test_governance_v13_accepts_single_seed_tmc_without_sd():
    validate_transition(_governance(), _amendment(), _outcome())
    validate_visual_release_transition(
        _governance_v13(),
        _amendment(),
        _outcome(),
        _confirmatory_outcome(),
        _automated_audit(),
        _visual_qa(),
        _manuscript_status(),
    )


def test_governance_v13_rejects_fabricated_tmc_standard_deviation():
    outcome = _outcome()
    outcome["summary"][0]["accuracy_sample_std"] = 0.01

    with pytest.raises(ValueError, match="single-seed"):
        validate_transition(_governance(), _amendment(), outcome)
    with pytest.raises(ValueError, match="single-seed"):
        validate_visual_release_transition(
            _governance_v13(),
            _amendment(),
            outcome,
            _confirmatory_outcome(),
            _automated_audit(),
            _visual_qa(),
            _manuscript_status(),
        )
