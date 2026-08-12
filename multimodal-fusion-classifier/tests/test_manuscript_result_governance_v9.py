from __future__ import annotations

import pytest

from build_manuscript_result_governance_v9 import _validate_source_state


def _governance():
    return {
        "valid": True,
        "version": 8,
        "decision": {"confirmed_option": "A"},
    }


def _protocol():
    return {
        "valid": True,
        "state": "frozen_before_unseen_condition_evaluation",
        "conditions": {f"condition_{index}": {} for index in range(11)},
        "model_seeds": [42, 43, 44],
    }


def test_validate_source_state_accepts_frozen_option_a_extension():
    _validate_source_state(_governance(), _protocol())


def test_validate_source_state_rejects_changed_condition_count():
    protocol = _protocol()
    protocol["conditions"].pop("condition_10")

    with pytest.raises(ValueError, match="disagree"):
        _validate_source_state(_governance(), protocol)
