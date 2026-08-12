from __future__ import annotations

from evaluate_patent_stack_intervention import (
    changed_state_keys,
    expected_modulator_state_keys,
)
from prepare_patent_stack_intervention import (
    PROTOCOL_NAME,
    parse_args,
)


def test_intervention_defaults_preserve_worker_acceleration(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["prepare"])

    args = parse_args()

    assert args.workers == 4
    assert args.batch_size == 64
    assert args.image_size == 640
    assert PROTOCOL_NAME.endswith("_v1")


def test_expected_intervention_changes_only_six_modulator_tensors() -> None:
    expected = expected_modulator_state_keys()
    before = {"other": "same", **{name: "before" for name in expected}}
    after = {"other": "same", **{name: "after" for name in expected}}

    assert len(expected) == 6
    assert changed_state_keys(before, after) == expected
