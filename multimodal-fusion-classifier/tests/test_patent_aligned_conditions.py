from __future__ import annotations

from evaluate_patent_aligned_conditions import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    parse_args,
)


def test_patent_condition_defaults_use_v6_run(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate"])

    args = parse_args()

    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.output == DEFAULT_OUTPUT
