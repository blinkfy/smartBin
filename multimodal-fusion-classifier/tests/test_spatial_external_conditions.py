from __future__ import annotations

from evaluate_spatial_external_conditions import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    parse_args,
)


def test_spatial_condition_defaults_use_v5_run(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate"])

    args = parse_args()

    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.output == DEFAULT_OUTPUT
