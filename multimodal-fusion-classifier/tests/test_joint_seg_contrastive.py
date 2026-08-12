from __future__ import annotations

import pytest

from mmfc.joint_seg_contrastive import build_batch_mismatch_plan


def test_mismatch_plan_is_deterministic_and_never_uses_self() -> None:
    source_ids = [f"image-{index}" for index in range(16)]

    first = build_batch_mismatch_plan(
        source_ids, probability=1.0, seed=42
    )
    second = build_batch_mismatch_plan(
        source_ids, probability=1.0, seed=42
    )

    assert first == second
    assert all(donor is not None for donor in first)
    assert all(donor != index for index, donor in enumerate(first))


def test_mismatch_plan_respects_boundary_probabilities() -> None:
    source_ids = ["a", "b", "c"]

    assert build_batch_mismatch_plan(
        source_ids, probability=0.0, seed=42
    ) == [None, None, None]
    assert build_batch_mismatch_plan(
        ["only"], probability=1.0, seed=42
    ) == [None]

    with pytest.raises(ValueError, match="probability"):
        build_batch_mismatch_plan(
            source_ids, probability=1.1, seed=42
        )
