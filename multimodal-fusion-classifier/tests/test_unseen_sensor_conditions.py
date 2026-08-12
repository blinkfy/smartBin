from __future__ import annotations

import torch

from evaluate_unseen_sensor_conditions import (
    CONDITIONS,
    apply_condition,
    build_derangement,
)


def _fixtures():
    sample_ids = ["a", "b", "c", "d"]
    clean = torch.stack(
        [
            torch.full((1, 4, 4), value, dtype=torch.float32)
            for value in (0.1, 0.3, 0.6, 0.9)
        ]
    )
    bank = {
        sample_id: clean[index].clone()
        for index, sample_id in enumerate(sample_ids)
    }
    derangement = build_derangement(sample_ids, seed=7)
    return sample_ids, clean, bank, derangement


def test_derangement_has_no_fixed_points_and_is_deterministic():
    values = ["a", "b", "c", "d", "e"]

    first = build_derangement(values, seed=123)
    second = build_derangement(values, seed=123)

    assert first == second
    assert set(first) == set(values)
    assert set(first.values()) == set(values)
    assert all(source != target for source, target in first.items())


def test_all_conditions_are_bounded_deterministic_and_aligned():
    sample_ids, clean, bank, derangement = _fixtures()

    for condition in CONDITIONS:
        first = apply_condition(
            condition,
            clean,
            sample_ids,
            sensor_bank=bank,
            derangement=derangement,
            seed=99,
        )
        second = apply_condition(
            condition,
            clean,
            sample_ids,
            sensor_bank=bank,
            derangement=derangement,
            seed=99,
        )
        observed, availability, quality = first
        assert torch.equal(observed, second[0])
        assert torch.equal(availability, second[1])
        assert torch.equal(quality, second[2])
        assert observed.shape == clean.shape
        assert availability.shape == (len(sample_ids), 1)
        assert quality.shape == (len(sample_ids), 1)
        assert torch.all((0.0 <= observed) & (observed <= 1.0))
        assert torch.all((0.0 <= availability) & (availability <= 1.0))
        assert torch.all((0.0 <= quality) & (quality <= 1.0))


def test_availability_errors_are_separated_from_observation_quality():
    sample_ids, clean, bank, derangement = _fixtures()

    false_negative = apply_condition(
        "clean_false_missing",
        clean,
        sample_ids,
        sensor_bank=bank,
        derangement=derangement,
    )
    false_positive = apply_condition(
        "missing_false_available",
        clean,
        sample_ids,
        sensor_bank=bank,
        derangement=derangement,
    )
    correct_missing = apply_condition(
        "missing_correct",
        clean,
        sample_ids,
        sensor_bank=bank,
        derangement=derangement,
    )

    assert torch.equal(false_negative[0], clean)
    assert torch.all(false_negative[1] == 0)
    assert torch.all(false_negative[2] == 1)
    assert torch.all(false_positive[0] == 0)
    assert torch.all(false_positive[1] == 1)
    assert torch.all(false_positive[2] == 0)
    assert torch.all(correct_missing[0] == 0)
    assert torch.all(correct_missing[1] == 0)
    assert torch.all(correct_missing[2] == 0)
