import json

import numpy as np
import pytest

from mmfc.ensemble import (
    build_ensemble_result,
    load_prediction_table,
    select_validation_weights,
    selective_risk_profile,
    simplex_grid,
    validate_aligned_tables,
    validation_locked_selective_points,
    weighted_probability_pool,
)


def _write_predictions(path, probabilities, targets=(0, 1, 0, 1)) -> None:
    rows = []
    for index, (target, probability) in enumerate(zip(targets, probabilities)):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "source_image_id": f"source-{index // 2}",
                "target_id": target,
                "probabilities": probability,
            }
        )
    path.write_text(
        "\n".join(json.dumps(row) for row in reversed(rows)) + "\n",
        encoding="utf-8",
    )


def test_simplex_grid_has_expected_size_and_normalization() -> None:
    weights = list(simplex_grid(3, 4))

    assert len(weights) == 15
    assert all(np.isclose(vector.sum(), 1.0) for vector in weights)
    assert all(np.all(vector >= 0.0) for vector in weights)


def test_validation_selection_and_result_are_deterministic(tmp_path) -> None:
    strong_path = tmp_path / "strong.jsonl"
    weak_path = tmp_path / "weak.jsonl"
    _write_predictions(
        strong_path,
        ([0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]),
    )
    _write_predictions(
        weak_path,
        ([0.2, 0.8], [0.8, 0.2], [0.4, 0.6], [0.6, 0.4]),
    )
    strong = load_prediction_table(strong_path)
    weak = load_prediction_table(weak_path)

    selection = select_validation_weights(
        (strong, weak),
        ("class-a", "class-b"),
        denominator=4,
    )
    result, probabilities = build_ensemble_result(
        (strong, weak),
        ("strong", "weak"),
        selection["selected"]["weights"],
        ("class-a", "class-b"),
    )

    assert selection["test_used_for_selection"] is False
    assert selection["selected"]["weights"] == [1.0, 0.0]
    assert result["metrics"]["accuracy"] == pytest.approx(1.0)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(4))
    assert result["prediction_records"][0]["sample_id"] == "sample-0"
    assert set(result["prediction_records"][0]["member_outputs"]) == {
        "strong",
        "weak",
    }


def test_alignment_rejects_changed_targets(tmp_path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    probabilities = ([0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8])
    _write_predictions(first_path, probabilities)
    _write_predictions(second_path, probabilities, targets=(0, 0, 0, 1))

    with pytest.raises(ValueError, match="targets differ"):
        validate_aligned_tables(
            (
                load_prediction_table(first_path),
                load_prediction_table(second_path),
            )
        )


def test_weighted_pool_rejects_invalid_weights(tmp_path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    probabilities = ([0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8])
    _write_predictions(first_path, probabilities)
    _write_predictions(second_path, probabilities)
    tables = (
        load_prediction_table(first_path),
        load_prediction_table(second_path),
    )

    with pytest.raises(ValueError, match="non-negative"):
        weighted_probability_pool(tables, (1.0, -0.2))


def test_selective_metrics_keep_coverage_explicit() -> None:
    targets = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray(
        [[0.95, 0.05], [0.1, 0.9], [0.4, 0.6], [0.45, 0.55]]
    )

    profile = selective_risk_profile(
        targets,
        probabilities,
        coverage_points=(0.5, 1.0),
    )
    locked = validation_locked_selective_points(
        targets,
        probabilities,
        targets,
        probabilities,
        target_accuracies=(1.0,),
    )

    assert profile["points"][0]["accepted_samples"] == 2
    assert profile["points"][0]["selective_accuracy"] == pytest.approx(1.0)
    assert profile["points"][1]["selective_accuracy"] == pytest.approx(0.75)
    assert locked[0]["validation"]["accepted_samples"] == 2
    assert locked[0]["test"]["coverage"] == pytest.approx(0.5)
