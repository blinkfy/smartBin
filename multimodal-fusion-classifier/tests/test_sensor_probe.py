from __future__ import annotations

import numpy as np
import pytest

from evaluate_sensor_probe import (
    fit_linear_sensor_probes,
    per_sample_standardize_grid,
)


def test_per_sample_standardize_grid_preserves_shape_and_normalizes_rows():
    values = np.stack(
        (
            np.arange(16, dtype=np.float64),
            np.arange(16, dtype=np.float64) * 2.0 + 5.0,
        )
    )

    standardized = per_sample_standardize_grid(values)

    assert standardized.shape == (2, 16)
    np.testing.assert_allclose(standardized.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=1), 1.0, atol=1e-12)


def test_fixed_linear_sensor_probes_return_all_feature_sets():
    generator = np.random.default_rng(7)
    train_labels = np.repeat(np.arange(2), 20)
    test_labels = np.repeat(np.arange(2), 6)
    train_grid = generator.normal(size=(40, 16)) + train_labels[:, None]
    test_grid = generator.normal(size=(12, 16)) + test_labels[:, None]
    train_properties = generator.normal(size=(40, 5)) + train_labels[:, None]
    test_properties = generator.normal(size=(12, 5)) + test_labels[:, None]

    result = fit_linear_sensor_probes(
        {
            "labels": train_labels,
            "grid": train_grid,
            "properties": train_properties,
        },
        {
            "labels": test_labels,
            "grid": test_grid,
            "properties": test_properties,
        },
        ("a", "b"),
        max_iter=200,
    )

    assert result["train_sample_count"] == 40
    assert result["evaluation_sample_count"] == 12
    assert set(result["probes"]) == {
        "normalized_grid_raw",
        "normalized_grid_per_sample_zscore",
        "five_properties",
        "grid_plus_properties",
        "train_class_prior",
    }
    for probe in result["probes"].values():
        assert 0.0 <= probe["metrics"]["accuracy"] <= 1.0
        assert 0.0 <= probe["metrics"]["macro_f1"] <= 1.0
