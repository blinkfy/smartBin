from __future__ import annotations

import numpy as np
import pytest

from mmfc.uncertainty_temperature import (
    apply_variance_temperature,
    fit_global_variance_temperature,
    grouped_bootstrap_mean_difference,
    per_sample_gaussian_nll,
    summarize_uncertainty_scale,
)


def test_closed_form_temperature_normalizes_standardized_error() -> None:
    actual = np.asarray([[1.0, 2.0], [3.0, 5.0]])
    predicted = np.asarray([[0.0, 2.0], [2.0, 3.0]])
    log_variance = np.log(np.full_like(actual, 4.0))
    parameters = fit_global_variance_temperature(
        actual, predicted, log_variance
    )
    calibrated = apply_variance_temperature(
        log_variance, parameters["scale"]
    )
    summary = summarize_uncertainty_scale(actual, predicted, calibrated)
    assert parameters["scale"] == pytest.approx(0.375)
    assert (
        summary["mean_squared_standardized_residual"]
        == pytest.approx(1.0)
    )


def test_temperature_improves_nll_and_preserves_rank() -> None:
    actual = np.asarray([[0.0, 1.0], [2.0, 4.0], [1.0, 3.0]])
    predicted = actual + np.asarray(
        [[0.1, -0.2], [0.3, -0.1], [0.05, -0.4]]
    )
    log_variance = np.log(
        np.asarray([[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]])
    )
    parameters = fit_global_variance_temperature(
        actual, predicted, log_variance
    )
    calibrated = apply_variance_temperature(
        log_variance, parameters["scale"]
    )
    raw_summary = summarize_uncertainty_scale(
        actual, predicted, log_variance
    )
    calibrated_summary = summarize_uncertainty_scale(
        actual, predicted, calibrated
    )
    assert (
        calibrated_summary["gaussian_nll_mean"]
        < raw_summary["gaussian_nll_mean"]
    )
    assert (
        calibrated_summary["variance_error_correlation"]["elementwise"][
            "spearman"
        ]
        == raw_summary["variance_error_correlation"]["elementwise"][
            "spearman"
        ]
    )


def test_grouped_bootstrap_is_deterministic_and_paired() -> None:
    baseline = np.asarray([1.0, 1.2, 0.8, 1.1])
    candidate = baseline - np.asarray([0.2, 0.1, 0.3, 0.4])
    groups = ["a", "a", "b", "c"]
    first = grouped_bootstrap_mean_difference(
        baseline, candidate, groups, repeats=200, seed=7
    )
    second = grouped_bootstrap_mean_difference(
        baseline, candidate, groups, repeats=200, seed=7
    )
    assert first == second
    assert first["estimate"] == pytest.approx(-0.25)
    assert first["interval"][1] < 0.0


def test_invalid_temperature_inputs_are_rejected() -> None:
    values = np.ones((2, 2))
    with pytest.raises(ValueError):
        fit_global_variance_temperature(
            values, values[:, :1], values
        )
    with pytest.raises(ValueError):
        apply_variance_temperature(values, 0.0)
    with pytest.raises(ValueError):
        per_sample_gaussian_nll(
            values, values, np.asarray([[np.nan, 0.0], [0.0, 0.0]])
        )
