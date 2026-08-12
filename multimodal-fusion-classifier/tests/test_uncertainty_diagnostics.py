from __future__ import annotations

import numpy as np
import pytest

from mmfc.uncertainty_diagnostics import (
    compute_uncertainty_diagnostics,
    grouped_bootstrap_uncertainty_intervals,
)


def test_uncertainty_diagnostics_reports_bounds_calibration_and_coverage():
    actual = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [0.2, 0.8, 2.4],
            [0.4, 1.2, 1.8],
            [0.6, 0.6, 2.8],
        ]
    )
    predicted = np.asarray(
        [
            [0.0, 0.9, 2.1],
            [0.1, 1.0, 2.0],
            [0.2, 0.9, 2.2],
            [0.3, 1.0, 2.1],
        ]
    )
    log_variance = np.asarray(
        [
            [-3.0, -2.8, -2.5],
            [-2.9, -2.5, -2.0],
            [-2.7, -2.1, -1.5],
            [-2.2, -1.7, -1.0],
        ]
    )

    result = compute_uncertainty_diagnostics(
        actual,
        predicted,
        log_variance,
        lower_bound=-3.0,
        upper_bound=2.0,
        bin_count=3,
    )

    assert result["sample_count"] == 4
    assert result["latent_dimension"] == 3
    assert result["element_count"] == 12
    assert result["bound_proximity"]["lower"]["within_1e-06"] == pytest.approx(
        1.0 / 12.0
    )
    assert len(result["variance_calibration"]["equal_count_bins"]) == 3
    assert 0.0 <= result["standardized_residual"][
        "coverage_absolute_le_1_96"
    ] <= 1.0
    assert np.isfinite(result["gaussian_nll"]["mean"])
    assert result["derived_interpretation_values"][
        "minimum_log_variance_distance_above_lower_bound"
    ] == pytest.approx(0.0)
    assert result["derived_interpretation_values"][
        "predicted_variance_to_mse_ratio"
    ] > 0.0


def test_uncertainty_diagnostics_recovers_monotonic_variance_error_ranking():
    predicted = np.zeros((4, 2), dtype=np.float64)
    errors = np.arange(1.0, 9.0, dtype=np.float64).reshape(4, 2) / 10.0
    actual = predicted + errors
    log_variance = np.log(np.square(errors))

    result = compute_uncertainty_diagnostics(
        actual,
        predicted,
        log_variance,
        lower_bound=-6.0,
        upper_bound=2.0,
        bin_count=2,
    )

    correlation = result["variance_error_correlation"]["elementwise"]
    assert correlation["pearson"] == pytest.approx(1.0)
    assert correlation["spearman"] == pytest.approx(1.0)
    assert result["variance_calibration"][
        "global_mse_to_variance_ratio"
    ] == pytest.approx(1.0)


def test_uncertainty_diagnostics_uses_null_for_constant_correlation():
    actual = np.ones((3, 2), dtype=np.float64)
    predicted = np.zeros((3, 2), dtype=np.float64)
    log_variance = np.zeros((3, 2), dtype=np.float64)

    result = compute_uncertainty_diagnostics(
        actual,
        predicted,
        log_variance,
        lower_bound=-3.0,
        upper_bound=2.0,
        bin_count=2,
    )

    assert result["variance_error_correlation"]["elementwise"] == {
        "pearson": None,
        "spearman": None,
    }


@pytest.mark.parametrize(
    ("actual", "predicted", "log_variance", "message"),
    [
        (
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            np.zeros((2, 2)),
            "share shape",
        ),
        (
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.full((2, 2), -4.0),
            "exceeds configured bounds",
        ),
    ],
)
def test_uncertainty_diagnostics_rejects_invalid_arrays(
    actual,
    predicted,
    log_variance,
    message,
):
    with pytest.raises(ValueError, match=message):
        compute_uncertainty_diagnostics(
            actual,
            predicted,
            log_variance,
            lower_bound=-3.0,
            upper_bound=2.0,
            bin_count=2,
        )


def test_grouped_uncertainty_bootstrap_is_reproducible_and_clustered():
    variance = np.asarray([0.10, 0.12, 0.20, 0.25, 0.40, 0.45])
    error = np.asarray([0.01, 0.02, 0.03, 0.05, 0.08, 0.10])
    groups = ["a", "a", "b", "c", "c", "d"]

    first = grouped_bootstrap_uncertainty_intervals(
        variance,
        error,
        groups,
        repeats=40,
        confidence=0.9,
        seed=7,
    )
    second = grouped_bootstrap_uncertainty_intervals(
        variance,
        error,
        groups,
        repeats=40,
        confidence=0.9,
        seed=7,
    )

    assert first == second
    assert first["sample_count"] == 6
    assert first["cluster_count"] == 4
    assert len(
        first["metrics"]["sample_mean_spearman"]["replicates"]
    ) == 40
    assert first["metrics"]["sample_mean_spearman"]["lower"] <= first[
        "metrics"
    ]["sample_mean_spearman"]["estimate"] <= first["metrics"][
        "sample_mean_spearman"
    ]["upper"]
