from __future__ import annotations

import build_latent_uncertainty_temperature_addendum as addendum


def test_temperature_addendum_separates_floor_and_scale_findings() -> None:
    report = addendum.build()
    assert report["valid"] is True
    assert report["formal_paper_metric"] is False
    assert (
        report["training_time_floor_status"][
            "test_fraction_within_0_05_of_lower_bound"
        ]
        == 0.0
    )
    decision = report["scientific_decision"]
    assert decision["lower_bound_collapse_present"] is False
    assert decision["raw_absolute_variance_calibrated"] is False
    assert decision["posthoc_absolute_variance_calibrated"] is True
    assert decision["physical_sensor_uncertainty_calibrated"] is False


def test_temperature_addendum_reports_test_improvement() -> None:
    report = addendum.build()
    metrics = report["test_metrics"]
    assert metrics["raw_variance_to_mse_ratio"] > 40.0
    assert 0.9 < metrics["calibrated_variance_to_mse_ratio"] < 1.1
    assert metrics["calibrated_gaussian_nll"] < metrics["raw_gaussian_nll"]
    assert metrics["nll_cluster_bootstrap_interval"][1] < 0.0
    assert (
        abs(metrics["calibrated_coverage_absolute_le_1_96"] - 0.95)
        < 0.001
    )
