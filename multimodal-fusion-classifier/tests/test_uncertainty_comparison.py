from __future__ import annotations

import copy

import pytest

from compare_uncertainty_diagnostics import build_uncertainty_comparison


def _report(*, ratio: float, sample_spearman: float):
    return {
        "schema_version": 1,
        "valid": True,
        "diagnostic_scope": "sensor_encoder_latent_space",
        "dataset": {
            "split": "test",
            "max_samples": None,
        },
        "source_bindings": {
            "manifest": {"sha256": "manifest", "size_bytes": 10},
            "sensor_manifest": {"sha256": "sensor", "size_bytes": 20},
        },
        "bootstrap": {
            "method": "source_image_cluster_percentile_bootstrap",
            "repeats": 3,
            "confidence": 0.95,
            "seed": 7,
            "sample_count": 2,
            "cluster_count": 2,
            "source_group_assignment_sha256": "groups",
            "metrics": {
                "sample_mean_spearman": {
                    "estimate": sample_spearman,
                    "replicates": [
                        sample_spearman - 0.02,
                        sample_spearman,
                        sample_spearman + 0.02,
                    ],
                },
                "predicted_variance_to_mse_ratio": {
                    "estimate": ratio,
                    "replicates": [ratio - 1.0, ratio, ratio + 1.0],
                },
            },
        },
        "diagnostics": {
            "sample_count": 2,
            "latent_dimension": 3,
            "element_count": 6,
            "derived_interpretation_values": {
                "predicted_variance_to_mse_ratio": ratio,
                "minimum_log_variance_distance_above_lower_bound": 0.7,
            },
            "variance_error_correlation": {
                "elementwise": {"spearman": sample_spearman - 0.1},
                "sample_mean": {"spearman": sample_spearman},
                "latent_dimension_mean": {
                    "spearman": sample_spearman + 0.05
                },
            },
            "standardized_residual": {
                "standard_deviation": 0.2,
                "coverage_absolute_le_1": 0.99,
                "coverage_absolute_le_1_96": 1.0,
            },
            "bound_proximity": {
                "lower": {"within_0.1": 0.0},
            },
        },
    }


def test_uncertainty_comparison_reports_ranking_and_scale_improvement():
    result = build_uncertainty_comparison(
        _report(ratio=50.0, sample_spearman=0.80),
        _report(ratio=40.0, sample_spearman=0.88),
    )

    assert result["derived"][
        "predicted_variance_to_mse_ratio_reduction"
    ] == pytest.approx(10.0)
    assert result["derived"][
        "predicted_variance_to_mse_ratio_relative_reduction"
    ] == pytest.approx(0.2)
    assert result["derived"]["sample_mean_spearman_increase"] == pytest.approx(
        0.08
    )
    assert result["paired_bootstrap"]["metrics"][
        "sample_mean_spearman_increase"
    ]["lower"] == pytest.approx(0.08)
    assert result["paired_bootstrap"]["metrics"][
        "predicted_variance_to_mse_ratio_reduction"
    ]["lower"] == pytest.approx(10.0)


def test_uncertainty_comparison_rejects_data_source_mismatch():
    v2b = _report(ratio=50.0, sample_spearman=0.80)
    v3 = copy.deepcopy(v2b)
    v3["source_bindings"]["manifest"]["sha256"] = "changed"

    with pytest.raises(ValueError, match="data source mismatch"):
        build_uncertainty_comparison(v2b, v3)
