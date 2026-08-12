from __future__ import annotations

import pytest

from compare_robustness_summaries import (
    EXPECTED_SEEDS,
    SCENARIOS,
    compare_summaries,
)
from summarize_matched_seed_comparisons import METRICS, summarize_comparisons


def _paired(offset: float) -> dict:
    return {
        "reference": {"name": "ablation", "metrics": {"accuracy": 0.8}},
        "candidate": {"name": "full", "metrics": {"accuracy": 0.8 + offset}},
        "mcnemar_exact": {"p_value": 0.1},
        "paired_bootstrap": {
            "repeats": 5000,
            "sample_count": 826,
            "cluster_count": 570,
            "differences": {
                name: {"point": offset + index * 0.001}
                for index, name in enumerate(METRICS)
            },
        },
    }


def _robustness(offset: float) -> dict:
    return {
        "seeds": EXPECTED_SEEDS,
        "seed_count": 5,
        "summary": [
            {
                "scenario": scenario,
                "seed_count": 5,
                "accuracy_mean": 0.8 + offset,
                "macro_f1_mean": 0.7 + offset,
                "ece_mean": 0.1 - offset,
            }
            for scenario in SCENARIOS
        ],
    }


def test_matched_seed_summary_uses_sample_standard_deviation():
    report = summarize_comparisons(
        [
            ("seed42", _paired(0.01)),
            ("seed43", _paired(0.02)),
            ("seed44", _paired(0.03)),
        ]
    )

    assert report["run_count"] == 3
    assert report["aggregate"]["accuracy"]["mean"] == pytest.approx(0.02)
    assert report["aggregate"]["accuracy"]["sample_std"] == pytest.approx(
        0.01
    )
    assert report["aggregate"]["accuracy"]["positive_seed_count"] == 3


def test_robustness_comparison_uses_full_minus_ablation_effect():
    report = compare_summaries(
        _robustness(0.02),
        _robustness(0.0),
    )

    assert report["scenario_count"] == len(SCENARIOS)
    assert report["rows"][0]["full_minus_ablation_accuracy_pp"] == (
        pytest.approx(2.0)
    )
    assert report["rows"][0]["ablation_minus_full_ece_pp"] == (
        pytest.approx(2.0)
    )
