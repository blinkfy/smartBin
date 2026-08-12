from __future__ import annotations

import pytest

from summarize_seed_evaluations import DEFAULT_METRICS, summarize_runs


def _evaluation(offset: float, sample_count: int = 10) -> dict:
    return {
        "metrics": {
            "sample_count": sample_count,
            **{
                name: 0.5 + offset + index * 0.001
                for index, name in enumerate(DEFAULT_METRICS)
            },
        },
        "checkpoint": {"epoch": 90},
        "class_names": ["a", "b"],
    }


def test_seed_summary_uses_sample_standard_deviation():
    report = summarize_runs(
        [
            ("seed42", _evaluation(0.0)),
            ("seed43", _evaluation(0.1)),
            ("seed44", _evaluation(0.2)),
        ]
    )

    accuracy = report["aggregate"]["accuracy"]
    assert report["run_count"] == 3
    assert accuracy["mean"] == pytest.approx(0.6)
    assert accuracy["sample_std"] == pytest.approx(0.1)
    assert accuracy["min"] == pytest.approx(0.5)
    assert accuracy["max"] == pytest.approx(0.7)


def test_seed_summary_rejects_different_sample_counts():
    with pytest.raises(ValueError, match="sample count"):
        summarize_runs(
            [
                ("seed42", _evaluation(0.0, sample_count=10)),
                ("seed43", _evaluation(0.1, sample_count=9)),
            ]
        )


def test_seed_summary_rejects_missing_metric():
    second = _evaluation(0.1)
    del second["metrics"]["macro_f1"]
    with pytest.raises(ValueError, match="macro_f1"):
        summarize_runs(
            [
                ("seed42", _evaluation(0.0)),
                ("seed43", second),
            ]
        )
