from __future__ import annotations

import pytest

from plot_class_performance import build_class_rows


def _evaluation() -> dict:
    return {
        "per_class": [
            {"label": "a", "f1": 0.8},
            {"label": "b", "f1": 0.6},
        ]
    }


def test_class_performance_rows_are_sorted_by_training_count():
    rows = build_class_rows(
        ["a", "b"],
        [{"label": "a"}, {"label": "a"}, {"label": "b"}],
        [("model", _evaluation())],
    )

    assert [row["label"] for row in rows] == ["a", "b"]
    assert rows[0]["train_count"] == 2
    assert rows[1]["f1"]["model"] == pytest.approx(0.6)


def test_class_performance_rejects_evaluation_label_mismatch():
    with pytest.raises(ValueError, match="class labels"):
        build_class_rows(
            ["a", "b"],
            [{"label": "a"}, {"label": "b"}],
            [
                (
                    "bad",
                    {"per_class": [{"label": "a", "f1": 0.8}]},
                )
            ],
        )
