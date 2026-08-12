from __future__ import annotations

import pytest

from audit_joint_multimodal_seg import compare_metrics


def test_compare_joint_metrics_reports_percentage_points() -> None:
    keys = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
    )
    base = {key: 0.8 for key in keys}
    joint = {key: 0.8125 for key in keys}

    result = compare_metrics(joint, base)

    for value in result.values():
        assert value["joint_minus_base"] == pytest.approx(0.0125)
        assert value["joint_minus_base_percentage_points"] == pytest.approx(
            1.25
        )


def test_compare_joint_metrics_fails_closed_on_missing_metric() -> None:
    with pytest.raises(KeyError, match="missing comparison metric"):
        compare_metrics({}, {})
