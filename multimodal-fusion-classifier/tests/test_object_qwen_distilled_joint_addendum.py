from __future__ import annotations

import build_object_qwen_distilled_joint_addendum as addendum


def test_addendum_reports_negative_result_without_claim_inflation() -> None:
    report = addendum.build()
    assert report["valid"] is True
    assert report["formal_paper_metric"] is False
    assert (
        report["scene_grid_diagnostic"]["relative_mse_reduction"] > 0.30
    )
    assert report["joint_test"]["gate_ordering_all_passed"] is True
    assert (
        report["joint_test"][
            "matched_sensor_accuracy_contribution_detected"
        ]
        is False
    )
    interpretation = report["interpretation"]
    assert interpretation["supports_compatibility_claim"] is True
    assert interpretation["supports_accuracy_improvement_claim"] is False
    assert (
        interpretation["independent_segmentation_information_demonstrated"]
        is False
    )


def test_addendum_binds_the_selected_checkpoint() -> None:
    report = addendum.build()
    checkpoint = report["joint_training"]["checkpoint"]
    assert checkpoint["sha256"] == report["bindings"]["checkpoint"]["sha256"]
    assert checkpoint["size_bytes"] > 20_000_000
