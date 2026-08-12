from __future__ import annotations

from build_paper_readiness_v6_addendum import build_addendum


def test_v6_readiness_addendum_is_additive_and_bounded(tmp_path) -> None:
    result = build_addendum(
        output=tmp_path / "PAPER_READINESS_V6_ADDENDUM.json"
    )

    assert result["valid"] is True
    assert result["historical_audits_modified"] is False
    assert result["object_checkpoint_recovery"]["found"] is True
    assert result["object_checkpoint_recovery"]["selected_epoch"] == 97
    assert result["object_checkpoint_recovery"][
        "test_split_external_blind"
    ] is False
    assert result["readiness"]["joint_native_segment_head"] == (
        "development_mechanism_evidence"
    )
    assert result["readiness"]["full_patent_reproduction"] is False
    assert result["readiness"]["full_paper_motivation_confirmed"] is False
    assert (
        result["uncertainty_status"][
            "physical_sensor_calibration_established"
        ]
        is False
    )
