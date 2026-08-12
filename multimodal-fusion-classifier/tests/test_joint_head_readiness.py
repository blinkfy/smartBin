from __future__ import annotations

from build_joint_head_readiness import build_readiness


def test_joint_head_readiness_uses_independent_retraining(tmp_path) -> None:
    report = build_readiness(tmp_path / "PAPER_READINESS.json")

    assert report["valid"] is True
    assert (
        report["run_counts"]["independently_retrained_branch_controls"] == 6
    )
    evidence = report["evidence_status"]
    assert evidence["same_checkpoint_masking_used_as_formal_ablation"] is False
    assert evidence["formal_metric_forward_is_rgb_only"] is True
    assert evidence["stable_positive_text_synergy_over_sensor_only_mask"] is False
