from __future__ import annotations

from build_joint_head_extension_index import build_index


def test_joint_head_extension_index_preserves_claim_boundaries(tmp_path) -> None:
    index = build_index(tmp_path / "JOINT_HEAD_EXTENSION_INDEX.json")

    assert index["state"] == "complete"
    assert index["base_evidence_preserved"] is True
    assert index["formal_run_counts"][
        "independently_retrained_branch_controls"
    ] == 6
    interpretation = index["audited_interpretation"]
    assert interpretation["both_single_branches_improve_over_rgb_mean"] is True
    assert interpretation["strong_synergy_claim_supported"] is False
    assert interpretation["statistical_significance_claim_supported"] is False
    assert index["claim_boundaries"]["formal_metric_forward_inputs"] == ["RGB"]
