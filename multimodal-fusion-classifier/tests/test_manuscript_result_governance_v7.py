from __future__ import annotations

import build_manuscript_result_governance_v7 as governance


def test_governance_keeps_motivation_unconfirmed() -> None:
    report = governance.build()
    assert report["valid"] is True
    scope = report["paper_scope"]
    assert scope["recommended_option"] == "A"
    assert scope["user_confirmed"] is False
    assert scope["draft_evidence_ready"] is True
    assert scope["submission_ready"] is False


def test_governance_excludes_unsupported_claims() -> None:
    report = governance.build()
    exclusions = {
        item["id"]: item for item in report["mandatory_exclusions"]
    }
    assert set(exclusions) == {"X1", "X2", "X3", "X4", "X5", "X6"}
    assert "98.8%" in exclusions["X1"]["claim"]
    assert "physical" in exclusions["X2"]["claim"]
    assert "RGB-D" in exclusions["X3"]["claim"]
    assert "Qwen" in exclusions["X4"]["claim"]


def test_governance_routes_development_results_to_supplement() -> None:
    report = governance.build()
    supplements = {
        item["id"]: item
        for item in report["supplementary_result_inclusion"]
    }
    assert supplements["S2"]["accuracy_gain_detected"] is False
    assert supplements["S3"]["test_nll_change_interval"][1] < 0.0
    assert supplements["S4"]["support"]["scale"]["small"] == 11
    assert supplements["S4"]["support"]["overlap_proxy"]["overlap_high"] == 20


def test_patent_snapshot_does_not_claim_completion() -> None:
    snapshot = governance.build()["patent_alignment_snapshot"]
    assert snapshot == {
        "implemented": 6,
        "partially_implemented": 9,
        "not_implemented": 3,
        "intentional_scope_change": 1,
        "unsupported_performance_claim": 1,
        "complete_patent_replication": False,
    }
