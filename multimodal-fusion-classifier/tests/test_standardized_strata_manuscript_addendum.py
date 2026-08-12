from __future__ import annotations

from build_standardized_strata_manuscript_addendum import build_addendum


def test_standardized_strata_addendum_preserves_claim_boundary(tmp_path) -> None:
    result = build_addendum(
        output=tmp_path / "MANUSCRIPT_ADDENDUM.json"
    )

    assert result["valid"] is True
    assert result["formal_paper_metric"] is False
    assert result["support"]["scale"]["small"] == 11
    assert result["support"]["overlap_proxy"]["overlap_high"] == 20
    assert result["claim_boundary"]["formal_blind_metric"] is False
    assert (
        result["claim_boundary"]["supports_original_patent_percentages"]
        is False
    )
    assert result["numerical_sensitivity_percentage_points"][
        "exact_batch_invariance"
    ] is False
