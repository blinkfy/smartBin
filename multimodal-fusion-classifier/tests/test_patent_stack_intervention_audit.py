from __future__ import annotations

from audit_patent_stack_intervention import classify_contribution_pattern


def test_classifies_collective_effect_without_sidecar_effect() -> None:
    pattern = classify_contribution_pattern(
        {
            "metrics/mAP50-95(B)": 0.004,
            "metrics/mAP50-95(M)": 0.001,
        },
        sidecar_effect_max_abs=8e-6,
    )

    assert pattern == (
        "collective_stack_effect_without_measurable_sidecar_effect"
    )


def test_classifies_sidecar_effect_at_resolution() -> None:
    pattern = classify_contribution_pattern(
        {
            "metrics/mAP50-95(B)": 0.004,
            "metrics/mAP50-95(M)": 0.001,
        },
        sidecar_effect_max_abs=2e-4,
    )

    assert pattern == "sidecar_effect_measurable_at_resolution"
