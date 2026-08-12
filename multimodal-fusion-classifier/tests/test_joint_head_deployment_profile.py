from __future__ import annotations

from build_joint_head_deployment_profile import build_profile


def test_deployment_profile_keeps_latency_claim_descriptive(tmp_path) -> None:
    report = build_profile(
        tmp_path / "DEPLOYMENT_PROFILE.json",
        parameter_counts={
            "rgb": 10_089_641,
            "joint_seed42": 10_573_135,
            "joint_seed43": 10_573_135,
            "joint_seed44": 10_573_135,
        },
    )

    assert report["valid"] is True
    parameters = report["checkpoint_parameters_unfused"]
    assert parameters["joint_added"] == 483_494
    assert report["comparison_identity"]["formal_metric_forward_inputs"] == [
        "RGB"
    ]
    interpretation = report["interpretation"]
    assert interpretation["latency_is_amortized_validator_speed"] is True
    assert interpretation["single_sample_latency_claim_supported"] is False
    assert (
        interpretation["patent_15_2_ms_reproduction_claim_supported"] is False
    )
