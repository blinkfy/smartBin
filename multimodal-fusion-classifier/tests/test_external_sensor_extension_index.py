from __future__ import annotations

from build_external_sensor_extension_index import build_index


def test_external_extension_index_does_not_promote_rule_smoke(
    tmp_path,
) -> None:
    index = build_index(tmp_path / "EXTERNAL_SENSOR_EXTENSION_INDEX.json")

    assert index["valid"] is True
    assert index["scene_rule_data"]["annotation_inputs"] is False
    assert index["scene_rule_data"]["formal_performance_evidence"] is False
    assert index["smoke_result"]["formal_performance_evidence"] is False
    assert index["smoke_result"]["shuffled_self_assignment_count"] == 0
    assert (
        index["capability_status"]["shuffled_sidecar_negative_control"]
        == "implemented_and_smoke_tested"
    )
    assert index["capability_status"]["full_image_qwen_sidecars"] == (
        "awaiting_api_credentials"
    )
    assert index["claim_boundaries"]["qwen_is_hardware"] is False
