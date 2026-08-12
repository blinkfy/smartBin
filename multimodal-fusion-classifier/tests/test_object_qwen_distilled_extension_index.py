from __future__ import annotations

import build_object_qwen_distilled_extension_index as extension


def test_extension_index_keeps_negative_result_and_open_gates() -> None:
    index = extension.build()
    assert index["valid"] is True
    assert index["formal_paper_metric"] is False
    readiness = index["readiness"]
    assert readiness["segmentation_accuracy_gain"] == "not_demonstrated"
    assert readiness["reliability_gate_discrimination"] == "demonstrated"
    assert (
        readiness["direct_full_image_qwen_three_seed_study"]
        == "preregistered_awaiting_credentials"
    )
    assert readiness["physical_sensor_validation"] == "not_implemented"
    assert index["claim_decision"]["patent_requirement_status_changed"] is False


def test_extension_index_binds_all_evidence() -> None:
    index = extension.build()
    assert len(index["evidence_bindings"]) == len(extension.EVIDENCE)
    assert all(
        len(binding["sha256"]) == 64
        for binding in index["evidence_bindings"].values()
    )
