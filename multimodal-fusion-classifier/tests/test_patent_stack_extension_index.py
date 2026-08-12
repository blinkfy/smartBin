from __future__ import annotations

from build_patent_stack_extension_index import build_index


def test_patent_stack_index_preserves_development_boundary(tmp_path) -> None:
    index = build_index(tmp_path / "PATENT_STACK_EXTENSION_INDEX.json")

    assert index["valid"] is True
    assert index["formal_paper_metric"] is False
    assert index["v6_development_result"]["formal_performance_evidence"] is False
    assert index["model_selection"]["v6_promoted_for_task_performance"] is False
    assert index["governance"]["patent_document_in_extension_archive"] is False
    assert index["capability_status"]["full_image_qwen_sidecars"] == (
        "awaiting_api_credentials"
    )
