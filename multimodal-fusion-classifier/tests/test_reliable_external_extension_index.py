from __future__ import annotations

from build_reliable_external_extension_index import build_index


def test_reliable_extension_index_keeps_development_boundary(
    tmp_path,
) -> None:
    index = build_index(
        tmp_path / "RELIABLE_EXTERNAL_EXTENSION_INDEX.json"
    )

    assert index["valid"] is True
    assert index["formal_paper_metric"] is False
    assert index["selected_development_result"][
        "formal_performance_evidence"
    ] is False
    assert index["governance"]["overcalibrated_candidate_promoted"] is False
    assert index["capability_status"]["full_image_qwen_sidecars"] == (
        "awaiting_api_credentials"
    )
