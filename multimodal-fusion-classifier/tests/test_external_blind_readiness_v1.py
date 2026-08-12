from __future__ import annotations

import json
from pathlib import Path

from build_external_blind_readiness_v1 import build_readiness


def test_external_blind_readiness_detects_model_upgrade_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "EXTERNAL_BLIND_READINESS_V1.json"
    report = build_readiness(
        output=output,
        bundle=tmp_path / "no-external-bundle",
        evaluation_dir=tmp_path / "no-evaluation",
    )
    assert report["valid"] is False
    assert report["tooling_ready"] is False
    assert report["bound_sources_current"] is False
    assert set(report["drifted_bindings"]) == {
        "dataset_core",
        "evaluation_core",
        "model_core",
    }
    assert report["confirmed_manuscript_option"] == "A"
    assert report["hardware_release_interface_ready"] is True
    assert report["physical_hardware_record_count"] == 0
    assert report["physical_hardware_validation_ready"] is False
    assert report["external_bundle_available"] is False
    assert report["formal_external_result_available"] is False
    assert report["gate_state"] == "awaiting_external_bundle"
    assert report["submission_ready"] is False
    assert all(
        action["owner"] != "user" for action in report["next_actions"]
    )
    assert report["registered_minimums"]["total_samples"] == 380
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["protocol_sha256"] == report["protocol_sha256"]
