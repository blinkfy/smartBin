import json
from pathlib import Path

from capture_image_recovery_incident import build_incident_snapshot


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_incident_snapshot_binds_failed_status_and_partial_bundle(tmp_path):
    files = {
        "formal_status": tmp_path / "status.json",
        "history": tmp_path / "history.json",
        "best_checkpoint": tmp_path / "best.pt",
        "last_checkpoint": tmp_path / "last.pt",
        "config": tmp_path / "config.json",
        "train_manifest": tmp_path / "train.jsonl",
        "formal_run_manifest": tmp_path / "formal.json",
        "recovery_audit": tmp_path / "recovery.json",
        "runtime_shim": tmp_path / "sitecustomize.py",
        "recovery_chain": tmp_path / "recovery.ps1",
    }
    _write_json(
        files["formal_status"],
        {
            "state": "failed",
            "stage": "pipeline",
            "message": "full_no_degradation_training failed with exit code 1",
        },
    )
    _write_json(
        files["history"],
        {
            "experiment": {"name": "full_no_degradation"},
            "config_fingerprint": "fingerprint",
            "history": [{"epoch": index} for index in range(1, 6)],
        },
    )
    _write_json(
        files["recovery_audit"],
        {
            "valid": True,
            "source": {"sha256": "source"},
            "fallback": {"sha256": "fallback"},
            "comparison": {"bytes": {"differing_byte_count": 16}},
            "manifest_occurrences": [],
        },
    )
    for name, path in files.items():
        if not path.exists():
            path.write_text(name, encoding="utf-8")

    first = build_incident_snapshot(files=files)
    second = build_incident_snapshot(files=files)

    assert first["valid"] is True
    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    assert first["snapshot"]["partial_training"]["completed_epochs"] == 5
    assert first["recovery_invariants"]["original_image_modified"] is False
    assert set(first["source_bindings"]) == set(files)
