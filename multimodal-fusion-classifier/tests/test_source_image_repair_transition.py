import hashlib
import json
from pathlib import Path

from PIL import Image

from audit_source_image_repair_transition import (
    build_source_repair_transition,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_repair_transition_binds_historical_and_current_states(
    tmp_path,
):
    files = {
        name: tmp_path / relative
        for name, relative in {
            "recovery_audit": "recovery.json",
            "incident": "incident.json",
            "addendum": "addendum.json",
            "outcome": "outcome.json",
            "previous_integrity": "previous.json",
            "current_integrity": "current.json",
            "source": "source.jpg",
            "fallback": "fallback.jpg",
        }.items()
    }
    Image.new("RGB", (8, 8), (10, 20, 30)).save(files["source"])
    files["fallback"].write_bytes(files["source"].read_bytes())
    current_hash = _sha256(files["source"])
    old_hash = "0" * 64
    _write_json(
        files["recovery_audit"],
        {
            "valid": True,
            "source": {"sha256": old_hash},
            "fallback": {"sha256": current_hash},
        },
    )
    recovery_hash = _sha256(files["recovery_audit"])
    _write_json(
        files["previous_integrity"],
        {
            "valid": True,
            "summary": {
                "raw_inventory_sha256": "1" * 64,
                "effective_inventory_sha256": "2" * 64,
            },
        },
    )
    _write_json(
        files["current_integrity"],
        {
            "valid": True,
            "expected_recovery_state": "repaired",
            "summary": {
                "decode_failure_count": 0,
                "documented_recovery_count": 0,
                "raw_inventory_sha256": "2" * 64,
                "effective_inventory_sha256": "2" * 64,
            },
        },
    )
    _write_json(
        files["incident"],
        {
            "valid": True,
            "snapshot": {
                "recovery_mapping": {
                    "source": {"sha256": old_hash},
                    "fallback": {"sha256": current_hash},
                }
            },
            "source_bindings": {
                "recovery_audit": {"sha256": "historical-revision"}
            },
        },
    )
    _write_json(files["addendum"], {"valid": True})
    _write_json(
        files["outcome"],
        {
            "valid": True,
            "source_bindings": {
                "recovery_audit": {"sha256": recovery_hash},
                "source_integrity": {
                    "sha256": _sha256(files["previous_integrity"])
                },
            },
        },
    )

    report = build_source_repair_transition(files=files)

    assert report["valid"] is True
    assert report["transition"]["cause"] == "unknown_external_state_change"
    assert report["transition"]["source_now_equals_locked_fallback"] is True
    assert report["transition"]["current_raw_inventory_sha256"] == "2" * 64

    files["fallback"].write_bytes(b"changed")
    tampered = build_source_repair_transition(files=files)
    assert tampered["valid"] is False
    assert "current fallback hash drifted" in tampered["errors"]
