import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from build_v3_reliability_ablation_extension_archive import (
    EXCLUDED_SUFFIXES,
    _refuse_secrets,
)
from verify_v3_reliability_ablation_extension_archive import verify_archive


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_extension_secret_scan_rejects_api_key_like_tokens(tmp_path):
    source = tmp_path / "unsafe.json"
    fake_token = "sk-" + "abcdefghijklmnop123456"
    source.write_text(
        json.dumps({"token": fake_token}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="secret-like"):
        _refuse_secrets([source])


def test_extension_excludes_checkpoints_and_logs():
    assert ".pt" in EXCLUDED_SUFFIXES
    assert ".log" in EXCLUDED_SUFFIXES
    assert ".tmp" in EXCLUDED_SUFFIXES


def test_archive_verifier_accepts_consistent_deterministic_fixture(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    payload_name = "reports/qwen-material/fake.json"
    payload = b'{"valid": true}\n'
    evidence = {
        "schema_version": 1,
        "scope": "fixture",
        "file_count": 1,
        "files": [
            {
                "path": payload_name,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
        ],
    }
    evidence_bytes = (
        json.dumps(evidence, ensure_ascii=False, indent=2).encode("utf-8")
        + b"\n"
    )
    evidence_path = archive_dir / "evidence-manifest.json"
    evidence_path.write_bytes(evidence_bytes)
    archive_path = archive_dir / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence-manifest.json", evidence_bytes)
        archive.writestr(payload_name, payload)
    archive_manifest = {
        "archive": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "evidence_manifest_sha256": _sha256(evidence_bytes),
        "file_count": 1,
        "model_checkpoints_included": False,
        "raw_dataset_images_included": False,
        "original_patent_included": False,
        "training_logs_included": False,
    }
    (archive_dir / "archive-manifest.json").write_text(
        json.dumps(archive_manifest),
        encoding="utf-8",
    )
    report = verify_archive(archive_dir)
    assert report["valid"], report["errors"]
