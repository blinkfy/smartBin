"""Verify the deterministic v3 reliability evidence-extension archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "submission-evidence"
    / "reliability-ablation-extension"
)
FORBIDDEN_SUFFIXES = {".pt", ".log", ".pid", ".tmp"}
SECRET_PATTERNS = (re.compile(rb"(?i)\bsk-[a-z0-9_-]{16,}"),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the v3 reliability evidence-extension archive"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(
    archive_dir: Path = DEFAULT_ARCHIVE_DIR,
) -> dict[str, Any]:
    archive_dir = _resolve(archive_dir)
    archive_manifest_path = archive_dir / "archive-manifest.json"
    evidence_manifest_path = archive_dir / "evidence-manifest.json"
    errors = []
    if not archive_manifest_path.is_file() or not evidence_manifest_path.is_file():
        return {
            "valid": False,
            "error_count": 1,
            "errors": ["reliability extension manifests are missing"],
        }
    archive_manifest = json.loads(
        archive_manifest_path.read_text(encoding="utf-8-sig")
    )
    evidence_manifest = json.loads(
        evidence_manifest_path.read_text(encoding="utf-8-sig")
    )
    archive_path = archive_dir / str(archive_manifest.get("archive", ""))
    if not archive_path.is_file():
        return {
            "valid": False,
            "error_count": 1,
            "errors": ["reliability extension ZIP is missing"],
        }
    if (
        archive_manifest.get("archive_size_bytes") != archive_path.stat().st_size
        or archive_manifest.get("archive_sha256") != _sha256_file(archive_path)
        or archive_manifest.get("evidence_manifest_sha256")
        != _sha256_file(evidence_manifest_path)
        or archive_manifest.get("file_count")
        != evidence_manifest.get("file_count")
    ):
        errors.append("external archive manifest mismatch")
    expected = {
        item["path"]: item for item in evidence_manifest.get("files", [])
    }
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if names != ["evidence-manifest.json", *sorted(expected)]:
            errors.append("ZIP entry order or set mismatch")
        try:
            embedded = json.loads(
                archive.read("evidence-manifest.json").decode("utf-8-sig")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            embedded = {}
            errors.append("embedded evidence manifest is unreadable")
        if embedded != evidence_manifest:
            errors.append("embedded and external evidence manifests differ")
        for name, entry in expected.items():
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                errors.append(f"unsafe ZIP entry: {name}")
                continue
            if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append(f"forbidden ZIP entry: {name}")
            try:
                content = archive.read(name)
            except KeyError:
                continue
            if len(content) != entry.get("size_bytes"):
                errors.append(f"size mismatch: {name}")
            if _sha256_bytes(content) != entry.get("sha256"):
                errors.append(f"hash mismatch: {name}")
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                errors.append(f"secret-like token in ZIP entry: {name}")
    for field in (
        "model_checkpoints_included",
        "raw_dataset_images_included",
        "original_patent_included",
        "training_logs_included",
    ):
        if archive_manifest.get(field) is not False:
            errors.append(f"exclusion flag mismatch: {field}")
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "archive": archive_path.as_posix(),
        "archive_sha256": _sha256_file(archive_path),
        "file_count": evidence_manifest.get("file_count"),
    }


def main() -> None:
    report = verify_archive(parse_args().archive_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("valid") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
