"""Verify a deterministic paper evidence archive without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
FORBIDDEN_SUFFIXES = {".pt", ".jsonl", ".log", ".pid", ".tmp"}
ORIGINAL_PATENT_NAME = (
    "2026.4.3-1基于多模态识别的医疗废品识别方法.docx"
)
ALLOWED_CHECKPOINT_METADATA = {"history.json", "training_summary.json"}
ALLOWED_CHECKPOINT_PREFIXES = (
    ("project", "checkpoints", "ablations"),
    (
        "project",
        "checkpoints",
        "confirmatory-ablations",
    ),
    ("project", "checkpoints", "recent-robust-baselines"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify paper evidence ZIP hashes and file allowlist"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "submission-evidence"
        ),
    )
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        name == path.as_posix()
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and ":" not in name
        and not name.endswith("/")
        and bool(path.parts)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _allowed_checkpoint_metadata(path: PurePosixPath) -> bool:
    return (
        path.name in ALLOWED_CHECKPOINT_METADATA
        and any(
            path.parts[: len(prefix)] == prefix
            for prefix in ALLOWED_CHECKPOINT_PREFIXES
        )
    )


def _forbidden_member_reason(name: str) -> str | None:
    path = PurePosixPath(name)
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return "forbidden suffix"
    if name.startswith("project/data/"):
        return "raw data or annotation path"
    if name.startswith("project/checkpoints/"):
        if _allowed_checkpoint_metadata(path):
            return None
        return "checkpoint path"
    if name == f"project/{ORIGINAL_PATENT_NAME}":
        return "original patent source"
    return None


def verify_evidence_archive(archive_dir: Path) -> dict[str, Any]:
    archive_dir = archive_dir.resolve()
    archive_manifest_path = archive_dir / "archive-manifest.json"
    evidence_manifest_path = archive_dir / "evidence-manifest.json"
    archive_path = archive_dir / "paper-evidence.zip"
    errors: list[str] = []
    for path in (
        archive_manifest_path,
        evidence_manifest_path,
        archive_path,
    ):
        if not path.is_file():
            errors.append(f"required archive artifact is missing: {path}")
    if errors:
        return {
            "valid": False,
            "error_count": len(errors),
            "errors": errors,
        }

    try:
        archive_manifest = json.loads(
            archive_manifest_path.read_text(encoding="utf-8-sig")
        )
        evidence_manifest_bytes = evidence_manifest_path.read_bytes()
        evidence_manifest = json.loads(
            evidence_manifest_bytes.decode("utf-8-sig")
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        errors.append(f"cannot parse archive manifests: {exc}")
        return {
            "valid": False,
            "error_count": len(errors),
            "errors": errors,
        }
    if not isinstance(archive_manifest, dict):
        errors.append("archive manifest root must be an object")
        archive_manifest = {}
    if not isinstance(evidence_manifest, dict):
        errors.append("evidence manifest root must be an object")
        evidence_manifest = {}
    if archive_manifest.get("archive") != archive_path.name:
        errors.append("archive manifest filename does not match")
    if archive_manifest.get("archive_sha256") != _sha256_file(archive_path):
        errors.append("archive SHA-256 does not match")
    if archive_manifest.get(
        "archive_size_bytes"
    ) != archive_path.stat().st_size:
        errors.append("archive byte size does not match")
    if archive_manifest.get(
        "evidence_manifest_sha256"
    ) != _sha256_bytes(evidence_manifest_bytes):
        errors.append("external evidence manifest SHA-256 does not match")
    if archive_manifest.get("file_count") != evidence_manifest.get(
        "file_count"
    ):
        errors.append("archive and evidence manifest file counts differ")
    for key in (
        "original_patent_included",
        "raw_data_included",
        "annotations_included",
        "checkpoints_included",
        "model_checkpoints_included",
        "prediction_jsonl_included",
    ):
        if archive_manifest.get(key) is not False:
            errors.append(f"archive exclusion flag is not false: {key}")

    manifest_files = evidence_manifest.get("files")
    if not isinstance(manifest_files, list):
        errors.append("evidence manifest files must be an array")
        manifest_files = []
    expected_members: dict[str, dict[str, Any]] = {}
    total_uncompressed_bytes = 0
    for index, entry in enumerate(manifest_files):
        label = f"evidence manifest file {index}"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not _safe_member_name(
            f"project/{relative}"
        ):
            errors.append(f"{label} has an unsafe path")
            continue
        name = f"project/{relative}"
        if name in expected_members:
            errors.append(f"evidence manifest contains duplicate path: {name}")
            continue
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"{label} has an invalid byte size")
            continue
        if not _is_sha256(entry.get("sha256")):
            errors.append(f"{label} has an invalid SHA-256")
            continue
        total_uncompressed_bytes += size
        expected_members[name] = entry
    if evidence_manifest.get("file_count") != len(manifest_files):
        errors.append("evidence manifest file count does not match its files")
    if (
        evidence_manifest.get("total_uncompressed_bytes")
        != total_uncompressed_bytes
    ):
        errors.append(
            "evidence manifest uncompressed byte total does not match"
        )
    training_metadata_count = sum(
        isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and _allowed_checkpoint_metadata(
            PurePosixPath("project") / entry["path"]
        )
        for entry in manifest_files
    )
    if (
        archive_manifest.get("training_metadata_file_count")
        != training_metadata_count
    ):
        errors.append("training metadata file count does not match")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                errors.append("ZIP contains duplicate member names")
            for name in names:
                if not _safe_member_name(name):
                    errors.append(f"unsafe ZIP member path: {name}")
                reason = _forbidden_member_reason(name)
                if reason is not None:
                    errors.append(f"forbidden ZIP member ({reason}): {name}")
            embedded_name = "evidence-manifest.json"
            if embedded_name not in names:
                errors.append("embedded evidence manifest is missing")
            else:
                embedded = archive.read(embedded_name)
                if embedded != evidence_manifest_bytes:
                    errors.append(
                        "embedded and external evidence manifests differ"
                    )
            actual_members = set(names) - {embedded_name}
            if actual_members != set(expected_members):
                missing = sorted(set(expected_members) - actual_members)
                extra = sorted(actual_members - set(expected_members))
                if missing:
                    errors.append(
                        f"ZIP is missing {len(missing)} manifest files"
                    )
                if extra:
                    errors.append(
                        f"ZIP contains {len(extra)} unmanifested files"
                    )
            for name in sorted(actual_members & set(expected_members)):
                value = archive.read(name)
                entry = expected_members[name]
                if len(value) != entry["size_bytes"]:
                    errors.append(f"ZIP member size mismatch: {name}")
                if _sha256_bytes(value) != entry["sha256"]:
                    errors.append(f"ZIP member hash mismatch: {name}")
    except (
        KeyError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
    ) as exc:
        errors.append(f"cannot read evidence ZIP: {exc}")

    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "archive": archive_path.as_posix(),
        "archive_sha256": (
            _sha256_file(archive_path) if archive_path.is_file() else None
        ),
        "file_count": evidence_manifest.get("file_count"),
    }


def main() -> None:
    args = parse_args()
    result = verify_evidence_archive(args.archive_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
