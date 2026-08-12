"""Build a deterministic, checkpoint-free confirmatory evidence extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
SEED_ROOT = REPORTS_ROOT / "full_cross_modal_quality" / "seed-replication"
DEFAULT_OUTPUT = (
    REPORTS_ROOT / "submission-evidence" / "confirmatory-extension"
)
CODE_FILES = (
    "audit_v3_seed_replication.py",
    "audit_v3_seed_replication_recovery.py",
    "benchmark_training_input_pipeline.py",
    "build_confirmatory_extension_archive.py",
    "build_confirmatory_extension_index.py",
    "build_v3_seed_replication_addendum.py",
    "build_v3_seed_replication_protocol.py",
    "build_v3_seed_replication_recovery_addendum.py",
    "capture_v3_seed_replication_queue_incident.py",
    "prepare_equivalent_fast_config.py",
    "prepare_seed_replicates.py",
    "summarize_seed_evaluations.py",
    "audit_checkpoints.py",
    "evaluate.py",
    "train.py",
    "verify_confirmatory_extension_archive.py",
    "mmfc/checkpoints.py",
    "mmfc/data.py",
    "mmfc/evaluation.py",
    "mmfc/losses.py",
    "mmfc/model.py",
    "mmfc/training.py",
    "reports/qwen-material/training/run_v3_seed_replication.ps1",
    "reports/qwen-material/training/run_v3_seed_replication_recovery.ps1",
)
CONFIG_FILES = (
    "configs/ablations/full_cross_modal_quality.json",
    "configs/replicates/full_cross_modal_quality_seed43.json",
    "configs/replicates/full_cross_modal_quality_seed44.json",
    "configs/performance/full_cross_modal_quality_workers4.json",
    "configs/performance/full_cross_modal_quality_workers4.provenance.json",
)
CHECKPOINT_METADATA = (
    "checkpoints/ablations/full_cross_modal_quality/history.json",
    "checkpoints/ablations/full_cross_modal_quality/training_summary.json",
    (
        "checkpoints/ablations/v3_seed_replicates/"
        "full_cross_modal_quality_seed43/history.json"
    ),
    (
        "checkpoints/ablations/v3_seed_replicates/"
        "full_cross_modal_quality_seed43/training_summary.json"
    ),
    (
        "checkpoints/ablations/v3_seed_replicates/"
        "full_cross_modal_quality_seed44/history.json"
    ),
    (
        "checkpoints/ablations/v3_seed_replicates/"
        "full_cross_modal_quality_seed44/training_summary.json"
    ),
)
BASE_LINK_FILES = (
    "reports/qwen-material/SUBMISSION_INDEX.json",
    "reports/qwen-material/FINAL_EVIDENCE_AUDIT.json",
    "reports/qwen-material/PAPER_READINESS_AUDIT.json",
    "reports/qwen-material/submission-evidence/archive-manifest.json",
    "reports/qwen-material/submission-evidence/evidence-manifest.json",
)
EXTENSION_FILES = (
    "reports/qwen-material/CONFIRMATORY_EXTENSION_INDEX.json",
    "reports/qwen-material/CONFIRMATORY_EXTENSION_INDEX.md",
    "reports/qwen-material/training/INPUT_PIPELINE_BENCHMARK.json",
    "reports/qwen-material/training/INPUT_PIPELINE_BENCHMARK.md",
    (
        "reports/qwen-material/training/"
        "full_cross_modal_quality.seed_replicates_status.json"
    ),
    (
        "reports/qwen-material/training/"
        "full_cross_modal_quality.seed_replicates_recovery_status.json"
    ),
)
TEST_FILES = (
    "tests/test_equivalent_fast_config.py",
    "tests/test_training_input_pipeline_benchmark.py",
    "tests/test_v3_seed_replication.py",
)
EXCLUDED_SUFFIXES = {".pt", ".log", ".pid", ".tmp"}
SECRET_PATTERNS = (re.compile(rb"(?i)\bsk-[a-z0-9_-]{16,}"),)
TEXT_SUFFIXES = {
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the confirmatory evidence-extension archive"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def collect_extension_files() -> list[Path]:
    files = {
        (PROJECT_ROOT / relative).resolve()
        for relative in (
            *CODE_FILES,
            *CONFIG_FILES,
            *CHECKPOINT_METADATA,
            *BASE_LINK_FILES,
            *EXTENSION_FILES,
            *TEST_FILES,
        )
    }
    files.update(
        path.resolve()
        for path in SEED_ROOT.rglob("*")
        if path.is_file()
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "confirmatory archive inputs missing: "
            + ", ".join(path.as_posix() for path in sorted(missing))
        )
    selected = []
    for path in files:
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        path.relative_to(PROJECT_ROOT)
        selected.append(path)
    return sorted(
        selected,
        key=lambda item: item.relative_to(PROJECT_ROOT).as_posix(),
    )


def _refuse_secrets(files: Iterable[Path]) -> None:
    flagged = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            flagged.append(path)
    if flagged:
        raise ValueError(
            "secret-like tokens found in extension evidence: "
            + ", ".join(path.as_posix() for path in flagged)
        )


def _evidence_manifest(files: list[Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        entries.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    base_archive = (
        REPORTS_ROOT / "submission-evidence" / "paper-evidence.zip"
    )
    return {
        "schema_version": 1,
        "scope": "confirmatory_v3_training_seed_replication_extension",
        "base_paper_evidence_archive": {
            "path": base_archive.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": base_archive.stat().st_size,
            "sha256": _sha256(base_archive),
        },
        "file_count": len(entries),
        "files": entries,
        "exclusions": {
            "model_checkpoints": True,
            "raw_images": True,
            "original_patent": True,
            "api_credentials": True,
        },
        "prediction_jsonl_included": any(
            entry["path"].endswith("predictions.jsonl")
            for entry in entries
        ),
    }


def _write_zip(
    archive_path: Path,
    *,
    files: list[Path],
    evidence_manifest: dict[str, Any],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        manifest_info = zipfile.ZipInfo("evidence-manifest.json")
        manifest_info.date_time = (1980, 1, 1, 0, 0, 0)
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o644 << 16
        archive.writestr(
            manifest_info,
            json.dumps(
                evidence_manifest,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            + b"\n",
        )
        for source in files:
            info = zipfile.ZipInfo(
                source.relative_to(PROJECT_ROOT).as_posix()
            )
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    temporary.replace(archive_path)


def build_archive(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("confirmatory extension archive directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = collect_extension_files()
    _refuse_secrets(files)
    evidence_manifest = _evidence_manifest(files)
    evidence_manifest_path = output_dir / "evidence-manifest.json"
    _atomic_write(
        evidence_manifest_path,
        json.dumps(evidence_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    archive_path = output_dir / "confirmatory-extension.zip"
    _write_zip(
        archive_path,
        files=files,
        evidence_manifest=evidence_manifest,
    )
    archive_manifest = {
        "schema_version": 1,
        "archive": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256(archive_path),
        "evidence_manifest": evidence_manifest_path.name,
        "evidence_manifest_sha256": _sha256(evidence_manifest_path),
        "file_count": evidence_manifest["file_count"],
        "base_paper_evidence_archive_sha256": evidence_manifest[
            "base_paper_evidence_archive"
        ]["sha256"],
        "model_checkpoints_included": False,
        "raw_images_included": False,
        "original_patent_included": False,
        "prediction_jsonl_included": evidence_manifest[
            "prediction_jsonl_included"
        ],
    }
    _atomic_write(
        output_dir / "archive-manifest.json",
        json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return archive_manifest


def main() -> None:
    report = build_archive(parse_args().output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
