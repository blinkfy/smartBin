"""Build a deterministic checkpoint-free v3-v5 reliability archive."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
PRIMARY_NAME = (
    "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
)
CALIBRATED_NAME = (
    "yolo11s-seg-joint-spatial-calibrated-rule-dev-seed42"
)
DEFAULT_OUTPUT = (
    REPORTS
    / "submission-evidence"
    / "reliable-external-sensor-extension"
    / "reliable-external-sensor-extension.zip"
)
SOURCE_FILES = (
    "mmfc/joint_seg.py",
    "mmfc/joint_seg_external.py",
    "mmfc/joint_seg_reliable.py",
    "mmfc/joint_seg_contrastive.py",
    "mmfc/joint_seg_spatial.py",
    "mmfc/model.py",
    "mmfc/training.py",
    "train_reliable_external_sensor_joint_seg.py",
    "train_contrastive_reliable_external_sensor_joint_seg.py",
    "train_spatial_contrastive_external_sensor_joint_seg.py",
    "evaluate_reliable_external_conditions.py",
    "evaluate_contrastive_external_conditions.py",
    "evaluate_spatial_external_conditions.py",
    "diagnose_spatial_sensor_separation.py",
    "audit_spatial_reliability_extension.py",
    "build_reliable_external_extension_index.py",
    "build_reliable_external_extension_archive.py",
    "tests/test_joint_seg_reliable.py",
    "tests/test_reliable_external_sensor_training.py",
    "tests/test_reliable_external_conditions.py",
    "tests/test_joint_seg_contrastive.py",
    "tests/test_contrastive_reliable_training.py",
    "tests/test_contrastive_external_conditions.py",
    "tests/test_joint_seg_spatial.py",
    "tests/test_spatial_contrastive_training.py",
    "tests/test_spatial_external_conditions.py",
    "tests/test_spatial_sensor_separation.py",
    "tests/test_spatial_reliability_audit.py",
    "tests/test_reliable_external_extension_index.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v3-v5 reliable external sensor archive"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_name(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _run_report_files(root: Path, names: tuple[str, ...]) -> list[Path]:
    return [root / name for name in names]


def _training_metadata(project: str, name: str) -> list[Path]:
    root = PROJECT_ROOT / "checkpoints" / project / name
    return [root / "args.yaml", root / "results.csv"]


def collect_files() -> list[Path]:
    files = [PROJECT_ROOT / relative for relative in SOURCE_FILES]
    files.extend(
        [
            PROJECT_ROOT
            / "data"
            / "scene-sensor-sidecars"
            / "observations.jsonl",
            PROJECT_ROOT
            / "data"
            / "scene-sensor-sidecars"
            / "BUILD_REPORT.json",
            PROJECT_ROOT
            / "data"
            / "scene-sensor-sidecars"
            / "VALIDATION_REPORT.json",
            REPORTS / "SCENE_SENSOR_DATA.md",
            REPORTS / "SPATIAL_RELIABILITY_AUDIT.json",
            REPORTS / "SPATIAL_RELIABILITY_AUDIT.md",
            REPORTS / "RELIABLE_EXTERNAL_EXTENSION_INDEX.json",
            REPORTS / "RELIABLE_EXTERNAL_EXTENSION_INDEX.md",
            REPORTS
            / "joint-multimodal-seg-external"
            / "QWEN_FORMAL_PROTOCOL.json",
            REPORTS
            / "joint-multimodal-seg-external"
            / "QWEN_FORMAL_PROTOCOL.md",
        ]
    )
    v3_name = "yolo11s-seg-joint-reliable-rule-dev-seed42-retry4"
    v3_report = (
        REPORTS / "joint-multimodal-seg-reliable" / v3_name
    )
    files.extend(
        _run_report_files(
            v3_report,
            (
                "PROTOCOL.json",
                "evaluation.json",
                "RELIABILITY_CONDITIONS.json",
            ),
        )
    )
    files.extend(
        _training_metadata("joint-multimodal-seg-reliable", v3_name)
    )

    v4_name = (
        "yolo11s-seg-joint-reliable-contrastive-rule-dev-seed42"
    )
    v4_report = (
        REPORTS / "joint-multimodal-seg-contrastive" / v4_name
    )
    files.extend(
        _run_report_files(
            v4_report,
            (
                "PROTOCOL.json",
                "evaluation.json",
                "RELIABILITY_CONDITIONS.json",
            ),
        )
    )
    files.extend(
        _training_metadata("joint-multimodal-seg-contrastive", v4_name)
    )

    primary_report = (
        REPORTS / "joint-multimodal-seg-spatial" / PRIMARY_NAME
    )
    files.extend(
        _run_report_files(
            primary_report,
            (
                "PROTOCOL.json",
                "evaluation.json",
                "RELIABILITY_CONDITIONS.json",
                "SENSOR_SEPARATION.json",
            ),
        )
    )
    files.extend(
        _training_metadata("joint-multimodal-seg-spatial", PRIMARY_NAME)
    )

    calibrated_report = (
        REPORTS / "joint-multimodal-seg-spatial" / CALIBRATED_NAME
    )
    files.extend(
        _run_report_files(
            calibrated_report,
            ("PROTOCOL.json", "evaluation.json"),
        )
    )
    files.extend(
        _training_metadata("joint-multimodal-seg-spatial", CALIBRATED_NAME)
    )
    calibrated_workers0 = (
        REPORTS
        / "joint-multimodal-seg-spatial"
        / f"{CALIBRATED_NAME}-workers0-eval"
    )
    files.extend(
        _run_report_files(
            calibrated_workers0,
            (
                "PROTOCOL.json",
                "evaluation.json",
                "RELIABILITY_CONDITIONS.json",
            ),
        )
    )
    unique = sorted(set(path.resolve() for path in files), key=_archive_name)
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reliable archive inputs missing: {missing}")
    return unique


def build_archive(output: Path = DEFAULT_OUTPUT) -> dict:
    output = _resolve(output)
    files = collect_files()
    bindings = {
        _archive_name(path): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    }
    manifest = {
        "schema_version": 1,
        "archive": "reliable-external-sensor-extension",
        "evidence_role": "single_seed_rule_sidecar_mechanism_development",
        "selected_development_model": "spatial_v5_primary",
        "overcalibrated_candidate_promoted": False,
        "formal_qwen_metrics_included": False,
        "rule_sidecar_metrics_promoted_to_formal": False,
        "checkpoint_included": False,
        "source_images_included": False,
        "annotation_files_included": False,
        "hardware_sensor_claim": False,
        "rgb_d_claim": False,
        "file_count_excluding_manifest": len(files),
        "bindings": bindings,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.writestr(_zip_info("ARCHIVE_MANIFEST.json"), manifest_bytes)
        for path in files:
            archive.writestr(_zip_info(_archive_name(path)), path.read_bytes())
    temporary.replace(output)

    errors: list[str] = []
    secret_pattern = re.compile(rb"sk-[A-Za-z0-9_-]{16,}")
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        if names != ["ARCHIVE_MANIFEST.json", *bindings]:
            errors.append("archive inventory or order differs from manifest")
        if len(names) != len(set(names)):
            errors.append("archive contains duplicate names")
        if json.loads(archive.read("ARCHIVE_MANIFEST.json")) != manifest:
            errors.append("embedded manifest differs")
        for name, binding in bindings.items():
            payload = archive.read(name)
            if (
                len(payload) != binding["size_bytes"]
                or _sha256_bytes(payload) != binding["sha256"]
            ):
                errors.append(f"archive binding mismatch: {name}")
            if secret_pattern.search(payload):
                errors.append(f"credential-like value included: {name}")
        forbidden_suffixes = {
            ".pt",
            ".onnx",
            ".engine",
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
        forbidden = [
            name
            for name in names
            if Path(name).suffix.lower() in forbidden_suffixes
        ]
        if forbidden:
            errors.append(f"forbidden binary/image files included: {forbidden[:5]}")
    result = {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "archive": output.as_posix(),
        "size_bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "member_count": 1 + len(files),
        "verified_at": datetime.now().astimezone().isoformat(),
    }
    verification = output.parent / "ARCHIVE_VERIFICATION.json"
    verification.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise ValueError("; ".join(errors))
    return result


def main() -> None:
    print(
        json.dumps(
            build_archive(parse_args().output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
