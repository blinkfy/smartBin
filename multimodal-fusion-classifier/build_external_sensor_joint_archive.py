"""Build a deterministic archive for the external-sensor joint-head v2."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
SMOKE_NAME = "yolo11s-seg-joint-external-rule-smoke"
DEFAULT_OUTPUT = (
    REPORTS_ROOT
    / "submission-evidence"
    / "external-sensor-joint-extension"
    / "external-sensor-joint-extension.zip"
)
SOURCE_FILES = (
    "mmfc/scene_sensors.py",
    "mmfc/joint_seg_external.py",
    "build_scene_sensor_sidecars.py",
    "validate_scene_sensor_sidecars.py",
    "train_external_sensor_joint_seg.py",
    "audit_external_sensor_joint_seg.py",
    "evaluate_external_sensor_conditions.py",
    "audit_external_sensor_worker_equivalence.py",
    "run_external_sensor_qwen_formal.py",
    "aggregate_external_sensor_formal.py",
    "prepare_external_sensor_qwen_protocol.py",
    "build_external_sensor_extension_index.py",
    "build_external_sensor_joint_archive.py",
    "tests/test_scene_sensor_sidecars.py",
    "tests/test_joint_seg_external.py",
    "tests/test_external_sensor_joint_training.py",
    "tests/test_external_sensor_joint_audit.py",
    "tests/test_external_sensor_conditions.py",
    "tests/test_external_sensor_worker_equivalence.py",
    "tests/test_external_sensor_qwen_runner.py",
    "tests/test_external_sensor_formal_aggregate.py",
    "tests/test_external_sensor_qwen_protocol.py",
    "tests/test_external_sensor_extension_index.py",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build external-sensor v2 evidence archive"
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


def _collect_files() -> list[Path]:
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
            REPORTS_ROOT / "SCENE_SENSOR_DATA.md",
            REPORTS_ROOT / "PATENT_ALIGNMENT.md",
            REPORTS_ROOT / "MANUSCRIPT_DECISION_BRIEF.md",
            REPORTS_ROOT / "EXTERNAL_SENSOR_EXTENSION_INDEX.json",
            REPORTS_ROOT / "EXTERNAL_SENSOR_EXTENSION_INDEX.md",
            REPORTS_ROOT
            / "joint-multimodal-seg-external"
            / "QWEN_FORMAL_PROTOCOL.json",
            REPORTS_ROOT
            / "joint-multimodal-seg-external"
            / "QWEN_FORMAL_PROTOCOL.md",
            REPORTS_ROOT
            / "joint-multimodal-seg-external"
            / "WORKER_EQUIVALENCE.json",
            REPORTS_ROOT
            / "joint-multimodal-seg-external"
            / "WORKER_EQUIVALENCE.md",
        ]
    )
    smoke_report = (
        REPORTS_ROOT / "joint-multimodal-seg-external" / SMOKE_NAME
    )
    files.extend(path for path in smoke_report.rglob("*") if path.is_file())
    smoke_training = (
        PROJECT_ROOT
        / "checkpoints"
        / "joint-multimodal-seg-external"
        / SMOKE_NAME
    )
    files.extend(
        path
        for path in smoke_training.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in {".pt", ".onnx", ".engine"}
    )
    benchmark_name = "yolo11s-seg-joint-external-rule-workers4-benchmark"
    benchmark_report = (
        REPORTS_ROOT / "joint-multimodal-seg-external" / benchmark_name
    )
    files.extend(
        benchmark_report / name
        for name in ("PROTOCOL.json", "evaluation.json")
    )
    benchmark_training = (
        PROJECT_ROOT
        / "checkpoints"
        / "joint-multimodal-seg-external"
        / benchmark_name
    )
    files.extend(
        benchmark_training / name
        for name in ("args.yaml", "results.csv")
    )
    unique = sorted(set(path.resolve() for path in files), key=_archive_name)
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"external archive inputs missing: {missing}")
    return unique


def build_archive(output: Path = DEFAULT_OUTPUT) -> dict:
    output = _resolve(output)
    files = _collect_files()
    bindings = {
        _archive_name(path): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    }
    manifest = {
        "schema_version": 1,
        "archive": "external-sensor-joint-extension",
        "evidence_role": "integration_smoke_and_qwen_preregistration",
        "formal_qwen_metrics_included": False,
        "rule_smoke_promoted_to_formal": False,
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
        forbidden = [
            name
            for name in names
            if Path(name).suffix.lower() in {".pt", ".onnx", ".engine"}
        ]
        if forbidden:
            errors.append(f"checkpoint-like files included: {forbidden[:5]}")
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
    print(json.dumps(build_archive(parse_args().output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
