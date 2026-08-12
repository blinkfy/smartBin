"""Build a deterministic checkpoint-free v6 patent-stack evidence archive."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile

from mmfc.model import PROJECT_ROOT
from train_patent_aligned_external_sensor_joint_seg import DEFAULT_NAME


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
RUN_ROOT = (
    REPORTS
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
)
DEFAULT_OUTPUT = (
    REPORTS
    / "submission-evidence"
    / "patent-stack-extension"
    / "patent-stack-extension.zip"
)
SOURCE_FILES = (
    "mmfc/joint_seg.py",
    "mmfc/joint_seg_external.py",
    "mmfc/joint_seg_reliable.py",
    "mmfc/joint_seg_contrastive.py",
    "mmfc/joint_seg_spatial.py",
    "mmfc/joint_seg_patent.py",
    "mmfc/model.py",
    "mmfc/training.py",
    "train_patent_aligned_external_sensor_joint_seg.py",
    "evaluate_patent_aligned_conditions.py",
    "audit_patent_aligned_joint_seg.py",
    "build_patent_requirement_matrix.py",
    "build_patent_stack_extension_index.py",
    "build_patent_stack_extension_archive.py",
    "build_patent_stack_manuscript_addendum.py",
    "benchmark_patent_stack_runtime.py",
    "prepare_patent_stack_intervention.py",
    "evaluate_patent_stack_intervention.py",
    "audit_patent_stack_intervention.py",
    "plot_patent_stack_architecture.py",
    "prepare_patent_stack_qwen_formal_protocol.py",
    "prepare_qwen_sensor_derangement.py",
    "train_patent_aligned_qwen_formal.py",
    "evaluate_patent_aligned_qwen_formal_conditions.py",
    "aggregate_patent_stack_qwen_formal.py",
    "run_patent_stack_qwen_formal.py",
    "audit_patent_stack_qwen_lineage.py",
    "build_paper_motivation_decision_packet.py",
    "build_paper_readiness_v6_addendum.py",
    "audit_external_sensor_worker_scaling_4_to_8.py",
    "prepare_standardized_instance_strata_protocol.py",
    "evaluate_standardized_instance_strata.py",
    "prepare_standardized_strata_runtime_addendum.py",
    "prepare_standardized_strata_chunking_addendum.py",
    "prepare_standardized_strata_numerical_adjudication.py",
    "prepare_standardized_strata_final_validation_addendum.py",
    "audit_standardized_instance_strata.py",
    "build_standardized_strata_manuscript_addendum.py",
    "plot_standardized_instance_strata.py",
    "tests/test_joint_seg_patent.py",
    "tests/test_patent_aligned_training.py",
    "tests/test_patent_aligned_conditions.py",
    "tests/test_patent_evidence_audit.py",
    "tests/test_patent_stack_extension_index.py",
    "tests/test_patent_stack_extension_archive.py",
    "tests/test_patent_stack_manuscript_addendum.py",
    "tests/test_patent_stack_runtime_benchmark.py",
    "tests/test_patent_stack_intervention.py",
    "tests/test_patent_stack_intervention_audit.py",
    "tests/test_patent_stack_architecture_figure.py",
    "tests/test_patent_stack_qwen_formal_extension.py",
    "tests/test_paper_motivation_decision_packet.py",
    "tests/test_paper_readiness_v6_addendum.py",
    "tests/test_external_sensor_worker_scaling.py",
    "tests/test_standardized_instance_strata.py",
    "tests/test_standardized_instance_strata_audit.py",
    "tests/test_standardized_strata_manuscript_addendum.py",
    "tests/test_standardized_instance_strata_figure.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v6 patent-stack evidence archive"
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


def _patent_document() -> Path:
    matches = sorted(PROJECT_ROOT.glob("2026.4.3-1*.docx"))
    if len(matches) != 1:
        raise FileNotFoundError("expected exactly one 2026.4.3-1 patent DOCX")
    return matches[0]


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
            REPORTS / "PATENT_REQUIREMENT_MATRIX.json",
            REPORTS / "PATENT_REQUIREMENT_MATRIX.md",
            REPORTS / "PATENT_STACK_EXTENSION_INDEX.json",
            REPORTS / "PATENT_STACK_EXTENSION_INDEX.md",
            REPORTS / "PATENT_STACK_MANUSCRIPT_ADDENDUM.json",
            REPORTS / "PATENT_STACK_MANUSCRIPT_ADDENDUM.md",
            REPORTS / "PATENT_STACK_COMPLEXITY.json",
            REPORTS / "PATENT_STACK_COMPLEXITY.md",
            REPORTS / "PATENT_STACK_INTERVENTION_AUDIT.json",
            REPORTS / "PATENT_STACK_INTERVENTION_AUDIT.md",
            REPORTS / "PATENT_STACK_FIGURE_CAPTION.md",
            REPORTS / "MOTIVATION_DECISION_PACKET.json",
            REPORTS / "MOTIVATION_DECISION_PACKET.md",
            REPORTS / "PAPER_READINESS_V6_ADDENDUM.json",
            REPORTS / "PAPER_READINESS_V6_ADDENDUM.md",
            REPORTS
            / "joint-multimodal-seg-external"
            / "WORKER_SCALING_4_TO_8.json",
            REPORTS
            / "joint-multimodal-seg-external"
            / "WORKER_SCALING_4_TO_8.md",
            REPORTS
            / "standardized-instance-strata"
            / "PROTOCOL.json",
            REPORTS
            / "standardized-instance-strata"
            / "PROTOCOL.md",
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_RECOVERY_ADDENDUM.json",
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_RECOVERY_ADDENDUM.md",
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_CHUNKING_ADDENDUM.json",
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_CHUNKING_ADDENDUM.md",
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_NUMERICAL_ADJUDICATION.json",
            REPORTS
            / "standardized-instance-strata"
            / "RUNTIME_NUMERICAL_ADJUDICATION.md",
            REPORTS
            / "standardized-instance-strata"
            / "FINAL_VALIDATION_ADDENDUM.json",
            REPORTS
            / "standardized-instance-strata"
            / "FINAL_VALIDATION_ADDENDUM.md",
            REPORTS
            / "standardized-instance-strata"
            / "AUDIT.json",
            REPORTS
            / "standardized-instance-strata"
            / "AUDIT.md",
            REPORTS
            / "standardized-instance-strata"
            / "MANUSCRIPT_ADDENDUM.json",
            REPORTS
            / "standardized-instance-strata"
            / "MANUSCRIPT_ADDENDUM.md",
            REPORTS
            / "standardized-instance-strata"
            / "standardized_instance_strata.json",
            REPORTS
            / "standardized-instance-strata"
            / "FIGURE_CAPTION.md",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation"
            / "rgb_yolo.json",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-recovery-batch8"
            / "rgb_yolo.json",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-recovery-batch8-chunk64"
            / "rgb_yolo.json",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-recovery-batch8-chunk64"
            / "v5_rgb_fallback.json",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-recovery-batch8-chunk64"
            / "v6_rgb_fallback.json",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "evaluation.json",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "REPORT.md",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "metrics.csv",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "paired_differences.csv",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "rgb_yolo.json",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "v5_rgb_fallback.json",
            REPORTS
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "v6_rgb_fallback.json",
            REPORTS
            / "joint-multimodal-seg-patent"
            / "patent_stack_architecture.json",
            REPORTS / "SPATIAL_RELIABILITY_AUDIT.json",
            REPORTS / "SPATIAL_RELIABILITY_AUDIT.md",
            REPORTS
            / "joint-multimodal-seg-external"
            / "QWEN_FORMAL_PROTOCOL.json",
            REPORTS
            / "joint-multimodal-seg-external"
            / "QWEN_FORMAL_PROTOCOL.md",
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PROTOCOL.json",
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PROTOCOL.md",
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_LINEAGE_AUDIT.json",
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_LINEAGE_AUDIT.md",
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PREFLIGHT.json",
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PREFLIGHT.md",
            RUN_ROOT / "PROTOCOL.json",
            RUN_ROOT / "evaluation.json",
            RUN_ROOT / "RELIABILITY_CONDITIONS.json",
            RUN_ROOT / "OUTCOME.json",
            RUN_ROOT / "OUTCOME.md",
            RUN_ROOT / "PATENT_STACK_INTERVENTION_PROTOCOL.json",
            RUN_ROOT / "PATENT_STACK_INTERVENTION_PROTOCOL.md",
            RUN_ROOT / "PATENT_STACK_INTERVENTION.json",
            RUN_ROOT / "PATENT_STACK_INTERVENTION.md",
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg-patent"
            / DEFAULT_NAME
            / "args.yaml",
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg-patent"
            / DEFAULT_NAME
            / "results.csv",
        ]
    )
    unique = sorted(set(path.resolve() for path in files), key=_archive_name)
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"patent-stack archive inputs missing: {missing}")
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
    patent_document = _patent_document()
    manifest = {
        "schema_version": 1,
        "archive": "patent-stack-extension",
        "evidence_role": "patent_stack_structural_mechanism_development",
        "formal_paper_metrics_included": False,
        "rule_sidecar_metrics_promoted_to_formal": False,
        "checkpoint_included": False,
        "source_images_included": False,
        "annotation_files_included": False,
        "rendered_figures_included": False,
        "patent_document_included": False,
        "credentials_included": False,
        "hardware_sensor_claim": False,
        "rgb_d_claim": False,
        "external_patent_binding": {
            "path": patent_document.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": patent_document.stat().st_size,
            "sha256": _sha256(patent_document),
        },
        "file_count_excluding_manifest": len(files),
        "bindings": bindings,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(_zip_info("ARCHIVE_MANIFEST.json"), manifest_bytes)
        for path in files:
            archive.writestr(_zip_info(_archive_name(path)), path.read_bytes())
    temporary.replace(output)

    errors: list[str] = []
    secret_pattern = re.compile(rb"sk-[A-Za-z0-9_-]{16,}")
    forbidden_suffixes = {
        ".pt",
        ".onnx",
        ".engine",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".docx",
    }
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
        forbidden = [
            name
            for name in names
            if Path(name).suffix.lower() in forbidden_suffixes
        ]
        if forbidden:
            errors.append(
                f"forbidden binary, image, or DOCX files included: {forbidden[:5]}"
            )
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
