"""Build a deterministic, checkpoint-free paper evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_FILES = (
    "analyze_class_metric_intervals.py",
    "analyze_error_patterns.py",
    "audit_bibliography.py",
    "audit_final_evidence.py",
    "audit_formal_queue_interruption_outcome.py",
    "audit_image_recovery.py",
    "audit_image_recovery_outcome.py",
    "audit_manuscript_stable_sections.py",
    "audit_method_contract.py",
    "audit_paper_readiness.py",
    "audit_sensor_retraining_exit_outcome.py",
    "audit_source_image_repair_transition.py",
    "audit_source_image_integrity.py",
    "audit_uncertainty_floor.py",
    "audit_variant_lineage.py",
    "audit_yolo_seg_dependency_outcome.py",
    "build_claim_evidence.py",
    "build_evidence_archive.py",
    "build_full_regression_audit.py",
    "build_paper_tables.py",
    "build_plan_a_delivery_manifest.py",
    "build_plan_a_submission_index.py",
    "build_scene_path_identity_addendum.py",
    "build_scene_strata_protocol_addendum.py",
    "build_scene_strata_reporting_recovery.py",
    "build_sensor_retraining_exit_addendum.py",
    "build_sensor_retraining_encoding_addendum.py",
    "build_submission_index.py",
    "capture_image_recovery_incident.py",
    "capture_formal_queue_interruption.py",
    "capture_formal_queue_interruption_addendum.py",
    "capture_yolo_seg_dependency_incident.py",
    "capture_sensor_retraining_exit_incident.py",
    "archive_yolo_seg_dependency_incident.py",
    "evaluate.py",
    "evaluate_uncertainty.py",
    "evaluate_scene_pipeline.py",
    "evaluate_scene_strata.py",
    "recover_scene_strata_report.py",
    "evaluate_sensor_sources.py",
    "compare_predictions.py",
    "compare_uncertainty_diagnostics.py",
    "prepare_sensor_source_retraining.py",
    "repair_sensor_retraining_log_encoding.py",
    "plot_paper_tables.py",
    "train.py",
    "train_yolo_seg.py",
    "verify_evidence_archive.py",
    "mmfc/__init__.py",
    "mmfc/ablations.py",
    "mmfc/checkpoints.py",
    "mmfc/data.py",
    "mmfc/evaluation.py",
    "mmfc/losses.py",
    "mmfc/model.py",
    "mmfc/robustness.py",
    "mmfc/scene_inference.py",
    "mmfc/sensors.py",
    "mmfc/training.py",
    "mmfc/uncertainty_diagnostics.py",
    "runtime_recovery/sitecustomize.py",
)
SOURCE_FILES = (
    "README.md",
    "requirements.txt",
    "requirements-repro.txt",
    "requirements-dev.txt",
    "pytest.ini",
)
SOURCE_GLOBS = (
    "*.py",
    "mmfc/*.py",
    "runtime_recovery/*.py",
    "configs/**/*.json",
    "checkpoints/ablations/*/history.json",
    "checkpoints/ablations/*/training_summary.json",
    "tests/*.py",
    "reports/qwen-material/training/*.ps1",
)
REPORT_GLOBS = (
    "*.md",
    "*.bib",
    "*.json",
    "*.csv",
    "*.png",
    "*.pdf",
    "paper-tables/*",
    "ablations/*",
    "ablations/*/evaluation.json",
    "ablations/*/classification_report.csv",
    "ablations/*/confusion_matrix.csv",
    "ablations/*/confusion_matrix.png",
    "ablations/*/CHECKPOINT_AUDIT.json",
    "full_quality_aware_prior/REPORT.md",
    "full_quality_aware_prior/*.json",
    "full_quality_aware_prior/*.csv",
    "full_quality_aware_prior/*.png",
    "full_quality_aware_prior/*.pdf",
    "full_quality_aware_prior/test/evaluation.json",
    "full_quality_aware_prior/test/classification_report.csv",
    "full_quality_aware_prior/test/confusion_matrix.csv",
    "full_quality_aware_prior/test/confusion_matrix.png",
    "full_quality_aware_prior/uncertainty/*",
    "full_cross_modal_quality/REPORT.md",
    "full_cross_modal_quality/*.json",
    "full_cross_modal_quality/*.csv",
    "full_cross_modal_quality/*.png",
    "full_cross_modal_quality/*.pdf",
    "full_cross_modal_quality/test/evaluation.json",
    "full_cross_modal_quality/test/classification_report.csv",
    "full_cross_modal_quality/test/confusion_matrix.csv",
    "full_cross_modal_quality/test/confusion_matrix.png",
    "full_cross_modal_quality/uncertainty/*",
    "full_cross_modal_quality_balanced/REPORT.md",
    "full_cross_modal_quality_balanced/*.json",
    "full_cross_modal_quality_balanced/*.csv",
    "full_cross_modal_quality_balanced/*.png",
    "full_cross_modal_quality_balanced/*.pdf",
    "full_cross_modal_quality_balanced/test/evaluation.json",
    "full_cross_modal_quality_balanced/test/classification_report.csv",
    "full_cross_modal_quality_balanced/test/confusion_matrix.csv",
    "full_cross_modal_quality_balanced/test/confusion_matrix.png",
    "yolo-seg/*.json",
    "yolo-seg/*.md",
    "yolo-seg/*.csv",
    "yolo-seg/*.png",
    "yolo-seg/*.pdf",
    "yolo-seg/integrated/evaluation.json",
    "yolo-seg/integrated/*.csv",
    "yolo-seg/integrated/*.md",
    "yolo-seg/integrated/*.png",
    "yolo-seg/integrated/*.pdf",
    "yolo-seg/integrated/stratified/*",
    "sensor-source-transfer/PROTOCOL.json",
    "sensor-source-transfer/evaluation.json",
    "sensor-source-transfer/REPORT.md",
    "sensor-source-transfer/*/evaluation.json",
    "sensor-source-transfer/*/classification_report.csv",
    "sensor-source-transfer/*/confusion_matrix.csv",
    "sensor-source-transfer/*/confusion_matrix.png",
    "sensor-source-retraining/PROTOCOL.json",
    "sensor-source-retraining/resolved_rule_v1_config.json",
    "sensor-source-retraining/*.json",
    "sensor-source-retraining/*.md",
    "sensor-source-retraining/rule-v1/CHECKPOINT_AUDIT.json",
    "sensor-source-retraining/rule-v1/test/evaluation.json",
    "sensor-source-retraining/rule-v1/test/classification_report.csv",
    "sensor-source-retraining/rule-v1/test/confusion_matrix.csv",
    "sensor-source-retraining/rule-v1/test/confusion_matrix.png",
    "yolo-seg/SCENE_STRATA_REPORTING_INCIDENT.json",
    "yolo-seg/SCENE_STRATA_REPORTING_INCIDENT.md",
    "yolo-seg/SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json",
    "yolo-seg/SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.md",
    "yolo-seg/SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json",
    "yolo-seg/SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.md",
)
PLAN_A_TRAINING_RUN_DIRS = (
    (
        "checkpoints/confirmatory-ablations/"
        "v3_no_reliability_workers0/seed42"
    ),
    (
        "checkpoints/confirmatory-ablations/"
        "v3_no_reliability_workers0/seed43"
    ),
    (
        "checkpoints/confirmatory-ablations/"
        "v3_no_reliability_workers0/seed44"
    ),
    (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        "v3_no_grid_reconstruction/seed42"
    ),
    (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        "v3_no_grid_reconstruction/seed43"
    ),
    (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        "v3_no_grid_reconstruction/seed44"
    ),
    (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        "v3_no_ranking/seed42"
    ),
    (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        "v3_no_ranking/seed43"
    ),
    (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        "v3_no_ranking/seed44"
    ),
    "checkpoints/recent-robust-baselines/qmf/seed42",
    "checkpoints/recent-robust-baselines/qmf/seed43",
    "checkpoints/recent-robust-baselines/qmf/seed44",
    "checkpoints/recent-robust-baselines/tmc/seed42",
)
PLAN_A_REQUIRED_FILES = (
    "reports/qwen-material/PLAN_A_SUBMISSION_INDEX.json",
    "reports/qwen-material/PLAN_A_SUBMISSION_INDEX.md",
    "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V12.json",
    "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V12.md",
    "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V13.json",
    "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V13.md",
    "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V14.json",
    "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V14.md",
    "reports/qwen-material/PLAN_A_PAPER_READINESS_AUDIT.json",
    (
        "reports/qwen-material/"
        "PLAN_A_MANUSCRIPT_STABLE_SECTIONS_AUDIT.json"
    ),
    (
        "reports/qwen-material/"
        "PLAN_A_MANUSCRIPT_STABLE_SECTIONS_AUDIT.md"
    ),
    (
        "reports/qwen-material/confirmatory-ablations/"
        "v3-loss-components/OUTCOME.json"
    ),
    (
        "reports/qwen-material/unseen-sensor-conditions-v1/"
        "OUTCOME.json"
    ),
    "reports/qwen-material/recent-robust-baselines/OUTCOME.json",
    (
        "reports/qwen-material/recent-robust-baselines/"
        "PROTOCOL_AMENDMENT.json"
    ),
    (
        "reports/qwen-material/recent-robust-baselines/"
        "PROTOCOL_AMENDMENT.md"
    ),
    "reports/qwen-material/plan-a-confirmatory-outcome/OUTCOME.json",
    (
        "reports/qwen-material/training/"
        "v3_loss_component_ablation_status.json"
    ),
    (
        "reports/qwen-material/training/"
        "unseen_sensor_conditions_status.json"
    ),
    (
        "reports/qwen-material/training/"
        "recent_robust_baseline_status.json"
    ),
    (
        "reports/qwen-material/training/"
        "plan_a_manuscript_refresh_status.json"
    ),
    "paper_rewriting_output/final_paper/main.tex",
    "paper_rewriting_output/final_paper/references.bib",
    "paper_rewriting_output/final_paper/paper.zh.docx",
    "paper_rewriting_output/final_paper/paper.pdf",
    "paper_rewriting_output/final_paper/main.pdf",
    "paper_rewriting_output/final_artifact_manifest.md",
    "paper_rewriting_output/latex_report.md",
    "paper_rewriting_output/word_report.zh.md",
    "paper_rewriting_output/artifact_check.md",
    "paper_rewriting_output/citation_bank_check.md",
    "paper_rewriting_output/citation_quality_audit.md",
    "paper_rewriting_output/integrity_audit.md",
    "paper_rewriting_output/progress.md",
    "paper_rewriting_output/plan_a_confirmatory_revision_audit.json",
    "paper_rewriting_output/plan_a_automated_build_audit.json",
    "paper_rewriting_output/plan_a_automated_build_audit.md",
    "paper_rewriting_output/plan_a_visual_qa.json",
    "paper_rewriting_output/plan_a_visual_qa.md",
    (
        "paper_rewriting_output/plan_a_visual_qa/"
        "REVIEW_MANIFEST.json"
    ),
    (
        "paper_rewriting_output/plan_a_visual_qa/"
        "REVIEW_DECISION.json"
    ),
    *(
        f"{run_dir}/{filename}"
        for run_dir in PLAN_A_TRAINING_RUN_DIRS
        for filename in ("history.json", "training_summary.json")
    ),
)
PLAN_A_GLOBS = (
    "reports/qwen-material/confirmatory-ablations/v3-loss-components/**/*",
    "reports/qwen-material/unseen-sensor-conditions-v1/**/*",
    "reports/qwen-material/recent-robust-baselines/**/*",
    "reports/qwen-material/plan-a-confirmatory-outcome/**/*",
    "paper_rewriting_output/plan_a_visual_qa/pages/*.png",
    "checkpoints/confirmatory-ablations/**/history.json",
    "checkpoints/confirmatory-ablations/**/training_summary.json",
    "checkpoints/recent-robust-baselines/**/history.json",
    "checkpoints/recent-robust-baselines/**/training_summary.json",
)
EXCLUDED_NAMES = {
    "FINAL_EVIDENCE_AUDIT.pending.json",
    "PAPER_READINESS_AUDIT.pending.json",
    "PROJECT_COMPLETION_AUDIT_V1.json",
    "PROJECT_COMPLETION_AUDIT_V1.md",
    "PLAN_A_SUBMISSION_INDEX.json",
    "PLAN_A_SUBMISSION_INDEX.md",
}
EXCLUDED_SUFFIXES = {".pt", ".log", ".pid", ".jsonl", ".tmp"}
TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    re.compile(rb"(?i)\bsk-[a-z0-9_-]{16,}"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic paper evidence ZIP without raw data"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "submission-evidence"
        ),
    )
    parser.add_argument(
        "--include-plan-a",
        action="store_true",
        help=(
            "Require and include the visually approved Plan A manuscript, "
            "confirmatory outcomes, and page-review evidence"
        ),
    )
    return parser.parse_args()


def _resolve(path: Path, *, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refuse_secret_like_text(files: Iterable[Path]) -> None:
    flagged = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            flagged.append(path)
    if flagged:
        paths = ", ".join(path.as_posix() for path in flagged)
        raise ValueError(
            "refusing to archive files with secret-like tokens: " + paths
        )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)


def collect_evidence_files(
    *,
    project_root: Path,
    reports_root: Path,
    include_plan_a: bool = False,
) -> list[Path]:
    project_root = project_root.resolve()
    reports_root = reports_root.resolve()
    files: set[Path] = set()
    for relative in CODE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"archive source code is missing: {path}")
        files.add(path.resolve())
    for relative in SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"archive reproducibility file is missing: {path}"
            )
        files.add(path.resolve())
    for pattern in SOURCE_GLOBS:
        for path in project_root.glob(pattern):
            if path.is_file():
                files.add(path.resolve())
    for pattern in REPORT_GLOBS:
        for path in reports_root.glob(pattern):
            if not path.is_file():
                continue
            if path.name in EXCLUDED_NAMES:
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            if "submission-evidence" in path.parts:
                continue
            files.add(path.resolve())
    required = (
        reports_root / "FINAL_EVIDENCE_AUDIT.json",
        reports_root / "PAPER_READINESS_AUDIT.json",
        reports_root / "FINAL_CLAIM_EVIDENCE.json",
        reports_root / "FINAL_CLAIM_EVIDENCE.md",
        reports_root / "SUBMISSION_INDEX.json",
        reports_root / "SUBMISSION_INDEX.md",
        reports_root / "paper-tables" / "manifest.json",
        reports_root / "paper-tables" / "TABLES.md",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"required final evidence is missing: {path}"
            )
        files.add(path.resolve())
    if include_plan_a:
        plan_a_paths = [
            (project_root / relative).resolve()
            for relative in PLAN_A_REQUIRED_FILES
        ]
        missing = [path for path in plan_a_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "required Plan A release evidence is missing: "
                + ", ".join(path.as_posix() for path in missing)
            )
        files.update(plan_a_paths)
        for pattern in PLAN_A_GLOBS:
            for path in project_root.glob(pattern):
                if (
                    path.is_file()
                    and path.suffix.lower() not in EXCLUDED_SUFFIXES
                    and "interrupted"
                    not in path.relative_to(project_root).parts
                ):
                    files.add(path.resolve())
    return sorted(
        files,
        key=lambda path: path.relative_to(project_root).as_posix(),
    )


def _manifest(
    files: Iterable[Path],
    *,
    project_root: Path,
    readiness_path: Path,
    include_plan_a: bool = False,
) -> dict[str, Any]:
    entries = []
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "size_bytes": size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 2 if include_plan_a else 1,
        "scope": (
            "visually_approved_plan_a_internal_paper_evidence"
            if include_plan_a
            else "curated_internal_paper_evidence"
        ),
        "source_readiness_audit": (
            readiness_path.relative_to(project_root).as_posix()
        ),
        "source_readiness_audit_sha256": _sha256(readiness_path),
        "file_count": len(entries),
        "total_uncompressed_bytes": total_bytes,
        "exclusions": [
            "original patent source document",
            "raw images and annotations",
            "Qwen/rule-v1 JSONL sidecars",
            "model checkpoints",
            "training logs and PID/status control files",
            "per-sample prediction JSONL",
            "pending audit snapshots",
        ],
        "plan_a_release_included": include_plan_a,
        "files": entries,
    }


def _write_deterministic_zip(
    path: Path,
    files: Iterable[Path],
    *,
    project_root: Path,
    evidence_manifest_text: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        manifest_info = zipfile.ZipInfo(
            "evidence-manifest.json",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o100644 << 16
        archive.writestr(
            manifest_info,
            evidence_manifest_text.encode("utf-8"),
        )
        for source in files:
            relative = source.relative_to(project_root).as_posix()
            info = zipfile.ZipInfo(
                f"project/{relative}",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    temporary.replace(path)


def build_evidence_archive(
    *,
    project_root: Path,
    reports_root: Path,
    output_dir: Path,
    include_plan_a: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    reports_root = _resolve(reports_root, root=project_root)
    output_dir = _resolve(output_dir, root=project_root)
    readiness_path = reports_root / (
        "PLAN_A_PAPER_READINESS_AUDIT.json"
        if include_plan_a
        else "PAPER_READINESS_AUDIT.json"
    )
    if not readiness_path.is_file():
        raise FileNotFoundError(readiness_path)
    readiness = _read_json(readiness_path)
    if (
        readiness.get("state") != "complete"
        or readiness.get("internal_paper_evidence_ready") is not True
    ):
        raise ValueError("internal paper evidence is not ready")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty archive directory: {output_dir}"
        )
    files = collect_evidence_files(
        project_root=project_root,
        reports_root=reports_root,
        include_plan_a=include_plan_a,
    )
    if include_plan_a:
        plan_a_index = _read_json(
            reports_root / "PLAN_A_SUBMISSION_INDEX.json"
        )
        if (
            plan_a_index.get("state") != "complete"
            or plan_a_index.get("valid") is not True
            or plan_a_index.get("gates", {}).get(
                "plan_a_visual_qa_complete"
            )
            is not True
            or plan_a_index.get("gates", {}).get(
                "paper_spine_final_gate_passed"
            )
            is not True
        ):
            raise ValueError("Plan A submission index is not release-ready")
    _refuse_secret_like_text(files)
    manifest = _manifest(
        files,
        project_root=project_root,
        readiness_path=readiness_path,
        include_plan_a=include_plan_a,
    )
    training_metadata_file_count = sum(
        path.name in {"history.json", "training_summary.json"}
        and "checkpoints" in path.relative_to(project_root).parts
        for path in files
    )
    manifest_text = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        manifest_path = staging_dir / "evidence-manifest.json"
        _atomic_write(manifest_path, manifest_text)
        archive_path = staging_dir / "paper-evidence.zip"
        _write_deterministic_zip(
            archive_path,
            files,
            project_root=project_root,
            evidence_manifest_text=manifest_text,
        )
        archive_manifest = {
            "schema_version": 1,
            "archive": archive_path.name,
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_sha256": _sha256(archive_path),
            "evidence_manifest": manifest_path.name,
            "evidence_manifest_sha256": _sha256(manifest_path),
            "file_count": manifest["file_count"],
            "original_patent_included": False,
            "raw_data_included": False,
            "annotations_included": False,
            "checkpoints_included": False,
            "model_checkpoints_included": False,
            "training_metadata_file_count": training_metadata_file_count,
            "prediction_jsonl_included": False,
            "plan_a_release_included": include_plan_a,
        }
        archive_manifest_path = staging_dir / "archive-manifest.json"
        _atomic_write(
            archive_manifest_path,
            json.dumps(
                archive_manifest,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        if output_dir.exists():
            output_dir.rmdir()
        staging_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return {
        "state": "complete",
        "output": output_dir.as_posix(),
        **archive_manifest,
    }


def main() -> None:
    args = parse_args()
    result = build_evidence_archive(
        project_root=PROJECT_ROOT,
        reports_root=args.reports,
        output_dir=args.output,
        include_plan_a=args.include_plan_a,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
