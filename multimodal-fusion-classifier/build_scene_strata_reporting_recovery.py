"""Freeze the report-only recovery after scene-strata metrics were persisted."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_ARCHIVE = (
    REPORTS_ROOT
    / "training"
    / "interrupted"
    / "final_evidence_scene_strata_partial_20260728"
)
DEFAULT_OUTPUT_DIR = REPORTS_ROOT / "yolo-seg" / "integrated" / "stratified"
DEFAULT_INCIDENT = (
    REPORTS_ROOT / "yolo-seg" / "SCENE_STRATA_REPORTING_INCIDENT.json"
)
DEFAULT_ADDENDUM = (
    REPORTS_ROOT
    / "yolo-seg"
    / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze scene-strata report-only recovery evidence"
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--incident", type=Path, default=DEFAULT_INCIDENT)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _row_count(metrics_path: Path) -> int:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def build_reporting_recovery(
    *,
    archive_dir: Path,
    output_dir: Path,
    incident_path: Path,
    addendum_path: Path,
) -> dict[str, Any]:
    archive_dir = _resolve(archive_dir)
    output_dir = _resolve(output_dir)
    incident_path = _resolve(incident_path)
    addendum_path = _resolve(addendum_path)
    evaluation_path = output_dir / "evaluation.json"
    metrics_path = output_dir / "metrics.csv"
    report_path = output_dir / "REPORT.md"
    outcome_path = (
        REPORTS_ROOT
        / "yolo-seg"
        / "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json"
    )
    if incident_path.exists() or addendum_path.exists():
        raise FileExistsError("refusing to overwrite reporting governance files")
    if report_path.exists() or outcome_path.exists():
        raise ValueError("reporting recovery output already exists")
    for path in (archive_dir, output_dir):
        path.relative_to(PROJECT_ROOT)
    required_archive = (
        "evaluation.json",
        "metrics.csv",
        "evaluate_scene_strata.pre_reporting_recovery.py",
        "field_mapping_diagnostic.stderr.log",
        "final_evidence.log",
        "final_evidence.status.json",
        "sensor_retraining_post_exit_recovery.status.json",
        "sensor_retraining_post_exit_recovery_v3.final_evidence.stderr.log",
        "sensor_retraining_post_exit_recovery_v3.final_evidence.stdout.log",
        "STRATIFIED_EVALUATION_PROTOCOL.json",
        "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json",
        "run_final_evidence_audit.ps1",
    )
    missing = [
        name for name in required_archive if not (archive_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "reporting incident archive is incomplete: " + ", ".join(missing)
        )
    if not evaluation_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("persisted scene-strata metric artifacts are missing")
    evaluation = _read_json(evaluation_path)
    rows = list(evaluation.get("rows", []))
    csv_rows = _row_count(metrics_path)
    if (
        evaluation.get("protocol")
        != "preregistered_scene_stratified_analysis"
        or len(rows) != 42
        or csv_rows != 42
        or evaluation.get("dataset", {}).get("source_image_count") != 570
        or evaluation.get("dataset", {}).get("ground_truth_object_count") != 825
    ):
        raise ValueError("persisted scene-strata metrics are incomplete")
    if _sha256(evaluation_path) != _sha256(archive_dir / "evaluation.json"):
        raise ValueError("current and archived evaluation JSON differ")
    if _sha256(metrics_path) != _sha256(archive_dir / "metrics.csv"):
        raise ValueError("current and archived metrics CSV differ")
    frozen_source = (
        archive_dir / "evaluate_scene_strata.pre_reporting_recovery.py"
    )
    current_source = PROJECT_ROOT / "evaluate_scene_strata.py"
    protocol_addendum = _read_json(
        REPORTS_ROOT
        / "yolo-seg"
        / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
    )
    expected_source_hash = (
        protocol_addendum.get("implementation_fix", {})
        .get("revised_strata_source", {})
        .get("sha256")
    )
    if (
        _sha256(frozen_source) != expected_source_hash
        or _sha256(current_source) != expected_source_hash
    ):
        raise ValueError("frozen scene-strata evaluator identity mismatch")
    diagnostic_text = (
        archive_dir / "field_mapping_diagnostic.stderr.log"
    ).read_text(encoding="utf-16")
    if (
        "KeyError" not in diagnostic_text
        or "classification_accuracy_on_localized_ci_lower"
        not in diagnostic_text
    ):
        raise ValueError("field-mapping diagnostic does not bind the failure")

    archive_bindings = {
        name: _binding(archive_dir / name) for name in required_archive
    }
    incident = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "failure": {
            "stage": "scene_strata_markdown_rendering",
            "exception_type": "KeyError",
            "missing_key": "classification_accuracy_on_localized_ci_lower",
            "persisted_interval_prefix": "classification_accuracy",
            "process_exit_code": 1,
        },
        "partial_outputs": {
            "evaluation": _binding(evaluation_path),
            "metrics": _binding(metrics_path),
            "report_missing": True,
            "row_count": len(rows),
            "csv_row_count": csv_rows,
            "source_image_count": 570,
            "ground_truth_object_count": 825,
        },
        "scope": {
            "metric_computation_completed": True,
            "metric_values_invalidated": False,
            "training_or_detector_affected": False,
            "strict_final_outputs_created": False,
        },
        "archive": {
            "path": archive_dir.relative_to(PROJECT_ROOT).as_posix(),
            "file_count": len(archive_bindings),
            "bindings": archive_bindings,
        },
        "errors": [],
    }
    _atomic_write(
        incident_path,
        json.dumps(incident, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        incident_path.with_suffix(".md"),
        "\n".join(
            [
                "# Scene-strata reporting incident",
                "",
                "- The preregistered computation persisted all 42 JSON/CSV rows.",
                "- Markdown rendering failed on an interval-field name lookup.",
                "- No training, detector inference, matching, bootstrap, or metric "
                "definition is invalidated.",
                "- Strict final evidence artifacts were not created.",
                "",
            ]
        ),
    )

    source_paths = {
        "builder_source": Path(__file__),
        "recovery_source": PROJECT_ROOT / "recover_scene_strata_report.py",
        "recovery_queue_v4": (
            REPORTS_ROOT
            / "training"
            / "run_scene_strata_reporting_recovery_v4.ps1"
        ),
        "frozen_evaluator": current_source,
        "archived_frozen_evaluator": frozen_source,
        "final_evidence_queue": (
            REPORTS_ROOT / "training" / "run_final_evidence_audit.ps1"
        ),
        "final_evidence_audit": PROJECT_ROOT / "audit_final_evidence.py",
        "paper_readiness_audit": PROJECT_ROOT / "audit_paper_readiness.py",
        "stable_manuscript_audit": (
            PROJECT_ROOT / "audit_manuscript_stable_sections.py"
        ),
        "submission_index_builder": PROJECT_ROOT / "build_submission_index.py",
        "evidence_archive_builder": PROJECT_ROOT / "build_evidence_archive.py",
        "stratified_protocol": (
            REPORTS_ROOT
            / "yolo-seg"
            / "STRATIFIED_EVALUATION_PROTOCOL.json"
        ),
        "stratified_protocol_addendum": (
            REPORTS_ROOT
            / "yolo-seg"
            / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
        ),
        "integrated_report": (
            REPORTS_ROOT / "yolo-seg" / "integrated" / "evaluation.json"
        ),
        "partial_evaluation": evaluation_path,
        "partial_metrics": metrics_path,
        "sensor_retraining_outcome": (
            REPORTS_ROOT / "SENSOR_RETRAINING_EXIT_OUTCOME.json"
        ),
    }
    missing_sources = [
        name for name, path in source_paths.items() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "reporting addendum sources are missing: "
            + ", ".join(missing_sources)
        )
    addendum = {
        "schema_version": 1,
        "status": "frozen_after_metrics_complete_before_report_recovery",
        "valid": True,
        "incident": {
            "path": incident_path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": incident_path.stat().st_size,
            "sha256": _sha256(incident_path),
        },
        "implementation_fix": {
            "failure": (
                "Markdown renderer derived interval keys from the displayed "
                "metric name"
            ),
            "recovery": (
                "map classification_accuracy_on_localized to the persisted "
                "classification_accuracy interval prefix"
            ),
            "scope": "presentation_artifact_only",
        },
        "invariants": {
            "metric_recomputation": False,
            "training_rerun": False,
            "detector_rerun": False,
            "matching_rerun": False,
            "bootstrap_rerun": False,
            "expected_row_count": 42,
            "confidence_thresholds": [0.25, 0.8],
            "bootstrap_repeats": 2000,
            "bootstrap_seed": 20260727,
            "allowed_new_files": [
                "REPORT.md",
                "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json",
                "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.md",
            ],
        },
        "source_bindings": {
            name: _binding(path) for name, path in source_paths.items()
        },
        "errors": [],
    }
    _atomic_write(
        addendum_path,
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        addendum_path.with_suffix(".md"),
        "\n".join(
            [
                "# Scene-strata reporting protocol addendum",
                "",
                "- Frozen after all 42 metric rows were persisted and before "
                "report recovery.",
                "- The recovery may create only the missing Markdown report and "
                "its outcome record.",
                "- `evaluation.json` and `metrics.csv` must remain byte-identical.",
                "- Training, detector inference, matching, bootstrap, thresholds, "
                "and metric definitions are unchanged.",
                "",
            ]
        ),
    )
    return {
        "state": "complete",
        "valid": True,
        "incident": incident_path.as_posix(),
        "addendum": addendum_path.as_posix(),
        "row_count": len(rows),
        "source_binding_count": len(source_paths),
    }


def main() -> None:
    args = parse_args()
    result = build_reporting_recovery(
        archive_dir=args.archive,
        output_dir=args.output_dir,
        incident_path=args.incident,
        addendum_path=args.addendum,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
