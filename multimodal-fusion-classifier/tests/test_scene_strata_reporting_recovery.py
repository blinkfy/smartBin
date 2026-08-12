import csv
import hashlib
import json
from pathlib import Path

import pytest

import build_scene_strata_reporting_recovery as builder
import recover_scene_strata_report as recovery


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _binding(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _evaluation() -> dict:
    rows = []
    for index in range(42):
        rows.append(
            {
                "confidence_threshold": 0.25 if index < 21 else 0.8,
                "prediction_source": "detector",
                "dimension": "relative_bbox_area",
                "stratum": f"stratum-{index}",
                "support": 10,
                "localization_recall": 0.9,
                "localization_recall_ci_lower": 0.8,
                "localization_recall_ci_upper": 1.0,
                "classification_accuracy_on_localized": 0.7,
                "classification_accuracy_ci_lower": 0.6,
                "classification_accuracy_ci_upper": 0.8,
                "end_to_end_recall": 0.6,
                "end_to_end_recall_ci_lower": 0.5,
                "end_to_end_recall_ci_upper": 0.7,
            }
        )
    return {
        "protocol": "preregistered_scene_stratified_analysis",
        "dataset": {
            "source_image_count": 570,
            "ground_truth_object_count": 825,
        },
        "rows": rows,
    }


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row"])
        writer.writeheader()
        writer.writerows({"row": index} for index in range(42))


def test_report_recovery_uses_persisted_interval_prefix(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    reports = project / "reports" / "qwen-material"
    output_dir = reports / "yolo-seg" / "integrated" / "stratified"
    evaluation_path = output_dir / "evaluation.json"
    metrics_path = output_dir / "metrics.csv"
    incident_path = (
        reports / "yolo-seg" / "SCENE_STRATA_REPORTING_INCIDENT.json"
    )
    addendum_path = (
        reports
        / "yolo-seg"
        / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
    )
    outcome_path = (
        reports
        / "yolo-seg"
        / "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json"
    )
    source_path = project / "recover_scene_strata_report.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("source", encoding="utf-8")
    _write_json(evaluation_path, _evaluation())
    _write_metrics(metrics_path)
    _write_json(
        incident_path,
        {
            "schema_version": 1,
            "state": "complete",
            "valid": True,
            "failure": {"exception_type": "KeyError"},
            "partial_outputs": {"row_count": 42},
        },
    )
    _write_json(
        addendum_path,
        {
            "schema_version": 1,
            "status": "frozen_after_metrics_complete_before_report_recovery",
            "valid": True,
            "incident": {"sha256": _sha256(incident_path)},
            "invariants": {
                "metric_recomputation": False,
                "training_rerun": False,
                "detector_rerun": False,
                "expected_row_count": 42,
                "allowed_new_files": [
                    "REPORT.md",
                    "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json",
                    "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.md",
                ],
            },
            "source_bindings": {
                "partial_evaluation": _binding(evaluation_path, project),
                "partial_metrics": _binding(metrics_path, project),
                "recovery_source": _binding(source_path, project),
            },
            "errors": [],
        },
    )
    monkeypatch.setattr(recovery, "PROJECT_ROOT", project.resolve())
    monkeypatch.setattr(recovery, "REPORTS_ROOT", reports.resolve())
    monkeypatch.setattr(recovery, "__file__", str(source_path))
    before = (_sha256(evaluation_path), _sha256(metrics_path))

    result = recovery.recover_scene_strata_report(
        output_dir=output_dir,
        incident_path=incident_path,
        addendum_path=addendum_path,
        outcome_path=outcome_path,
    )

    assert result["valid"] is True
    assert before == (_sha256(evaluation_path), _sha256(metrics_path))
    report = (output_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "70.00% [60.00%,80.00%]" in report
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["recovery"]["metric_hashes_unchanged"] is True
    assert outcome["recovery"]["metric_recomputation"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        recovery.recover_scene_strata_report(
            output_dir=output_dir,
            incident_path=incident_path,
            addendum_path=addendum_path,
            outcome_path=outcome_path,
        )


def test_reporting_builder_freezes_partial_metrics_before_recovery(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    reports = project / "reports" / "qwen-material"
    archive = (
        reports
        / "training"
        / "interrupted"
        / "final_evidence_scene_strata_partial_20260728"
    )
    output_dir = reports / "yolo-seg" / "integrated" / "stratified"
    evaluation_path = output_dir / "evaluation.json"
    metrics_path = output_dir / "metrics.csv"
    _write_json(evaluation_path, _evaluation())
    _write_metrics(metrics_path)
    archive.mkdir(parents=True)
    (archive / "evaluation.json").write_bytes(evaluation_path.read_bytes())
    (archive / "metrics.csv").write_bytes(metrics_path.read_bytes())
    evaluator = project / "evaluate_scene_strata.py"
    evaluator.parent.mkdir(parents=True, exist_ok=True)
    evaluator.write_text("frozen evaluator", encoding="utf-8")
    (
        archive / "evaluate_scene_strata.pre_reporting_recovery.py"
    ).write_bytes(evaluator.read_bytes())
    (archive / "field_mapping_diagnostic.stderr.log").write_text(
        "KeyError: classification_accuracy_on_localized_ci_lower",
        encoding="utf-16",
    )
    for name in (
        "final_evidence.log",
        "final_evidence.status.json",
        "sensor_retraining_post_exit_recovery.status.json",
        "sensor_retraining_post_exit_recovery_v3.final_evidence.stderr.log",
        "sensor_retraining_post_exit_recovery_v3.final_evidence.stdout.log",
        "STRATIFIED_EVALUATION_PROTOCOL.json",
        "run_final_evidence_audit.ps1",
    ):
        (archive / name).write_text(name, encoding="utf-8")
    protocol_addendum = (
        reports / "yolo-seg" / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
    )
    _write_json(
        protocol_addendum,
        {
            "implementation_fix": {
                "revised_strata_source": {"sha256": _sha256(evaluator)}
            }
        },
    )
    (archive / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json").write_bytes(
        protocol_addendum.read_bytes()
    )
    required_sources = (
        "build_scene_strata_reporting_recovery.py",
        "recover_scene_strata_report.py",
        "audit_final_evidence.py",
        "audit_paper_readiness.py",
        "audit_manuscript_stable_sections.py",
        "build_submission_index.py",
        "build_evidence_archive.py",
        "reports/qwen-material/training/"
        "run_scene_strata_reporting_recovery_v4.ps1",
        "reports/qwen-material/training/run_final_evidence_audit.ps1",
        "reports/qwen-material/yolo-seg/"
        "STRATIFIED_EVALUATION_PROTOCOL.json",
        "reports/qwen-material/yolo-seg/integrated/evaluation.json",
        "reports/qwen-material/SENSOR_RETRAINING_EXIT_OUTCOME.json",
    )
    for relative in required_sources:
        path = project / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    builder_source = project / "build_scene_strata_reporting_recovery.py"
    monkeypatch.setattr(builder, "PROJECT_ROOT", project.resolve())
    monkeypatch.setattr(builder, "REPORTS_ROOT", reports.resolve())
    monkeypatch.setattr(builder, "__file__", str(builder_source))
    incident_path = (
        reports / "yolo-seg" / "SCENE_STRATA_REPORTING_INCIDENT.json"
    )
    addendum_path = (
        reports
        / "yolo-seg"
        / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
    )

    result = builder.build_reporting_recovery(
        archive_dir=archive,
        output_dir=output_dir,
        incident_path=incident_path,
        addendum_path=addendum_path,
    )

    assert result["valid"] is True
    assert result["row_count"] == 42
    assert result["source_binding_count"] == 17
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    assert incident["archive"]["file_count"] == 12
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    assert addendum["invariants"]["metric_recomputation"] is False
    assert addendum["invariants"]["bootstrap_rerun"] is False
