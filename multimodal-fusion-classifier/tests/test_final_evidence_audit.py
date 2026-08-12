import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import audit_final_evidence as audit_module
from audit_final_evidence import (
    _audit_ablation_evaluation,
    _audit_file_entries,
    _audit_protocol_input_hashes,
    _audit_stratified_rows,
    _dependency_state,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_dependency_state_distinguishes_complete_pending_and_failed(tmp_path):
    errors = []
    pending = []
    status = tmp_path / "status.json"
    _write_json(status, {"state": "complete"})
    assert _dependency_state(
        status,
        "stage",
        allow_pending=True,
        errors=errors,
        pending=pending,
    )

    _write_json(status, {"state": "running"})
    assert not _dependency_state(
        status,
        "stage",
        allow_pending=True,
        errors=errors,
        pending=pending,
    )
    assert pending

    _write_json(status, {"state": "failed", "message": "boom"})
    assert not _dependency_state(
        status,
        "stage",
        allow_pending=True,
        errors=errors,
        pending=pending,
    )
    assert any("boom" in error for error in errors)


def test_dependency_state_defers_prior_failure_only_for_pending_audit(tmp_path):
    status = tmp_path / "status.json"
    _write_json(status, {"state": "failed", "message": "old upstream failure"})
    errors = []
    pending = []

    assert not _dependency_state(
        status,
        "downstream stage",
        allow_pending=True,
        errors=errors,
        pending=pending,
        defer_failed=True,
        defer_reason="replacement upstream",
    )

    assert errors == []
    assert any("replacement upstream is active" in item for item in pending)

    strict_errors = []
    strict_pending = []
    assert not _dependency_state(
        status,
        "downstream stage",
        allow_pending=False,
        errors=strict_errors,
        pending=strict_pending,
        defer_failed=True,
        defer_reason="replacement upstream",
    )
    assert any("old upstream failure" in item for item in strict_errors)
    assert strict_pending == []


def test_pending_audit_defers_stale_downstream_failures_during_restart(
    tmp_path,
    monkeypatch,
):
    reports = tmp_path / "reports"
    training = reports / "training"
    _write_json(
        training / "formal_ablations.status.json",
        {"state": "waiting", "stage": "restarted_training"},
    )
    for name in (
        "formal_ablations.reporting_status.json",
        "yolo_seg.status.json",
        "sensor_source_transfer.status.json",
        "sensor_source_retraining.status.json",
    ):
        _write_json(
            training / name,
            {"state": "failed", "message": "superseded failure"},
        )
    monkeypatch.setattr(audit_module, "_audit_dataset", lambda *_: {})

    pending_report = audit_module.audit_final_evidence(
        reports_root=reports,
        allow_pending=True,
    )
    strict_report = audit_module.audit_final_evidence(
        reports_root=reports,
        allow_pending=False,
    )

    assert pending_report["state"] == "pending"
    assert pending_report["error_count"] == 0
    assert pending_report["pending_count"] == 5
    assert strict_report["state"] == "failed"
    assert strict_report["error_count"] == 5


def test_ablation_evaluation_audit_accepts_fixed_test_protocol(tmp_path):
    report_dir = tmp_path / "vision_only"
    evaluation_path = report_dir / "evaluation.json"
    class_names = [f"class-{index}" for index in range(19)]
    _write_json(
        evaluation_path,
        {
            "metrics": {
                "sample_count": 826,
                "accuracy": 0.8,
                "macro_precision": 0.7,
                "macro_recall": 0.6,
                "macro_f1": 0.65,
                "weighted_f1": 0.78,
                "ece": 0.03,
            },
            "dataset": {
                "split": "test",
                "evaluated_samples": 826,
            },
            "evaluation_config": {"bootstrap_repeats": 2000},
            "class_names": class_names,
        },
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "predictions.jsonl").write_text(
        "\n".join("{}" for _ in range(826)) + "\n",
        encoding="utf-8",
    )
    errors = []

    result = _audit_ablation_evaluation(
        "vision_only",
        evaluation_path,
        class_names,
        errors,
    )

    assert result == class_names
    assert errors == []


def test_ablation_evaluation_audit_rejects_wrong_split_and_count(tmp_path):
    report_dir = tmp_path / "vision_only"
    evaluation_path = report_dir / "evaluation.json"
    _write_json(
        evaluation_path,
        {
            "metrics": {
                "sample_count": 825,
                "accuracy": 0.8,
                "macro_precision": 0.7,
                "macro_recall": 0.6,
                "macro_f1": 0.65,
                "weighted_f1": 0.78,
                "ece": 0.03,
            },
            "dataset": {
                "split": "val",
                "evaluated_samples": 825,
            },
            "evaluation_config": {"bootstrap_repeats": 100},
            "class_names": ["only-one"],
        },
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    errors = []

    _audit_ablation_evaluation(
        "vision_only",
        evaluation_path,
        None,
        errors,
    )

    assert any("sample_count" in error for error in errors)
    assert any("dataset.split" in error for error in errors)
    assert any("bootstrap_repeats" in error for error in errors)
    assert any("prediction_count" in error for error in errors)


def test_protocol_input_audit_detects_post_evaluation_drift(tmp_path, monkeypatch):
    source = tmp_path / "locked.json"
    source.write_bytes(b"locked")
    monkeypatch.setattr(
        "audit_final_evidence.PROJECT_ROOT",
        tmp_path,
    )
    protocol = {
        "inputs": {
            "source": {
                "path": "locked.json",
                "sha256": hashlib.sha256(b"locked").hexdigest(),
            }
        }
    }
    errors = []
    assert _audit_protocol_input_hashes(
        protocol,
        prefix="protocol",
        errors=errors,
    ) == 1
    assert errors == []

    source.write_bytes(b"changed")
    _audit_protocol_input_hashes(
        protocol,
        prefix="protocol",
        errors=errors,
    )
    assert any("hash mismatch" in error for error in errors)


def test_file_entry_audit_checks_size_and_hash(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"content")
    monkeypatch.setattr("audit_final_evidence.PROJECT_ROOT", tmp_path)
    entries = [
        {
            "path": "source.py",
            "size_bytes": 7,
            "sha256": hashlib.sha256(b"content").hexdigest(),
        }
    ]
    errors = []

    assert _audit_file_entries(
        entries,
        prefix="manifest",
        errors=errors,
    ) == 1
    assert errors == []

    entries[0]["size_bytes"] = 8
    _audit_file_entries(entries, prefix="manifest", errors=errors)
    assert any("size mismatch" in error for error in errors)


def test_file_entry_audit_accepts_bound_hash_override(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.py"
    source.write_bytes(b"revised")
    monkeypatch.setattr("audit_final_evidence.PROJECT_ROOT", tmp_path)
    entries = [
        {
            "path": "source.py",
            "sha256": hashlib.sha256(b"original").hexdigest(),
        }
    ]
    errors = []

    assert _audit_file_entries(
        entries,
        prefix="manifest",
        errors=errors,
        expected_hash_overrides={
            "source.py": hashlib.sha256(b"revised").hexdigest()
        },
    ) == 1
    assert errors == []


def test_stratified_row_audit_checks_combinations_support_and_intervals():
    protocol = {
        "evaluation": {
            "confidence_thresholds": [0.25],
            "prediction_sources": ["detector"],
        },
        "strata": {
            "scene_density": {
                "expected_support": {"single_object": 2}
            }
        },
    }
    row = {
        "confidence_threshold": 0.25,
        "prediction_source": "detector",
        "dimension": "scene_density",
        "stratum": "single_object",
        "support": 2,
        "localized_count": 1,
        "correct_count": 1,
        "localization_recall": 0.5,
        "localization_recall_ci_lower": 0.0,
        "localization_recall_ci_upper": 1.0,
        "classification_accuracy_on_localized": 1.0,
        "classification_accuracy_ci_lower": 1.0,
        "classification_accuracy_ci_upper": 1.0,
        "end_to_end_recall": 0.5,
        "end_to_end_recall_ci_lower": 0.0,
        "end_to_end_recall_ci_upper": 1.0,
    }
    errors = []

    _audit_stratified_rows([row], protocol, errors)

    assert errors == []
    row["support"] = 3
    row["end_to_end_recall_ci_upper"] = 0.4
    _audit_stratified_rows([row], protocol, errors)
    assert any("support" in error for error in errors)
    assert any("outside its interval" in error for error in errors)


def test_pending_cli_default_does_not_write_strict_artifact(
    tmp_path,
    monkeypatch,
):
    reports = tmp_path / "reports"
    monkeypatch.setattr(
        audit_module,
        "parse_args",
        lambda: SimpleNamespace(
            reports=reports,
            output=None,
            allow_pending=True,
        ),
    )
    monkeypatch.setattr(
        audit_module,
        "audit_final_evidence",
        lambda **_: {
            "state": "pending",
            "valid": False,
            "error_count": 0,
            "pending_count": 1,
            "warning_count": 0,
            "errors": [],
            "pending": ["waiting"],
            "warnings": [],
            "sections": {},
        },
    )

    audit_module.main()

    assert (reports / "FINAL_EVIDENCE_AUDIT.pending.json").is_file()
    assert not (reports / "FINAL_EVIDENCE_AUDIT.json").exists()


def test_strict_cli_default_writes_final_artifact(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    monkeypatch.setattr(
        audit_module,
        "parse_args",
        lambda: SimpleNamespace(
            reports=reports,
            output=None,
            allow_pending=False,
        ),
    )
    monkeypatch.setattr(
        audit_module,
        "audit_final_evidence",
        lambda **_: {
            "state": "complete",
            "valid": True,
            "error_count": 0,
            "pending_count": 0,
            "warning_count": 0,
            "errors": [],
            "pending": [],
            "warnings": [],
            "sections": {},
        },
    )

    audit_module.main()

    assert (reports / "FINAL_EVIDENCE_AUDIT.json").is_file()
    assert not (reports / "FINAL_EVIDENCE_AUDIT.pending.json").exists()
