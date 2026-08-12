import hashlib
import json
from pathlib import Path

from audit_sensor_retraining_exit_outcome import (
    DEFAULT_FILES as OUTCOME_FILES,
    build_sensor_retraining_exit_outcome,
)
from capture_sensor_retraining_exit_incident import (
    DEFAULT_FILES as INCIDENT_FILES,
    build_sensor_retraining_exit_incident,
)
from build_sensor_retraining_exit_addendum import (
    DEFAULT_FILES as ADDENDUM_FILES,
    build_sensor_retraining_exit_addendum,
)
from repair_sensor_retraining_log_encoding import repair_log_encoding


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_post_completion_exit_incident_accepts_complete_artifacts(
    tmp_path, monkeypatch
):
    files = {name: tmp_path / name for name in INCIDENT_FILES}
    files["queue_source"].write_text(
        "Test-TrainingComplete\n"
        "SKIP rule_v1_training (verified complete)",
        encoding="utf-8",
    )
    files["recovery_queue_source"].write_text(
        "run_sensor_source_retraining.ps1\n"
        "audit_sensor_retraining_exit_outcome.py\n"
        "run_final_evidence_audit.ps1",
        encoding="utf-8",
    )
    for name in ("train_source", "training_source", "model_source"):
        files[name].write_text(name, encoding="utf-8")
    files["best_checkpoint"].write_bytes(b"best")
    files["last_checkpoint"].write_bytes(b"last")
    rows = [{"epoch": epoch} for epoch in range(1, 101)]
    _write_json(
        files["archived_failed_status"],
        {
            "state": "failed",
            "message": "failed with exit code -1073741819",
        },
    )
    _write_json(
        files["archived_continuation_status"],
        {
            "state": "failed",
            "message": "sensor_source_retraining failed",
        },
    )
    files["archived_training_stdout"].write_text("summary", encoding="utf-8")
    files["archived_training_stderr"].write_text("", encoding="utf-8")
    _write_json(files["archived_history"], {"history": rows})
    _write_json(
        files["archived_training_summary"],
        {
            "completed_epochs": 100,
            "checkpoints_saved": True,
            "history": rows,
        },
    )
    _write_json(
        files["post_exit_checkpoint_audit"],
        {
            "valid": True,
            "training_summary_completed_epochs": 100,
            "history_json_rows": 100,
            "expected": {"config_fingerprint": "fingerprint"},
            "best": {
                "epoch": 91,
                "sha256": _sha256(files["best_checkpoint"]),
            },
            "last": {
                "epoch": 100,
                "sha256": _sha256(files["last_checkpoint"]),
            },
        },
    )
    _write_json(files["resolved_config"], {"experiment": "rule"})
    queue_path = (
        "reports/qwen-material/training/"
        "run_sensor_source_retraining.ps1"
    )
    _write_json(
        files["protocol"],
        {
            "status": "frozen_before_rule_v1_retraining",
            "protocol": "independent_sensor_source_retraining",
            "input_hashes": [
                {
                    "path": queue_path,
                    "sha256": _sha256(files["queue_source"]),
                },
                {
                    "path": "train.py",
                    "sha256": _sha256(files["train_source"]),
                },
                {
                    "path": "mmfc/training.py",
                    "sha256": _sha256(files["training_source"]),
                },
                {
                    "path": "mmfc/model.py",
                    "sha256": _sha256(files["model_source"]),
                },
            ],
        },
    )
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "snapshot.txt").write_text("snapshot", encoding="utf-8")
    monkeypatch.setattr(
        "capture_sensor_retraining_exit_incident.PROJECT_ROOT", tmp_path
    )

    report = build_sensor_retraining_exit_incident(
        files=files,
        archive_root=archive,
        downstream_report_root=tmp_path / "rule-v1",
        comparison_path=tmp_path / "comparison.json",
    )

    assert report["valid"] is True
    assert report["completed_training"]["completed_epochs"] == 100
    assert report["recovery"]["downstream_evidence_absent_before_recovery"]


def test_post_completion_exit_outcome_requires_verified_skip(
    tmp_path, monkeypatch
):
    files = {name: tmp_path / name for name in OUTCOME_FILES}
    files["best_checkpoint"].write_bytes(b"best")
    files["last_checkpoint"].write_bytes(b"last")
    rows = [{"epoch": epoch} for epoch in range(1, 101)]
    _write_json(files["protocol"], {"status": "frozen"})
    files["queue_source"].write_text("queue", encoding="utf-8")
    files["recovery_queue_source"].write_text("recovery", encoding="utf-8")
    _write_json(files["inner_status"], {"state": "complete", "stage": "all"})
    files["inner_pipeline_log"].write_text(
        "SKIP rule_v1_training (verified complete)", encoding="utf-8"
    )
    files["training_stdout"].write_text("summary", encoding="utf-8")
    files["archived_training_stdout"].write_text(
        "summary", encoding="utf-8"
    )
    _write_json(files["history"], {"history": rows})
    _write_json(
        files["training_summary"],
        {
            "completed_epochs": 100,
            "checkpoints_saved": True,
            "history": rows,
        },
    )
    best_sha = _sha256(files["best_checkpoint"])
    last_sha = _sha256(files["last_checkpoint"])
    _write_json(
        files["formal_checkpoint_audit"],
        {
            "valid": True,
            "training_summary_completed_epochs": 100,
            "history_json_rows": 100,
            "best": {"epoch": 91, "sha256": best_sha},
            "last": {"epoch": 100, "sha256": last_sha},
        },
    )
    _write_json(
        files["evaluation"],
        {
            "dataset": {"split": "test"},
            "metrics": {"sample_count": 826},
            "evaluation_config": {"bootstrap_repeats": 2000},
        },
    )
    files["predictions"].write_text(
        "".join("{}\n" for _ in range(826)), encoding="utf-8"
    )
    _write_json(
        files["comparison"],
        {
            "reference": {"name": "qwen_retrained"},
            "candidate": {"name": "rule_v1_retrained"},
            "paired_bootstrap": {
                "repeats": 5000,
                "sample_count": 826,
                "cluster_count": 570,
            },
        },
    )
    _write_json(
        files["incident"],
        {
            "state": "complete",
            "valid": True,
            "recovery": {
                "decision": (
                    "reuse_valid_complete_training_bundle_and_"
                    "skip_retraining"
                )
            },
            "completed_training": {
                "best_checkpoint": {"sha256": best_sha},
                "last_checkpoint": {"sha256": last_sha},
            },
            "source_bindings": {
                "archived_training_summary": {
                    "sha256": _sha256(files["training_summary"])
                },
                "archived_history": {
                    "sha256": _sha256(files["history"])
                },
            },
        },
    )
    _write_json(
        files["incident_addendum"],
        {
            "status": (
                "frozen_after_recovery_queue_path_failure_before_recovery"
            ),
            "valid": True,
            "incident": {"sha256": _sha256(files["incident"])},
            "implementation_fix": {
                "revised_recovery_queue": {
                    "sha256": _sha256(files["recovery_queue_source"])
                }
            },
        },
    )
    monkeypatch.setattr(
        "audit_sensor_retraining_exit_outcome.PROJECT_ROOT", tmp_path
    )

    report = build_sensor_retraining_exit_outcome(files=files)

    assert report["valid"] is True
    assert report["recovery"]["training_rerun"] is False
    assert report["recovery"]["test_sample_count"] == 826


def test_recovery_queue_addendum_preserves_training_bundle(
    tmp_path, monkeypatch
):
    files = {name: tmp_path / name for name in ADDENDUM_FILES}
    files["original_recovery_queue"].write_text(
        '"reports\\qwen-material\\training\\interrupted\\" +',
        encoding="utf-8",
    )
    files["revised_recovery_queue"].write_text(
        "$PostExitAuditRelative = (\n"
        "$PostExitAudit = Join-Path $ProjectRoot $PostExitAuditRelative\n"
        "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json\n"
        "run_sensor_source_retraining.ps1\n"
        "audit_sensor_retraining_exit_outcome.py\n"
        "run_final_evidence_audit.ps1\n",
        encoding="utf-8",
    )
    for name in (
        "frozen_training_queue",
        "protocol",
        "builder_source",
        "outcome_audit_source",
    ):
        files[name].write_text(name, encoding="utf-8")
    files["best_checkpoint"].write_bytes(b"best")
    files["last_checkpoint"].write_bytes(b"last")
    _write_json(
        files["post_exit_checkpoint_audit"],
        {
            "valid": True,
            "best": {"sha256": _sha256(files["best_checkpoint"])},
            "last": {"sha256": _sha256(files["last_checkpoint"])},
        },
    )
    _write_json(
        files["incident"],
        {
            "state": "complete",
            "valid": True,
            "source_bindings": {
                "recovery_queue_source": {
                    "size_bytes": files["original_recovery_queue"].stat().st_size,
                    "sha256": _sha256(files["original_recovery_queue"]),
                }
            },
            "completed_training": {
                "best_checkpoint": {
                    "sha256": _sha256(files["best_checkpoint"])
                },
                "last_checkpoint": {
                    "sha256": _sha256(files["last_checkpoint"])
                },
            },
        },
    )
    monkeypatch.setattr(
        "build_sensor_retraining_exit_addendum.PROJECT_ROOT", tmp_path
    )

    report = build_sensor_retraining_exit_addendum(
        files=files,
        downstream_report_root=tmp_path / "rule-v1",
        comparison_path=tmp_path / "comparison.json",
    )

    assert report["valid"] is True
    assert report["invariants"]["training_rerun"] is False


def test_log_encoding_repair_preserves_decoded_text(tmp_path, monkeypatch):
    text = "\n".join(
        (
            "SKIP rule_v1_training (verified complete)",
            "DONE rule_v1_checkpoint_audit",
            "DONE rule_v1_test",
            "DONE rule_v1_vs_qwen_comparison",
        )
    )
    archived = tmp_path / "archived.log"
    current = tmp_path / "current.log"
    archived.write_bytes(text.encode("utf-16"))
    current.write_bytes(archived.read_bytes())
    monkeypatch.setattr(
        "repair_sensor_retraining_log_encoding.PROJECT_ROOT", tmp_path
    )

    report = repair_log_encoding(
        archived_log=archived,
        current_log=current,
    )

    assert report["valid"] is True
    assert report["decoded_text_equal"] is True
    assert current.read_text(encoding="utf-8") == text
