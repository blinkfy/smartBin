import hashlib
import json
from pathlib import Path

from audit_image_recovery_outcome import build_recovery_outcome
from audit_paper_readiness import audit_image_recovery_outcome_bundle


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    files = {
        name: tmp_path / relative
        for name, relative in {
            "incident": "incident.json",
            "addendum": "addendum.json",
            "recovery_audit": "recovery.json",
            "source_integrity": "integrity.json",
            "history": "history.json",
            "checkpoint_audit": "checkpoint.json",
            "evaluation": "evaluation.json",
            "resume_stdout": "resume.stdout.log",
            "resume_stderr": "resume.stderr.log",
            "runtime_shim": "sitecustomize.py",
            "recovery_chain": "recovery.ps1",
            "config": "config.json",
            "train_manifest": "train.jsonl",
            "test_manifest": "test.jsonl",
        }.items()
    }
    files["runtime_shim"].write_text("shim", encoding="utf-8")
    files["recovery_chain"].write_text("chain", encoding="utf-8")
    files["resume_stdout"].write_text(
        json.dumps(
            {
                "event": "resumed",
                "checkpoint": "last.pt",
                "completed_epochs": 5,
                "target_epochs": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    files["resume_stderr"].write_text("", encoding="utf-8")
    _write_json(
        files["incident"],
        {
            "valid": True,
            "snapshot": {
                "partial_training": {"completed_epochs": 5}
            },
        },
    )
    _write_json(
        files["addendum"],
        {
            "valid": True,
            "resume_attempt_2": {"completed_epochs_before": 5},
            "functional_revision": {
                "sha256": _sha256(files["runtime_shim"]),
                "recovery_chain_sha256": _sha256(
                    files["recovery_chain"]
                ),
            },
        },
    )
    _write_json(files["recovery_audit"], {"valid": True})
    _write_json(
        files["source_integrity"],
        {
            "valid": True,
            "summary": {
                "decode_failure_count": 1,
                "documented_recovery_count": 1,
            },
        },
    )
    _write_json(
        files["history"],
        {
            "config_fingerprint": "fingerprint",
            "history": [
                {"epoch": epoch} for epoch in range(1, 101)
            ],
        },
    )
    _write_json(
        files["checkpoint_audit"],
        {
            "valid": True,
            "history_json_rows": 100,
            "training_summary_completed_epochs": 100,
            "expected": {"config_fingerprint": "fingerprint"},
            "best": {
                "epoch": 91,
                "best_val_accuracy": 0.91,
                "sha256": "best-hash",
            },
            "last": {"epoch": 100, "sha256": "last-hash"},
        },
    )
    _write_json(
        files["evaluation"],
        {
            "metrics": {
                "sample_count": 826,
                "accuracy": 0.88,
                "macro_precision": 0.81,
                "macro_recall": 0.85,
                "macro_f1": 0.82,
                "weighted_f1": 0.88,
                "ece": 0.025,
                "negative_log_likelihood": 0.38,
                "multiclass_brier_score": 0.18,
            },
            "dataset": {
                "split": "test",
                "evaluated_samples": 826,
                "full_split_samples": 826,
                "sensor_degradation_enabled": False,
            },
            "checkpoint": {"epoch": 91},
        },
    )
    _write_json(files["config"], {"epochs": 100})
    files["train_manifest"].write_text("{}\n", encoding="utf-8")
    files["test_manifest"].write_text("{}\n", encoding="utf-8")
    return files


def test_recovery_outcome_binds_resume_completion_and_test(tmp_path):
    report = build_recovery_outcome(files=_fixture(tmp_path))

    assert report["valid"] is True
    assert report["recovery"]["completed_epochs_before_resume"] == 5
    assert report["recovery"]["completed_epochs_after_resume"] == 100
    assert report["training"]["best_epoch"] == 91
    assert report["test"]["sample_count"] == 826
    assert set(report["source_bindings"]) == {
        "incident",
        "addendum",
        "recovery_audit",
        "source_integrity",
        "history",
        "checkpoint_audit",
        "evaluation",
        "resume_stdout",
        "resume_stderr",
        "runtime_shim",
        "recovery_chain",
        "config",
        "train_manifest",
        "test_manifest",
    }


def test_recovery_outcome_rejects_nonempty_stderr(tmp_path):
    files = _fixture(tmp_path)
    files["resume_stderr"].write_text("failure", encoding="utf-8")

    report = build_recovery_outcome(files=files)

    assert report["valid"] is False
    assert "attempt 2 stderr is not empty" in report["errors"]


def test_readiness_audit_rechecks_recovery_outcome_bindings(tmp_path):
    report_path = tmp_path / "IMAGE_READ_RECOVERY_OUTCOME.json"
    report = build_recovery_outcome(files=_fixture(tmp_path))
    _write_json(report_path, report)
    report_path.with_suffix(".md").write_text("# outcome\n", encoding="utf-8")

    result = audit_image_recovery_outcome_bundle(
        report_path,
        project_root=tmp_path,
    )

    assert result["valid"] is True
    assert result["source_binding_count"] == 14

    Path(report["source_bindings"]["config"]["path"]).write_text(
        "changed",
        encoding="utf-8",
    )
    tampered = audit_image_recovery_outcome_bundle(
        report_path,
        project_root=tmp_path,
    )
    assert tampered["valid"] is False
    assert any("source binding drift: config" in error for error in tampered["errors"])
