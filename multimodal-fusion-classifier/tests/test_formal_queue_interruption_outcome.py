import hashlib
import json
from pathlib import Path

from audit_formal_queue_interruption_outcome import (
    build_interruption_outcome,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    files = {
        name: tmp_path / name
        for name in (
            "snapshot",
            "addendum",
            "current_integrity",
            "config",
            "history",
            "training_summary",
            "best_checkpoint",
            "last_checkpoint",
            "checkpoint_audit",
            "evaluation",
            "predictions",
            "restart_stdout",
            "restart_stderr",
            "continuation_script",
            "archived_history",
            "archived_best",
            "archived_last",
            "train_manifest",
            "test_manifest",
            "sensor_manifest",
        )
    }
    for name in (
        "config",
        "best_checkpoint",
        "last_checkpoint",
        "restart_stdout",
        "restart_stderr",
        "continuation_script",
        "archived_best",
        "archived_last",
        "train_manifest",
        "test_manifest",
        "sensor_manifest",
    ):
        files[name].write_bytes(name.encode("ascii"))
    files["restart_stderr"].write_bytes(b"")
    files["predictions"].write_text("{}\n{}\n{}\n", encoding="utf-8")
    fingerprint = "f" * 64
    inventory = "i" * 64

    _write_json(
        files["archived_history"],
        {"history": [{"epoch": 1}, {"epoch": 2}]},
    )
    _write_json(
        files["snapshot"],
        {
            "valid": True,
            "partial_training": {
                "experiment": "full_no_reconstruction",
                "completed_epochs": 2,
                "last_completed_epoch": 2,
                "best_checkpoint_epoch": 1,
                "config_fingerprint": fingerprint,
            },
            "input_state": {
                "effective_inventory_sha256": inventory,
            },
            "source_bindings": {
                "config": {"sha256": _sha256(files["config"])},
            },
        },
    )
    _write_json(
        files["addendum"],
        {
            "valid": True,
            "snapshot": {"sha256": _sha256(files["snapshot"])},
            "failed_resume": {
                "decision": "restart_same_formal_config_from_epoch_1",
                "last_checkpoint_zip_readable": False,
                "best_checkpoint_zip_readable": True,
                "optimizer_state_available_from_best": False,
            },
            "archive": {
                "files": {
                    "history": _binding(files["archived_history"]),
                    "best_checkpoint": _binding(files["archived_best"]),
                    "last_checkpoint": _binding(files["archived_last"]),
                }
            },
        },
    )
    _write_json(
        files["current_integrity"],
        {
            "valid": True,
            "summary": {
                "decode_failure_count": 0,
                "documented_recovery_count": 0,
                "raw_inventory_sha256": inventory,
                "effective_inventory_sha256": inventory,
            },
        },
    )
    _write_json(
        files["history"],
        {
            "config_fingerprint": fingerprint,
            "resumed_from": None,
            "history": [
                {"epoch": 1},
                {"epoch": 2},
                {"epoch": 3},
            ],
        },
    )
    _write_json(
        files["training_summary"],
        {
            "start_epoch": 0,
            "completed_epochs": 3,
            "checkpoints_saved": True,
            "resumed_from": None,
            "config_fingerprint": fingerprint,
        },
    )
    _write_json(
        files["checkpoint_audit"],
        {
            "valid": True,
            "history_json_rows": 3,
            "training_summary_completed_epochs": 3,
            "expected": {"config_fingerprint": fingerprint},
            "best": {
                "epoch": 2,
                "best_val_accuracy": 0.8,
                "sha256": _sha256(files["best_checkpoint"]),
            },
            "last": {
                "epoch": 3,
                "sha256": _sha256(files["last_checkpoint"]),
            },
        },
    )
    _write_json(
        files["evaluation"],
        {
            "metrics": {
                "sample_count": 3,
                "accuracy": 0.8,
                "macro_precision": 0.7,
                "macro_recall": 0.6,
                "macro_f1": 0.65,
                "weighted_f1": 0.78,
                "ece": 0.03,
                "negative_log_likelihood": 0.4,
                "multiclass_brier_score": 0.2,
            },
            "dataset": {
                "split": "test",
                "evaluated_samples": 3,
                "full_split_samples": 3,
                "sensor_degradation_enabled": False,
            },
            "checkpoint": {"epoch": 2},
            "evaluation_config": {"bootstrap_repeats": 2000},
        },
    )
    return files


def test_formal_queue_interruption_outcome_accepts_clean_restart(tmp_path):
    report = build_interruption_outcome(
        files=_fixture(tmp_path),
        expected_partial_epochs=2,
        expected_total_epochs=3,
        expected_test_samples=3,
    )

    assert report["valid"] is True
    assert report["restart"]["history_contiguous"] is True
    assert report["training"]["best_epoch"] == 2
    assert report["test"]["sample_count"] == 3


def test_formal_queue_interruption_outcome_rejects_warm_start(tmp_path):
    files = _fixture(tmp_path)
    history = json.loads(files["history"].read_text(encoding="utf-8"))
    history["resumed_from"] = "archived_best.pt"
    _write_json(files["history"], history)

    report = build_interruption_outcome(
        files=files,
        expected_partial_epochs=2,
        expected_total_epochs=3,
        expected_test_samples=3,
    )

    assert report["valid"] is False
    assert any("unexpectedly records a resume" in error for error in report["errors"])
