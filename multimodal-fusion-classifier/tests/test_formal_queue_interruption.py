import hashlib
import json
from pathlib import Path
import zipfile

from capture_formal_queue_interruption import build_interruption_snapshot
from capture_formal_queue_interruption_addendum import (
    build_interruption_addendum,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_interruption_snapshot_records_contiguous_partial_run(tmp_path):
    files = {
        name: tmp_path / name
        for name in (
            "formal_status",
            "recovery_chain_status",
            "history",
            "best_checkpoint",
            "last_checkpoint",
            "config",
            "train_manifest",
            "training_stdout",
            "training_stderr",
            "source_transition",
            "current_integrity",
        )
    }
    _write_json(
        files["history"],
        {
            "history": [{"epoch": epoch} for epoch in range(1, 69)],
            "best_checkpoint_epoch": 67,
            "best_val_accuracy": 0.9,
            "config_fingerprint": "fingerprint",
        },
    )
    _write_json(
        files["source_transition"],
        {"valid": True, "transition": {"cause": "unknown_external_state_change"}},
    )
    _write_json(
        files["current_integrity"],
        {
            "valid": True,
            "expected_recovery_state": "repaired",
            "summary": {"effective_inventory_sha256": "inventory"},
        },
    )
    files["training_stdout"].write_text(
        "[mmfc-image-recovery] active:\n", encoding="utf-8"
    )
    files["training_stderr"].write_text("", encoding="utf-8")
    for name, path in files.items():
        if not path.exists():
            path.write_text(name, encoding="utf-8")

    report = build_interruption_snapshot(files=files)

    assert report["valid"] is True
    assert report["partial_training"]["completed_epochs"] == 68
    assert report["partial_training"]["best_checkpoint_epoch"] == 67
    assert report["input_state"]["effective_image_bytes_unchanged_across_resume"] is True


def test_interruption_addendum_rejects_corrupt_last_and_keeps_archive(
    tmp_path,
):
    archive = tmp_path / "checkpoints" / "interrupted" / "run"
    archive.mkdir(parents=True)
    history = archive / "history.json"
    best = archive / "best.pt"
    last = archive / "last.pt"
    history.write_text("{}", encoding="utf-8")
    with zipfile.ZipFile(best, "w") as handle:
        handle.writestr("data", "ok")
    last.write_bytes(b"incomplete zip")
    snapshot = tmp_path / "snapshot.json"
    _write_json(
        snapshot,
        {
            "valid": True,
            "source_bindings": {
                "history": {"sha256": _sha256(history)},
                "best_checkpoint": {"sha256": _sha256(best)},
                "last_checkpoint": {"sha256": _sha256(last)},
            },
        },
    )
    status = tmp_path / "status.json"
    _write_json(
        status,
        {
            "state": "failed",
            "message": "full_no_reconstruction_resume failed with exit code 1",
        },
    )
    stderr = tmp_path / "stderr.log"
    stderr.write_text("failed finding central directory", encoding="utf-8")

    report = build_interruption_addendum(
        snapshot_path=snapshot,
        archive_dir=archive,
        failed_status_path=status,
        failed_stderr_path=stderr,
    )

    assert report["valid"] is True
    assert report["failed_resume"]["last_checkpoint_zip_readable"] is False
    assert report["failed_resume"]["best_checkpoint_zip_readable"] is True
    assert report["failed_resume"]["decision"] == (
        "restart_same_formal_config_from_epoch_1"
    )
