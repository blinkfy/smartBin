import hashlib
import json
from pathlib import Path

from audit_yolo_seg_dependency_outcome import build_yolo_dependency_outcome


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _files(tmp_path: Path) -> dict[str, Path]:
    names = (
        "incident",
        "incident_addendum",
        "yolo_status",
        "run_manifest",
        "integrated_protocol",
        "export_report",
        "args",
        "results",
        "best_checkpoint",
        "last_checkpoint",
        "checkpoint_audit",
        "detector_evaluation",
        "integrated_evaluation",
        "integrated_protocol_addendum",
        "training_stdout",
        "training_stderr",
        "integrated_stdout",
        "integrated_stderr",
        "path_identity_failed_stderr",
        "requirements_repro",
        "environment",
        "queue_source",
        "current_integrity",
    )
    files = {name: tmp_path / name for name in names}
    files["best_checkpoint"].write_bytes(b"best")
    files["last_checkpoint"].write_bytes(b"last")
    files["queue_source"].write_text("queue", encoding="utf-8")
    best_sha = hashlib.sha256(b"best").hexdigest()
    last_sha = hashlib.sha256(b"last").hexdigest()
    _write_json(
        files["incident"],
        {
            "valid": True,
            "failure": {"completed_epochs": 1, "checkpoint_saved": False},
            "remediation": {
                "decision": "clean_restart_same_frozen_protocol_from_epoch_1"
            },
        },
    )
    _write_json(files["incident_addendum"], {"valid": True})
    _write_json(files["yolo_status"], {"state": "complete", "stage": "all"})
    _write_json(
        files["run_manifest"],
        {
            "protocol": "independent_yolo11_segmentation_training",
            "training": {
                "epochs": 100,
                "batch": 16,
                "imgsz": 640,
                "seed": 42,
            },
        },
    )
    _write_json(
        files["integrated_protocol"],
        {
            "inputs": {
                "queue_source": {
                    "sha256": hashlib.sha256(b"queue").hexdigest()
                }
            }
        },
    )
    _write_json(
        files["export_report"],
        {
            "valid": True,
            "source_image_count": 5990,
            "object_count": 8646,
            "copied_image_count": 0,
        },
    )
    files["args"].write_text(
        "epochs: 100\nbatch: 16\nimgsz: 640\nseed: 42\nresume: false\n",
        encoding="utf-8",
    )
    files["results"].write_text(
        "epoch,time\n"
        + "".join(f"{epoch},{epoch}.0\n" for epoch in range(1, 101)),
        encoding="utf-8",
    )
    _write_json(
        files["checkpoint_audit"],
        {
            "completed_epochs": 100,
            "best": {"sha256": best_sha},
            "last": {"sha256": last_sha},
        },
    )
    _write_json(
        files["detector_evaluation"],
        {
            "protocol": "held_out_source_grouped_test",
            "dataset": {
                "test_source_images": 570,
                "test_objects": 825,
                "class_count": 19,
            },
            "checkpoint": {"sha256": best_sha},
            "metrics": {},
        },
    )
    _write_json(
        files["integrated_protocol_addendum"],
        {
            "status": (
                "frozen_after_path_identity_failure_before_recovery_rerun"
            ),
            "valid": True,
        },
    )
    integrated_addendum_sha = hashlib.sha256(
        files["integrated_protocol_addendum"].read_bytes()
    ).hexdigest()
    _write_json(
        files["integrated_evaluation"],
        {
            "protocol": "fixed_source_grouped_scene_test",
            "frozen_protocol": {
                "addendum": {"sha256": integrated_addendum_sha}
            },
            "dataset": {
                "source_image_count": 570,
                "ground_truth_object_count": 825,
            },
            "detector": {"sha256": best_sha},
            "scenarios": {"0.25": {}, "0.8": {}},
        },
    )
    files["training_stderr"].write_text("", encoding="utf-8")
    files["integrated_stderr"].write_text("", encoding="utf-8")
    files["path_identity_failed_stderr"].write_text(
        "detector result order/path differs from test manifest: image0.jpg",
        encoding="utf-8",
    )
    files["requirements_repro"].write_text(
        "polars==1.36.1\n"
        "nvidia-ml-py==13.610.43\n"
        "ultralytics-thop==2.0.18\n",
        encoding="utf-8",
    )
    _write_json(
        files["current_integrity"],
        {"valid": True, "expected_recovery_state": "repaired"},
    )
    for path in files.values():
        if not path.exists():
            path.write_text(path.name, encoding="utf-8")
    return files


def test_yolo_dependency_outcome_accepts_clean_contiguous_restart(
    tmp_path, monkeypatch
):
    files = _files(tmp_path)
    monkeypatch.setattr(
        "audit_yolo_seg_dependency_outcome.PROJECT_ROOT", tmp_path
    )

    report = build_yolo_dependency_outcome(files=files)

    assert report["valid"] is True
    assert report["restart"]["formal_completed_epochs"] == 100
    assert report["restart"]["formal_epoch_sequence"] == [1, 100]
    assert report["restart"]["resume"] is False


def test_yolo_dependency_outcome_rejects_noncontiguous_epochs(
    tmp_path, monkeypatch
):
    files = _files(tmp_path)
    files["results"].write_text(
        "epoch,time\n1,1.0\n3,3.0\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "audit_yolo_seg_dependency_outcome.PROJECT_ROOT", tmp_path
    )

    report = build_yolo_dependency_outcome(files=files)

    assert report["valid"] is False
    assert any("not contiguous" in error for error in report["errors"])
