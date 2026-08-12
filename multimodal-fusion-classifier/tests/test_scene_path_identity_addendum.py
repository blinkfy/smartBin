import hashlib
import json
from pathlib import Path

import pytest

from build_scene_path_identity_addendum import build_scene_path_addendum


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_scene_path_addendum_preserves_analysis_contract(tmp_path, monkeypatch):
    files = {name: tmp_path / name for name in (
        "original_protocol",
        "original_evaluation_source",
        "revised_evaluation_source",
        "recovery_queue_source",
        "frozen_queue_source",
        "failed_integrated_stdout",
        "failed_integrated_stderr",
        "failed_yolo_status",
        "failed_continuation_status",
        "checkpoint_audit",
        "detector_evaluation",
        "requirements_repro",
        "environment",
    )}
    files["original_evaluation_source"].write_text(
        "old source", encoding="utf-8"
    )
    files["revised_evaluation_source"].write_text(
        "_result_path_matches\nimage{result_index}.jpg\n"
        "validate_protocol_addendum\n",
        encoding="utf-8",
    )
    files["frozen_queue_source"].write_text("queue", encoding="utf-8")
    files["recovery_queue_source"].write_text(
        'evaluate_scene_pipeline.py\n--confidence-thresholds\n'
        '"0.25", "0.8"\n"--bootstrap-repeats", "2000"\n',
        encoding="utf-8",
    )
    original_hash = hashlib.sha256(b"old source").hexdigest()
    queue_hash = hashlib.sha256(b"queue").hexdigest()
    _write_json(
        files["original_protocol"],
        {
            "status": "frozen_before_detector_training",
            "inputs": {
                "evaluation_source": {"sha256": original_hash},
                "queue_source": {"sha256": queue_hash},
            },
        },
    )
    files["failed_integrated_stderr"].write_text(
        "detector result order/path differs from test manifest: image0.jpg",
        encoding="utf-8",
    )
    _write_json(
        files["failed_yolo_status"],
        {"state": "failed", "message": "integrated scene evaluation failed"},
    )
    _write_json(
        files["failed_continuation_status"],
        {"state": "failed", "message": "yolo_seg_pipeline failed"},
    )
    _write_json(files["checkpoint_audit"], {"completed_epochs": 100})
    _write_json(
        files["detector_evaluation"],
        {
            "protocol": "held_out_source_grouped_test",
            "dataset": {"test_source_images": 570, "test_objects": 825},
        },
    )
    for path in files.values():
        if not path.exists():
            path.write_text(path.name, encoding="utf-8")
    monkeypatch.setattr(
        "build_scene_path_identity_addendum.PROJECT_ROOT", tmp_path
    )

    report = build_scene_path_addendum(
        files=files,
        integrated_output=tmp_path / "integrated",
    )

    assert report["valid"] is True
    assert report["invariants"]["confidence_thresholds"] == [0.25, 0.8]
    assert report["invariants"]["metric_definitions_unchanged"] is True


def test_scene_path_addendum_rejects_missing_source_map(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "build_scene_path_identity_addendum.PROJECT_ROOT", tmp_path
    )

    with pytest.raises(FileNotFoundError, match="source map is incomplete"):
        build_scene_path_addendum(
            files={},
            integrated_output=tmp_path / "integrated",
        )
