import json
from pathlib import Path

from capture_yolo_seg_dependency_incident import (
    REQUIRED_PACKAGES,
    build_dependency_incident,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    files = {
        name: tmp_path / f"{name}.txt"
        for name in (
            "formal_status",
            "formal_reporting_status",
            "yolo_status",
            "continuation_status",
            "yolo_stdout",
            "yolo_stderr",
            "continuation_yolo_stdout",
            "continuation_yolo_stderr",
            "run_manifest",
            "integrated_protocol",
            "stratified_protocol",
            "export_report",
            "requirements",
            "requirements_repro",
            "environment",
        )
    }
    _write_json(files["formal_status"], {"state": "complete"})
    _write_json(files["formal_reporting_status"], {"state": "complete"})
    _write_json(
        files["yolo_status"],
        {"state": "failed", "message": "YOLO-seg pipeline failed"},
    )
    _write_json(
        files["continuation_status"],
        {
            "state": "failed",
            "message": "yolo_seg_pipeline failed with exit code 1",
        },
    )
    files["yolo_stderr"].write_text(
        "save_model\nread_results_csv\n"
        "ModuleNotFoundError: No module named 'polars'\n",
        encoding="utf-8",
    )
    for name, path in files.items():
        if not path.exists():
            path.write_text(name, encoding="utf-8")

    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "args.yaml").write_text("epochs: 100\n", encoding="utf-8")
    (archive / "results.csv").write_text(
        "epoch,time\n1,10.0\n", encoding="utf-8"
    )
    return files, archive


def test_dependency_incident_accepts_archived_epoch_one_without_weights(
    tmp_path,
    monkeypatch,
):
    files, archive = _fixture(tmp_path)
    monkeypatch.setattr(
        "capture_yolo_seg_dependency_incident.PROJECT_ROOT", tmp_path
    )

    report = build_dependency_incident(
        files=files,
        archive_dir=archive,
        dependency_versions=REQUIRED_PACKAGES,
        ultralytics_requirements=[
            "polars>=0.20.0",
            "nvidia-ml-py>=12.0.0",
            "ultralytics-thop>=2.0.18",
        ],
    )

    assert report["valid"] is True
    assert report["failure"]["completed_epochs"] == 1
    assert report["failure"]["checkpoint_saved"] is False
    assert report["remediation"]["archived_file_count"] == 2


def test_dependency_incident_rejects_archived_weight(tmp_path, monkeypatch):
    files, archive = _fixture(tmp_path)
    weights = archive / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"not formal evidence")
    monkeypatch.setattr(
        "capture_yolo_seg_dependency_incident.PROJECT_ROOT", tmp_path
    )

    report = build_dependency_incident(
        files=files,
        archive_dir=archive,
        dependency_versions=REQUIRED_PACKAGES,
        ultralytics_requirements=[
            "polars>=0.20.0",
            "nvidia-ml-py>=12.0.0",
            "ultralytics-thop>=2.0.18",
        ],
    )

    assert report["valid"] is False
    assert any("unexpectedly contains model weights" in error for error in report["errors"])
