import hashlib
import json
from pathlib import Path

from archive_yolo_seg_dependency_incident import archive_incident_sources


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_incident_archive_mirrors_bound_sources(tmp_path, monkeypatch):
    source = tmp_path / "logs" / "failed.log"
    source.parent.mkdir()
    source.write_text("missing polars\n", encoding="utf-8")
    checkpoint_archive = tmp_path / "checkpoints" / "interrupted"
    checkpoint_archive.mkdir(parents=True)
    archived_result = checkpoint_archive / "results.csv"
    archived_result.write_text("epoch\n1\n", encoding="utf-8")
    incident = tmp_path / "incident.json"
    incident.write_text(
        json.dumps(
            {
                "valid": True,
                "source_bindings": {
                    "failed_log": {
                        "path": "logs/failed.log",
                        "size_bytes": source.stat().st_size,
                        "sha256": _sha256(source),
                    }
                },
                "archive_bindings": {
                    "results.csv": {
                        "path": "checkpoints/interrupted/results.csv",
                        "size_bytes": archived_result.stat().st_size,
                        "sha256": _sha256(archived_result),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "archive_yolo_seg_dependency_incident.PROJECT_ROOT", tmp_path
    )
    mirror = tmp_path / "mirror"

    report = archive_incident_sources(
        incident_path=incident,
        checkpoint_archive=checkpoint_archive,
        mirror_root=mirror,
    )

    assert report["valid"] is True
    assert report["source_mirror"]["file_count"] == 1
    mirrored = mirror / "sources" / "logs" / "failed.log"
    assert _sha256(mirrored) == _sha256(source)


def test_incident_archive_rejects_source_drift(tmp_path, monkeypatch):
    source = tmp_path / "source.log"
    source.write_text("changed\n", encoding="utf-8")
    checkpoint_archive = tmp_path / "archive"
    checkpoint_archive.mkdir()
    result = checkpoint_archive / "results.csv"
    result.write_text("epoch\n1\n", encoding="utf-8")
    incident = tmp_path / "incident.json"
    incident.write_text(
        json.dumps(
            {
                "valid": True,
                "source_bindings": {
                    "source": {
                        "path": "source.log",
                        "size_bytes": 3,
                        "sha256": hashlib.sha256(b"old").hexdigest(),
                    }
                },
                "archive_bindings": {
                    "results.csv": {
                        "sha256": _sha256(result),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "archive_yolo_seg_dependency_incident.PROJECT_ROOT", tmp_path
    )

    try:
        archive_incident_sources(
            incident_path=incident,
            checkpoint_archive=checkpoint_archive,
            mirror_root=tmp_path / "mirror",
        )
    except ValueError as exc:
        assert "bound source hash drift" in str(exc)
    else:
        raise AssertionError("source drift must be rejected")
