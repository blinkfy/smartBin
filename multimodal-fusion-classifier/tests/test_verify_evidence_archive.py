from pathlib import Path
import zipfile

from build_evidence_archive import build_evidence_archive
from test_evidence_archive import _prepare_minimal_project
from verify_evidence_archive import verify_evidence_archive


def test_verify_evidence_archive_accepts_generated_bundle(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    output = tmp_path / "archive"
    build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=output,
    )

    result = verify_evidence_archive(output)

    assert result["valid"] is True
    assert result["error_count"] == 0


def test_verify_evidence_archive_detects_unmanifested_member(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    output = tmp_path / "archive"
    build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=output,
    )
    archive_path = output / "paper-evidence.zip"
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("project/unmanifested.txt", "unexpected")

    result = verify_evidence_archive(output)

    assert result["valid"] is False
    assert any(
        "unmanifested" in error or "SHA-256" in error
        for error in result["errors"]
    )


def test_verify_evidence_archive_detects_external_manifest_drift(
    tmp_path,
):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    output = tmp_path / "archive"
    build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=output,
    )
    manifest_path = output / "evidence-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    result = verify_evidence_archive(output)

    assert result["valid"] is False
    assert any("manifest" in error for error in result["errors"])


def test_verify_evidence_archive_reports_malformed_manifest(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    output = tmp_path / "archive"
    build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=output,
    )
    (output / "evidence-manifest.json").write_text(
        "{broken",
        encoding="utf-8",
    )

    result = verify_evidence_archive(output)

    assert result["valid"] is False
    assert any("parse" in error for error in result["errors"])


def test_verify_evidence_archive_rejects_drive_style_member(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    output = tmp_path / "archive"
    build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=output,
    )
    archive_path = output / "paper-evidence.zip"
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("C:/escape.txt", "unexpected")

    result = verify_evidence_archive(output)

    assert result["valid"] is False
    assert any("unsafe ZIP member path" in error for error in result["errors"])


def test_verify_evidence_archive_rejects_raw_data_path(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    output = tmp_path / "archive"
    build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=output,
    )
    archive_path = output / "paper-evidence.zip"
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("project/data/labels/object.txt", "annotation")

    result = verify_evidence_archive(output)

    assert result["valid"] is False
    assert any("raw data or annotation" in error for error in result["errors"])


def test_verify_evidence_archive_rejects_model_checkpoint_path(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    output = tmp_path / "archive"
    build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=output,
    )
    archive_path = output / "paper-evidence.zip"
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr(
            "project/checkpoints/ablations/candidate/best.pt",
            "weights",
        )

    result = verify_evidence_archive(output)

    assert result["valid"] is False
    assert any(
        "forbidden ZIP member" in error and "best.pt" in error
        for error in result["errors"]
    )
