import hashlib
import json
from pathlib import Path
import zipfile

import pytest

import build_evidence_archive as evidence_archive_module
from build_evidence_archive import (
    CODE_FILES,
    PLAN_A_REQUIRED_FILES,
    SOURCE_FILES,
    build_evidence_archive,
)
from verify_evidence_archive import verify_evidence_archive


def _prepare_minimal_project(root: Path) -> Path:
    for relative in CODE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    for relative in SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    reports = root / "reports" / "qwen-material"
    reports.mkdir(parents=True)
    (reports / "PAPER_READINESS_AUDIT.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "internal_paper_evidence_ready": True,
            }
        ),
        encoding="utf-8",
    )
    for relative in (
        "FINAL_EVIDENCE_AUDIT.json",
        "FINAL_CLAIM_EVIDENCE.json",
        "FINAL_CLAIM_EVIDENCE.md",
        "SUBMISSION_INDEX.json",
        "SUBMISSION_INDEX.md",
        "paper-tables/manifest.json",
        "paper-tables/TABLES.md",
    ):
        path = reports / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (reports / "ignored.predictions.jsonl").write_text(
        "private row",
        encoding="utf-8",
    )
    (root / "source-patent.docx").write_bytes(b"private document")
    config = root / "configs" / "ablations" / "vision_only.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    test_file = root / "tests" / "test_smoke.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_smoke(): pass\n", encoding="utf-8")
    checkpoint_dir = root / "checkpoints" / "ablations" / "candidate"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "history.json").write_text(
        '{"history":[]}',
        encoding="utf-8",
    )
    (checkpoint_dir / "training_summary.json").write_text(
        '{"state":"complete"}',
        encoding="utf-8",
    )
    (checkpoint_dir / "best.pt").write_bytes(b"model weights")
    return reports


def test_evidence_archive_is_deterministic_and_excludes_jsonl(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    first = build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=tmp_path / "first",
    )
    second = build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=tmp_path / "second",
    )

    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["original_patent_included"] is False
    assert first["raw_data_included"] is False
    assert first["annotations_included"] is False
    assert first["model_checkpoints_included"] is False
    assert first["training_metadata_file_count"] == 2
    archive_path = tmp_path / "first" / "paper-evidence.zip"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == (
        first["archive_sha256"]
    )
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert "evidence-manifest.json" in names
    assert "project/configs/ablations/vision_only.json" in names
    assert "project/tests/test_smoke.py" in names
    assert (
        "project/checkpoints/ablations/candidate/history.json" in names
    )
    assert (
        "project/checkpoints/ablations/candidate/training_summary.json"
        in names
    )
    assert not any(name.endswith(".jsonl") for name in names)
    assert not any(name.endswith(".pt") for name in names)
    assert not any(name.endswith(".docx") for name in names)

    def fail_during_zip_write(*args, **kwargs):
        raise RuntimeError("simulated archive write interruption")

    monkeypatch.setattr(
        evidence_archive_module,
        "_write_deterministic_zip",
        fail_during_zip_write,
    )
    failed_output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated"):
        build_evidence_archive(
            project_root=project,
            reports_root=reports,
            output_dir=failed_output,
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


def test_evidence_archive_refuses_nonempty_output(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    output = tmp_path / "output"
    output.mkdir()
    (output / "unexpected").write_text("x", encoding="utf-8")

    try:
        build_evidence_archive(
            project_root=project,
            reports_root=reports,
            output_dir=output,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("non-empty output must not be overwritten")


def test_evidence_archive_refuses_secret_like_source(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    leaked = project / "tests" / "leaked.py"
    leaked.write_text(
        "TOKEN = " + repr("sk-" + "A" * 24),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="secret-like tokens"):
        build_evidence_archive(
            project_root=project,
            reports_root=reports,
            output_dir=tmp_path / "output",
        )


def test_evidence_archive_can_include_plan_a_release(tmp_path):
    project = tmp_path / "project"
    reports = _prepare_minimal_project(project)
    for relative in PLAN_A_REQUIRED_FILES:
        path = project / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(
                json.dumps({"state": "complete", "valid": True}),
                encoding="utf-8",
            )
        else:
            path.write_bytes(relative.encode("utf-8"))
    plan_a_index = reports / "PLAN_A_SUBMISSION_INDEX.json"
    plan_a_index.write_text(
        json.dumps(
            {
                "state": "complete",
                "valid": True,
                "gates": {
                    "plan_a_visual_qa_complete": True,
                    "paper_spine_final_gate_passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "PLAN_A_PAPER_READINESS_AUDIT.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "valid": True,
                "internal_paper_evidence_ready": True,
            }
        ),
        encoding="utf-8",
    )
    page = (
        project
        / "paper_rewriting_output"
        / "plan_a_visual_qa"
        / "pages"
        / "page-001.png"
    )
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_bytes(b"png")
    checkpoint = (
        project
        / "checkpoints"
        / "confirmatory-ablations"
        / "v3_loss_components"
        / "v3_no_grid_reconstruction"
        / "seed42"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "history.json").write_text(
        '{"history":[]}',
        encoding="utf-8",
    )
    (checkpoint / "training_summary.json").write_text(
        '{"state":"complete"}',
        encoding="utf-8",
    )
    (checkpoint / "best.pt").write_bytes(b"weights")
    interrupted = (
        project
        / "checkpoints"
        / "confirmatory-ablations"
        / "interrupted"
        / "failed-run"
    )
    interrupted.mkdir(parents=True, exist_ok=True)
    (interrupted / "history.json").write_text(
        '{"history":[{"epoch":1}]}',
        encoding="utf-8",
    )

    result = build_evidence_archive(
        project_root=project,
        reports_root=reports,
        output_dir=tmp_path / "plan-a",
        include_plan_a=True,
    )

    assert result["plan_a_release_included"] is True
    evidence_manifest = json.loads(
        (tmp_path / "plan-a" / "evidence-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    archived_paths = {
        item["path"] for item in evidence_manifest["files"]
    }
    assert (
        "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V12.json"
        in archived_paths
    )
    assert (
        "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V12.md"
        in archived_paths
    )
    assert (
        "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V13.json"
        in archived_paths
    )
    assert (
        "reports/qwen-material/MANUSCRIPT_RESULT_GOVERNANCE_V14.json"
        in archived_paths
    )
    assert (
        "reports/qwen-material/recent-robust-baselines/"
        "PROTOCOL_AMENDMENT.json"
        in archived_paths
    )
    paths = {entry["path"] for entry in evidence_manifest["files"]}
    assert evidence_manifest["schema_version"] == 2
    assert (
        "paper_rewriting_output/final_paper/paper.zh.docx" in paths
    )
    assert (
        "reports/qwen-material/PLAN_A_PAPER_READINESS_AUDIT.json"
        in paths
    )
    assert (
        "paper_rewriting_output/plan_a_visual_qa/pages/page-001.png"
        in paths
    )
    assert (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        "v3_no_grid_reconstruction/seed42/history.json"
        in paths
    )
    assert not any(path.endswith(".pt") for path in paths)
    assert not any("/interrupted/" in path for path in paths)
    assert verify_evidence_archive(tmp_path / "plan-a")["valid"] is True
