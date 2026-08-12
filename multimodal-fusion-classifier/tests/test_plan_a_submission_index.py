from __future__ import annotations

import hashlib
import json

import pytest

from build_plan_a_submission_index import (
    PAPER_FILES,
    PROJECT_FILES,
    REPORT_FILES,
    build_plan_a_submission_index,
    verify_plan_a_submission_index,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def _binding(path, root):
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _prepare(tmp_path):
    reports = tmp_path / "reports" / "qwen-material"
    paper = tmp_path / "paper_rewriting_output"
    for relative in REPORT_FILES.values():
        _write_json(
            reports / relative,
            {"state": "complete", "valid": True},
        )
    for relative in PAPER_FILES.values():
        path = paper / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            _write_json(path, {"state": "complete", "valid": True})
        else:
            path.write_text(relative, encoding="utf-8")
    for relative in PROJECT_FILES.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    final_path = reports / REPORT_FILES["base_final_evidence_audit"]
    readiness_path = reports / REPORT_FILES["paper_readiness_audit"]
    _write_json(
        readiness_path,
        {
            "state": "complete",
            "valid": True,
            "internal_paper_evidence_ready": True,
            "source_evidence_audit_sha256": _sha256(final_path),
        },
    )
    _write_json(
        reports / REPORT_FILES["final_claim_evidence"],
        {
            "state": "complete",
            "source_final_audit_sha256": _sha256(final_path),
        },
    )
    _write_json(
        reports / REPORT_FILES["manuscript_governance"],
        {
            "state": "complete",
            "valid": True,
            "decision": {"confirmed_option": "A"},
        },
    )
    _write_json(
        reports / REPORT_FILES["full_regression_audit"],
        {
            "state": "complete",
            "valid": True,
            "combined": {
                "all_collected_tests_passed": True,
                "unresolved_skips": 0,
            },
        },
    )
    _write_json(
        reports / REPORT_FILES["manuscript_queue_status"],
        {"state": "awaiting_visual_qa"},
    )
    _write_json(
        paper / PAPER_FILES["automated_build_audit"],
        {
            "state": "automated_build_complete_visual_qa_pending",
            "valid": True,
        },
    )
    pdf = paper / PAPER_FILES["paper_pdf"]
    alias = paper / PAPER_FILES["paper_pdf_alias"]
    pdf.write_bytes(b"pdf")
    alias.write_bytes(b"pdf")
    main = paper / PAPER_FILES["main_tex"]
    visual = paper / PAPER_FILES["visual_qa"]
    _write_json(
        visual,
        {
            "state": "complete",
            "valid": True,
            "visual_qa_complete": True,
            "issue_count": 0,
            "page_count": 12,
            "reviewer": "tester",
            "reviewed_at": "2026-07-29T12:00:00+08:00",
            "paper_pdf_sha256": _sha256(pdf),
            "source_bindings": {
                "paper_pdf": _binding(pdf, tmp_path),
                "main_tex": _binding(main, tmp_path),
            },
        },
    )
    (paper / PAPER_FILES["artifact_check"]).write_text(
        "Status: PASS",
        encoding="utf-8",
    )
    (paper / PAPER_FILES["progress"]).write_text(
        "Next stage: Complete",
        encoding="utf-8",
    )
    (paper / PAPER_FILES["integrity_audit"]).write_text(
        "Status: CLEAN",
        encoding="utf-8",
    )
    (paper / PAPER_FILES["word_report"]).write_text(
        "Status: PASS",
        encoding="utf-8",
    )
    return reports, paper, visual


def test_build_plan_a_submission_index_is_hash_bound(tmp_path):
    reports, paper, _ = _prepare(tmp_path)
    output = reports / "PLAN_A_SUBMISSION_INDEX.json"

    result = build_plan_a_submission_index(
        project_root=tmp_path,
        reports_root=reports,
        paper_root=paper,
        output_path=output,
    )

    assert result["valid"] is True
    assert result["gates"]["plan_a_visual_qa_complete"] is True
    assert result["gates"]["physical_hardware_validation_ready"] is False
    assert result["artifacts"]["paper_pdf"]["sha256"] == _sha256(
        paper / PAPER_FILES["paper_pdf"]
    )
    assert output.with_suffix(".md").is_file()
    assert verify_plan_a_submission_index(
        output,
        project_root=tmp_path,
    )["valid"] is True


def test_build_plan_a_submission_index_rejects_visual_drift(tmp_path):
    reports, paper, _ = _prepare(tmp_path)
    (paper / PAPER_FILES["main_tex"]).write_text(
        "changed",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="binding drift"):
        build_plan_a_submission_index(
            project_root=tmp_path,
            reports_root=reports,
            paper_root=paper,
            output_path=reports / "PLAN_A_SUBMISSION_INDEX.json",
        )


def test_verify_plan_a_submission_index_detects_late_drift(tmp_path):
    reports, paper, _ = _prepare(tmp_path)
    output = reports / "PLAN_A_SUBMISSION_INDEX.json"
    build_plan_a_submission_index(
        project_root=tmp_path,
        reports_root=reports,
        paper_root=paper,
        output_path=output,
    )
    (paper / PAPER_FILES["paper_pdf"]).write_bytes(b"late drift")

    report = verify_plan_a_submission_index(
        output,
        project_root=tmp_path,
    )

    assert report["valid"] is False
    assert any("paper_pdf" in error for error in report["errors"])
