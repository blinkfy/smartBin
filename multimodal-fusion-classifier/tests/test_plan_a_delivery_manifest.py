from __future__ import annotations

import hashlib
import json

import pytest

from build_plan_a_delivery_manifest import (
    CANONICAL_FILES,
    PRO_EXTRA_FILES,
    TRACEABILITY_FILES,
    build_delivery_manifest,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path, root):
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _prepare(tmp_path):
    paper = tmp_path / "paper_rewriting_output"
    for relative in (
        *CANONICAL_FILES.values(),
        *TRACEABILITY_FILES,
        *PRO_EXTRA_FILES,
        "final_paper/ieee.csl",
        "final_paper/style_docx.py",
        "final_paper/figures/figure.png",
    ):
        path = paper / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    main = paper / CANONICAL_FILES["main_tex"]
    main.write_text(
        "\\section{A}\nText \\cite{x}.\n\\section{B}\n",
        encoding="utf-8",
    )
    pdf = paper / CANONICAL_FILES["pdf"]
    alias = paper / CANONICAL_FILES["pdf_alias"]
    pdf.write_bytes(b"pdf")
    alias.write_bytes(b"pdf")
    visual = paper / "plan_a_visual_qa.json"
    visual.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "complete",
                "valid": True,
                "visual_qa_complete": True,
                "reviewer": "tester",
                "reviewed_at": "2026-07-29T12:00:00+08:00",
                "page_count": 12,
                "issue_count": 0,
                "paper_pdf_sha256": _sha256(pdf),
                "source_bindings": {
                    "paper_pdf": _binding(pdf, tmp_path),
                    "main_tex": _binding(main, tmp_path),
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paper, visual


def test_build_delivery_manifest_refreshes_hashes(tmp_path):
    paper, visual = _prepare(tmp_path)

    result = build_delivery_manifest(
        project_root=tmp_path,
        paper_root=paper,
        visual_qa_path=visual,
        manifest_path=paper / "final_artifact_manifest.md",
        latex_report_path=paper / "latex_report.md",
    )

    manifest = (paper / "final_artifact_manifest.md").read_text(
        encoding="utf-8"
    )
    report = (paper / "latex_report.md").read_text(encoding="utf-8")
    assert result["valid"] is True
    assert _sha256(paper / CANONICAL_FILES["pdf"]).upper() in manifest
    assert "Plan A Visual Approval" in manifest
    assert "逐页复核" in report
    assert "physical-sensor" in report


def test_build_delivery_manifest_rejects_visual_binding_drift(tmp_path):
    paper, visual = _prepare(tmp_path)
    (paper / CANONICAL_FILES["main_tex"]).write_text(
        "drift",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="binding drift"):
        build_delivery_manifest(
            project_root=tmp_path,
            paper_root=paper,
            visual_qa_path=visual,
            manifest_path=paper / "final_artifact_manifest.md",
            latex_report_path=paper / "latex_report.md",
        )
