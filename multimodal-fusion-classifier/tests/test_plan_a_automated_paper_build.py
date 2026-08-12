from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from apply_plan_a_confirmatory_manuscript_update import (
    BEGIN_MARKER,
    END_MARKER,
)
from audit_plan_a_automated_paper_build import (
    FORBIDDEN_PHRASES,
    REQUIRED_PHRASES,
    _text_contract,
    audit_build,
)


def test_text_contract_accepts_confirmatory_content():
    text = (
        " ".join(REQUIRED_PHRASES).replace("11种", "11 \n 种")
        + " 正式结果。"
    )

    assert _text_contract("fixture", text) == []


def test_text_contract_rejects_obsolete_and_broken_content():
    text = " ".join(REQUIRED_PHRASES) + " " + FORBIDDEN_PHRASES[0] + " ?? \ufffd"

    errors = _text_contract("fixture", text)

    assert any("obsolete" in error for error in errors)
    assert any("question-mark" in error for error in errors)
    assert any("replacement" in error for error in errors)


def test_full_machine_audit_reads_docx_tables_and_chinese_pdf(tmp_path):
    docx_module = pytest.importorskip("docx")
    fitz = pytest.importorskip("fitz")
    style_path = (
        Path(__file__).resolve().parents[1]
        / "paper_rewriting_output"
        / "final_paper"
        / "style_docx.py"
    )
    spec = importlib.util.spec_from_file_location("paper_style_docx", style_path)
    assert spec is not None and spec.loader is not None
    style_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style_module)

    class _Row:
        cells = [object()] * 5

    class _PandocTable:
        columns = []
        rows = [_Row()]

    fallback_widths = style_module.equal_table_widths(_PandocTable())
    assert len(fallback_widths) == 5
    assert sum(fallback_widths) == style_module.TOTAL_TABLE_WIDTH

    main_path = tmp_path / "main.tex"
    docx_path = tmp_path / "paper.zh.docx"
    pdf_path = tmp_path / "paper.pdf"
    alias_path = tmp_path / "main.pdf"
    revision_path = tmp_path / "revision.json"
    outcome_path = tmp_path / "outcome.json"
    output_path = tmp_path / "audit.json"
    required_text = " ".join(REQUIRED_PHRASES)
    main_path.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\begin{document}",
                BEGIN_MARKER,
                required_text,
                r"\cite{han2021trusted}",
                END_MARKER,
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )

    document = docx_module.Document()
    document.add_paragraph(required_text)
    for index in range(205):
        document.add_paragraph(f"审计段落 {index}")
    for index in range(8):
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = f"表 {index}"
        table.cell(0, 1).text = "结果"
        table.cell(1, 0).text = "1"
        table.cell(1, 1).text = "2"
    document.save(docx_path)

    pdf = fitz.open()
    for index in range(10):
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            required_text if index == 0 else f"审计页面 {index + 1}",
            fontname="china-s",
            fontsize=10,
        )
    pdf.save(pdf_path)
    pdf.close()
    alias_path.write_bytes(pdf_path.read_bytes())
    revision_path.write_text(
        json.dumps({"valid": True, "state": "complete"}),
        encoding="utf-8",
    )
    outcome_path.write_text(
        json.dumps({"valid": True, "state": "complete"}),
        encoding="utf-8",
    )

    report = audit_build(
        main_path,
        docx_path,
        pdf_path,
        alias_path,
        revision_path,
        outcome_path,
        output_path,
    )

    assert report["valid"] is True
    assert report["visual_qa_complete"] is False
    assert report["submission_ready"] is False
    assert report["document"]["docx_table_count"] == 8
    assert report["document"]["pdf_page_count"] == 10
