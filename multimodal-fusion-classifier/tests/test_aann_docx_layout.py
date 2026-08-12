from __future__ import annotations

import pytest

pytest.importorskip("docx")

from docx import Document
from docx.oxml.ns import qn

from paper_rewriting_output.aann2026_submission.style_ieee_docx import OUTPUT


def test_final_docx_keeps_text_below_last_full_width_figure() -> None:
    document = Document(OUTPUT)
    body_section = document._element.body.sectPr

    section_type = body_section.find(qn("w:type"))
    columns = body_section.find(qn("w:cols"))

    assert section_type is not None
    assert section_type.get(qn("w:val")) == "continuous"
    assert columns is not None
    assert columns.get(qn("w:num")) == "2"


def test_manual_headings_and_captions_do_not_inherit_template_numbers() -> None:
    document = Document(OUTPUT)

    for style_name in ("Heading 1", "Heading 2", "figure caption", "references"):
        style_properties = document.styles[style_name]._element.pPr
        assert style_properties.find(qn("w:numPr")) is None

    figure_three = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.startswith("Figure 3.")
    ]
    assert len(figure_three) == 1
