from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

fitz = pytest.importorskip("fitz")

from complete_plan_a_visual_qa import complete_visual_qa
from prepare_plan_a_visual_qa import prepare_visual_qa


def _inputs(tmp_path):
    pdf = tmp_path / "paper.pdf"
    document = fitz.open()
    for text in ("第一页", "第二页"):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=12)
    document.save(pdf)
    document.close()
    docx = tmp_path / "paper.docx"
    docx.write_bytes(b"docx-fixture")
    main = tmp_path / "main.tex"
    main.write_text("\\documentclass{article}", encoding="utf-8")
    automated = tmp_path / "automated.json"
    automated.write_text(
        json.dumps(
            {
                "state": "automated_build_complete_visual_qa_pending",
                "valid": True,
            }
        ),
        encoding="utf-8",
    )
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps({"state": "awaiting_visual_qa"}),
        encoding="utf-8",
    )
    return pdf, docx, main, automated, status


def _completed_decision(template_path, decision_path):
    decision = json.loads(template_path.read_text(encoding="utf-8"))
    decision["state"] = "complete"
    decision["reviewer"] = "codex-page-review"
    decision["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    decision["overall_pass"] = True
    for page in decision["pages"]:
        page["reviewed"] = True
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False),
        encoding="utf-8",
    )


def test_prepares_and_completes_hash_bound_page_review(tmp_path):
    pdf, docx, main, automated, status = _inputs(tmp_path)
    bundle = tmp_path / "visual"

    manifest = prepare_visual_qa(
        pdf,
        docx,
        main,
        automated,
        status,
        bundle,
        dpi=120,
    )

    assert manifest["visual_qa_complete"] is False
    assert manifest["page_count"] == 2
    assert len(list((bundle / "pages").glob("page-*.png"))) == 2
    decision = bundle / "REVIEW_DECISION.json"
    _completed_decision(
        bundle / "REVIEW_DECISION.template.json",
        decision,
    )
    output = tmp_path / "plan_a_visual_qa.json"
    report = complete_visual_qa(
        bundle / "REVIEW_MANIFEST.json",
        decision,
        output,
    )
    assert report["valid"] is True
    assert report["visual_qa_complete"] is True
    assert report["page_count"] == 2
    assert report["issue_count"] == 0
    assert output.with_suffix(".md").is_file()


def test_rejects_unreviewed_or_changed_page(tmp_path):
    pdf, docx, main, automated, status = _inputs(tmp_path)
    bundle = tmp_path / "visual"
    prepare_visual_qa(
        pdf,
        docx,
        main,
        automated,
        status,
        bundle,
    )
    decision = bundle / "REVIEW_DECISION.json"
    _completed_decision(
        bundle / "REVIEW_DECISION.template.json",
        decision,
    )
    value = json.loads(decision.read_text(encoding="utf-8"))
    value["pages"][0]["reviewed"] = False
    decision.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="page 1"):
        complete_visual_qa(
            bundle / "REVIEW_MANIFEST.json",
            decision,
            tmp_path / "audit.json",
        )

    _completed_decision(
        bundle / "REVIEW_DECISION.template.json",
        decision,
    )
    (bundle / "pages" / "page-001.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="page 1"):
        complete_visual_qa(
            bundle / "REVIEW_MANIFEST.json",
            decision,
            tmp_path / "audit.json",
        )


def test_rejects_delivery_changed_after_render(tmp_path):
    pdf, docx, main, automated, status = _inputs(tmp_path)
    bundle = tmp_path / "visual"
    prepare_visual_qa(
        pdf,
        docx,
        main,
        automated,
        status,
        bundle,
    )
    decision = bundle / "REVIEW_DECISION.json"
    _completed_decision(
        bundle / "REVIEW_DECISION.template.json",
        decision,
    )
    main.write_text("\\documentclass{book}", encoding="utf-8")

    with pytest.raises(ValueError, match="upstream bindings changed: main"):
        complete_visual_qa(
            bundle / "REVIEW_MANIFEST.json",
            decision,
            tmp_path / "audit.json",
        )


def test_prepare_requires_automated_pending_visual_state(tmp_path):
    pdf, docx, main, automated, status = _inputs(tmp_path)
    status.write_text(json.dumps({"state": "waiting"}), encoding="utf-8")

    with pytest.raises(ValueError, match="awaiting_visual_qa"):
        prepare_visual_qa(
            pdf,
            docx,
            main,
            automated,
            status,
            tmp_path / "visual",
        )
