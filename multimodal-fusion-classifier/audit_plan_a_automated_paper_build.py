"""Audit the unattended Plan A DOCX/PDF rebuild before visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from apply_plan_a_confirmatory_manuscript_update import (
    BEGIN_MARKER,
    END_MARKER,
    OLD_LIMITATION,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_ROOT = PROJECT_ROOT / "paper_rewriting_output"
FINAL_ROOT = PAPER_ROOT / "final_paper"
DEFAULT_MAIN = FINAL_ROOT / "main.tex"
DEFAULT_DOCX = FINAL_ROOT / "paper.zh.docx"
DEFAULT_PDF = FINAL_ROOT / "paper.pdf"
DEFAULT_PDF_ALIAS = FINAL_ROOT / "main.pdf"
DEFAULT_REVISION_AUDIT = (
    PAPER_ROOT / "plan_a_confirmatory_revision_audit.json"
)
DEFAULT_CONFIRMATORY_OUTCOME = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "plan-a-confirmatory-outcome"
    / "OUTCOME.json"
)
DEFAULT_OUTPUT = PAPER_ROOT / "plan_a_automated_build_audit.json"
REQUIRED_PHRASES = (
    "组件级归因、未见退化与直接基线",
    "QMF",
    "TMC",
    "单种子参考",
    "11种",
)
FORBIDDEN_PHRASES = (
    "训练范围外强度、未见故障以及availability错误尚未验证",
    "未直接实现最近邻鲁棒融合方法作为统一训练基线",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit unattended Plan A manuscript rebuild"
    )
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--pdf-alias", type=Path, default=DEFAULT_PDF_ALIAS)
    parser.add_argument(
        "--revision-audit",
        type=Path,
        default=DEFAULT_REVISION_AUDIT,
    )
    parser.add_argument(
        "--confirmatory-outcome",
        type=Path,
        default=DEFAULT_CONFIRMATORY_OUTCOME,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = resolved.as_posix()
    return {
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _docx_text(path: Path) -> tuple[str, int, int]:
    from docx import Document

    document = Document(path)
    parts = [paragraph.text for paragraph in document.paragraphs]
    table_paragraph_count = 0
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
                table_paragraph_count += len(cell.paragraphs)
    return (
        "\n".join(parts),
        len(document.paragraphs) + table_paragraph_count,
        len(document.tables),
    )


def _pdf_text(path: Path) -> tuple[str, int]:
    import fitz

    with fitz.open(path) as document:
        page_count = document.page_count
        text = "\n".join(page.get_text("text") for page in document)
    return text, page_count


def _text_contract(label: str, text: str) -> list[str]:
    errors = []
    compact_text = re.sub(r"\s+", "", text)
    for phrase in REQUIRED_PHRASES:
        if re.sub(r"\s+", "", phrase) not in compact_text:
            errors.append(f"{label} missing required phrase: {phrase}")
    for phrase in FORBIDDEN_PHRASES:
        if re.sub(r"\s+", "", phrase) in compact_text:
            errors.append(f"{label} retains obsolete phrase: {phrase}")
    if "??" in text:
        errors.append(f"{label} contains unresolved question-mark reference")
    if "\ufffd" in text:
        errors.append(f"{label} contains replacement characters")
    return errors


def audit_build(
    main_path: Path = DEFAULT_MAIN,
    docx_path: Path = DEFAULT_DOCX,
    pdf_path: Path = DEFAULT_PDF,
    pdf_alias_path: Path = DEFAULT_PDF_ALIAS,
    revision_audit_path: Path = DEFAULT_REVISION_AUDIT,
    confirmatory_outcome_path: Path = DEFAULT_CONFIRMATORY_OUTCOME,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    paths = {
        "main": _resolve(main_path),
        "docx": _resolve(docx_path),
        "pdf": _resolve(pdf_path),
        "pdf_alias": _resolve(pdf_alias_path),
        "revision_audit": _resolve(revision_audit_path),
        "confirmatory_outcome": _resolve(confirmatory_outcome_path),
    }
    output_path = _resolve(output_path)
    if output_path.exists() or output_path.with_suffix(".md").exists():
        raise FileExistsError("refusing to overwrite automated paper-build audit")
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "automated paper-build inputs missing: " + ", ".join(missing)
        )
    revision = _read_json(paths["revision_audit"])
    outcome = _read_json(paths["confirmatory_outcome"])
    errors = []
    if revision.get("valid") is not True or revision.get("state") != "complete":
        errors.append("canonical TeX revision audit is invalid")
    if outcome.get("valid") is not True or outcome.get("state") != "complete":
        errors.append("confirmatory outcome is invalid")

    main_text = paths["main"].read_text(encoding="utf-8")
    if (
        main_text.count(BEGIN_MARKER) != 1
        or main_text.count(END_MARKER) != 1
        or OLD_LIMITATION in main_text
        or r"\cite{han2021trusted}" not in main_text
    ):
        errors.append("canonical TeX marker or citation contract failed")
    errors.extend(_text_contract("TeX", main_text))

    docx_text, paragraph_count, table_count = _docx_text(paths["docx"])
    errors.extend(_text_contract("DOCX", docx_text))
    if paragraph_count < 200 or table_count < 8:
        errors.append("DOCX structure is unexpectedly small")

    pdf_text, page_count = _pdf_text(paths["pdf"])
    errors.extend(_text_contract("PDF", pdf_text))
    if not 10 <= page_count <= 30:
        errors.append(f"PDF page count outside expected range: {page_count}")
    if _sha256(paths["pdf"]) != _sha256(paths["pdf_alias"]):
        errors.append("paper.pdf and main.pdf are not byte-identical")

    report = {
        "schema_version": 1,
        "state": "automated_build_complete_visual_qa_pending",
        "valid": not errors,
        "scope": "plan_a_unattended_docx_pdf_content_audit",
        "visual_qa_complete": False,
        "submission_ready": False,
        "next_required_action": (
            "render and inspect every final DOCX/PDF page before release"
        ),
        "document": {
            "docx_paragraph_count": paragraph_count,
            "docx_table_count": table_count,
            "pdf_page_count": page_count,
        },
        "source_bindings": {
            name: _binding(path) for name, path in paths.items()
        },
        "errors": errors,
    }
    _atomic_text(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# 方案A无人值守论文重建审计",
        "",
        f"- 状态：{'PASS' if report['valid'] else 'FAIL'}。",
        f"- Word段落/表格：{paragraph_count}/{table_count}。",
        f"- PDF页数：{page_count}。",
        "- 逐页视觉复核：未完成。",
        "- 投稿就绪：否；须完成最终逐页视觉检查及PaperSpine总关卡。",
        "",
    ]
    if errors:
        lines.extend(["## 错误", "", *[f"- {error}" for error in errors], ""])
    _atomic_text(output_path.with_suffix(".md"), "\n".join(lines))
    return report


def main() -> None:
    args = parse_args()
    report = audit_build(
        args.main,
        args.docx,
        args.pdf,
        args.pdf_alias,
        args.revision_audit,
        args.confirmatory_outcome,
        args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["valid"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
