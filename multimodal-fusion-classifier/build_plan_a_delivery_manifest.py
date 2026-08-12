"""Refresh the canonical Plan A delivery report after visual approval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_ROOT = PROJECT_ROOT / "paper_rewriting_output"
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_VISUAL_QA = PAPER_ROOT / "plan_a_visual_qa.json"
DEFAULT_MANIFEST = PAPER_ROOT / "final_artifact_manifest.md"
DEFAULT_LATEX_REPORT = PAPER_ROOT / "latex_report.md"

CANONICAL_FILES = {
    "main_tex": "final_paper/main.tex",
    "bibliography": "final_paper/references.bib",
    "docx": "final_paper/paper.zh.docx",
    "pdf": "final_paper/paper.pdf",
    "pdf_alias": "final_paper/main.pdf",
    "word_report": "word_report.zh.md",
}

TRACEABILITY_FILES = (
    "paper_spine_config.json",
    "paper_spine_config.md",
    "source_inventory.md",
    "source_inventory.json",
    "source_map.md",
    "evidence_bank.md",
    "figure_asset_map.md",
    "claim_register.md",
    "citation_support_bank.md",
    "citation_bank_check.md",
    "citation_quality_audit.md",
    "confirmed_motivation.md",
    "confirmed_contribution.md",
    "section_blueprints.md",
    "writing_rationale_matrix.md",
    "results_validation.md",
    "results_validation_check.md",
    "original_logic_map.md",
    "rewrite_matrix.md",
    "logic_transfer_audit.md",
    "revision_audit.md",
    "structured_review.md",
    "reviewer_audit.md",
    "reviewer_audit_check.md",
    "integrity_audit.md",
    "artifact_check.md",
    "progress.md",
    "plan_a_confirmatory_revision_audit.json",
    "plan_a_automated_build_audit.json",
    "plan_a_automated_build_audit.md",
    "plan_a_visual_qa.json",
    "plan_a_visual_qa.md",
)

PRO_EXTRA_FILES = (
    "research_dossier.md",
    "exemplar_learning_dossier.md",
    "sota_gap_map.md",
    "motivation_options_after_research.md",
    "style_profile.md",
    "document_style_spec.md",
    "humanize_matrix.md",
    "humanize_report.md",
    "review_outputs/methods_reviewer.md",
    "review_outputs/contribution_reviewer.md",
    "review_outputs/clarity_reviewer.md",
    "citation_verification_en.md",
    "qa/patent_source.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the hash-bound Plan A delivery manifest"
    )
    parser.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    parser.add_argument("--visual-qa", type=Path, default=DEFAULT_VISUAL_QA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--latex-report",
        type=Path,
        default=DEFAULT_LATEX_REPORT,
    )
    return parser.parse_args()


def _resolve(path: Path, *, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (
        project_root / path
    ).resolve()


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


def _declared_binding_current(
    binding: Mapping[str, Any],
    *,
    project_root: Path,
) -> bool:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    path = _resolve(Path(raw_path), project_root=project_root)
    return (
        path.is_file()
        and path.stat().st_size == binding.get("size_bytes")
        and _sha256(path) == binding.get("sha256")
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _artifact_row(
    path: Path,
    *,
    paper_root: Path,
    purpose: str,
) -> str:
    relative = path.relative_to(paper_root).as_posix()
    return (
        f"| required | `{relative}` | {purpose} | "
        f"{path.stat().st_size:,} | `{_sha256(path).upper()}` |"
    )


def _require_files(root: Path, relatives: tuple[str, ...]) -> list[Path]:
    paths = [root / relative for relative in relatives]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Plan A delivery inputs missing: "
            + ", ".join(path.as_posix() for path in missing)
        )
    return paths


def _validate_visual_qa(
    visual: Mapping[str, Any],
    *,
    project_root: Path,
) -> None:
    if (
        visual.get("schema_version") != 1
        or visual.get("state") != "complete"
        or visual.get("valid") is not True
        or visual.get("visual_qa_complete") is not True
        or visual.get("issue_count") != 0
    ):
        raise ValueError("Plan A visual QA is not a complete clean pass")
    bindings = visual.get("source_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("Plan A visual-QA source bindings are missing")
    stale = [
        name
        for name, binding in bindings.items()
        if not isinstance(binding, Mapping)
        or not _declared_binding_current(
            binding,
            project_root=project_root,
        )
    ]
    if stale:
        raise ValueError(
            "Plan A visual-QA source binding drift: " + ", ".join(stale)
        )


def build_delivery_manifest(
    *,
    project_root: Path = PROJECT_ROOT,
    paper_root: Path = PAPER_ROOT,
    visual_qa_path: Path = DEFAULT_VISUAL_QA,
    manifest_path: Path = DEFAULT_MANIFEST,
    latex_report_path: Path = DEFAULT_LATEX_REPORT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    paper_root = _resolve(paper_root, project_root=project_root)
    visual_qa_path = _resolve(
        visual_qa_path,
        project_root=project_root,
    )
    manifest_path = _resolve(manifest_path, project_root=project_root)
    latex_report_path = _resolve(
        latex_report_path,
        project_root=project_root,
    )
    if manifest_path.parent != paper_root:
        raise ValueError("final artifact manifest must remain in paper root")
    if latex_report_path.parent != paper_root:
        raise ValueError("LaTeX report must remain in paper root")
    if not visual_qa_path.is_file():
        raise FileNotFoundError(visual_qa_path)
    visual = _read_json(visual_qa_path)
    _validate_visual_qa(visual, project_root=project_root)

    canonical = {
        name: paper_root / relative
        for name, relative in CANONICAL_FILES.items()
    }
    _require_files(paper_root, tuple(CANONICAL_FILES.values()))
    _require_files(paper_root, TRACEABILITY_FILES)
    _require_files(paper_root, PRO_EXTRA_FILES)
    helper_paths = _require_files(
        paper_root,
        (
            "final_paper/ieee.csl",
            "final_paper/style_docx.py",
        ),
    )
    if _sha256(canonical["pdf"]) != _sha256(canonical["pdf_alias"]):
        raise ValueError("paper.pdf and main.pdf are not byte-identical")
    if visual.get("paper_pdf_sha256") != _sha256(canonical["pdf"]):
        raise ValueError("visual QA is not bound to the canonical PDF")

    figure_paths = sorted(
        path
        for path in (paper_root / "final_paper" / "figures").iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"}
    )
    if not figure_paths:
        raise FileNotFoundError("final paper figure assets are missing")

    main_text = canonical["main_tex"].read_text(encoding="utf-8")
    section_count = len(
        re.findall(r"(?m)^\\section\{", main_text)
    )
    citation_count = len(
        re.findall(r"\\cite(?:t|p)?\{", main_text)
    )
    page_count = visual.get("page_count")
    reviewer = str(visual.get("reviewer", "")).strip()
    reviewed_at = str(visual.get("reviewed_at", "")).strip()

    canonical_rows = [
        _artifact_row(
            canonical["main_tex"],
            paper_root=paper_root,
            purpose="方案A规范可编辑稿源",
        ),
        _artifact_row(
            canonical["bibliography"],
            paper_root=paper_root,
            purpose="核验后的参考文献库",
        ),
        _artifact_row(
            canonical["docx"],
            paper_root=paper_root,
            purpose="经Word关卡检查的中文论文",
        ),
        _artifact_row(
            canonical["pdf"],
            paper_root=paper_root,
            purpose=f"经逐页复核的{page_count}页阅读版PDF",
        ),
        _artifact_row(
            canonical["pdf_alias"],
            paper_root=paper_root,
            purpose="与paper.pdf字节一致的交付别名",
        ),
        _artifact_row(
            canonical["word_report"],
            paper_root=paper_root,
            purpose="DOCX结构、字体与内容检查报告",
        ),
    ]
    trace_rows = [
        (
            f"| required | `{path.relative_to(paper_root).as_posix()}` | "
            "PaperSpine或方案A证据链产物 |"
        )
        for path in _require_files(paper_root, TRACEABILITY_FILES)
    ]
    pro_rows = [
        (
            f"| pro-extra | `{path.relative_to(paper_root).as_posix()}` | "
            "研究、写作或独立复核扩展材料 |"
        )
        for path in _require_files(paper_root, PRO_EXTRA_FILES)
    ]
    figure_rows = [
        (
            f"| required | `{path.relative_to(paper_root).as_posix()}` | "
            f"{path.stat().st_size:,} | `{_sha256(path).upper()}` |"
        )
        for path in figure_paths
    ]
    helper_rows = [
        (
            f"| required | `{path.relative_to(paper_root).as_posix()}` | "
            "可复现的文献样式或Word排版辅助程序 |"
        )
        for path in helper_paths
    ]
    manifest = "\n".join(
        [
            "# Final Artifact Manifest",
            "",
            "## Canonical Deliverables",
            "",
            "| Category | Artifact | Purpose | Size (bytes) | SHA-256 |",
            "|---|---|---|---:|---|",
            *canonical_rows,
            "",
            "## Required Traceability",
            "",
            "| Category | Artifact | Purpose |",
            "|---|---|---|",
            *trace_rows,
            (
                "| required | `final_artifact_manifest.md` | "
                "本规范交付清单 |"
            ),
            "",
            "## Plan A Visual Approval",
            "",
            f"- Reviewer: `{reviewer}`",
            f"- Reviewed at: `{reviewed_at}`",
            f"- Page count: `{page_count}`",
            f"- Recorded issue count: `{visual.get('issue_count')}`",
            f"- Canonical PDF SHA-256: `{_sha256(canonical['pdf']).upper()}`",
            "",
            "## Pro-Extra Analysis",
            "",
            "| Category | Artifact | Purpose |",
            "|---|---|---|",
            *pro_rows,
            "",
            "## Figure Assets",
            "",
            "| Category | Artifact | Size (bytes) | SHA-256 |",
            "|---|---|---:|---|",
            *figure_rows,
            "",
            "## Reproduction Helpers",
            "",
            "| Category | Artifact | Purpose |",
            "|---|---|---|",
            *helper_rows,
            "",
            "## Optional Packages",
            "",
            (
                "- `optional-translation`: not requested; the canonical "
                "manuscript is already Chinese."
            ),
            (
                "- `optional-submission`: not generated because a specific "
                "journal template has not been selected."
            ),
            (
                "- `optional-review-response`: not generated because there "
                "is no external reviewer letter."
            ),
            "",
            "## Canonicality Notes",
            "",
            (
                "`main.tex` is the canonical editable source, "
                "`paper.zh.docx` is the canonical office document, and "
                "`paper.pdf` is the canonical reading copy. The Plan A "
                "visual-QA record is hash-bound to all three."
            ),
            "",
        ]
    )
    latex_report = "\n".join(
        [
            "# LaTeX and Delivery Report",
            "",
            "- Status: **PASS**",
            "- Output language: Chinese",
            "- Manuscript source: `final_paper/main.tex`",
            "- Bibliography: `final_paper/references.bib`",
            "- Final Word: `final_paper/paper.zh.docx`",
            "- Final PDF: `final_paper/paper.pdf`",
            "",
            "## Source Validation",
            "",
            (
                f"The canonical source contains {section_count} numbered "
                f"top-level sections and {citation_count} citation commands. "
                "The post-visual PaperSpine final gate must be rerun after "
                "this delivery report is refreshed."
            ),
            "",
            "## Build Chain",
            "",
            (
                "The confirmatory queue rebuilt the Chinese DOCX from the "
                "guarded TeX source with Pandoc, applied the local Word style "
                "processor, ran `word_guard.py`, and exported the PDF through "
                "Microsoft Word."
            ),
            "",
            "## PDF and Visual QA",
            "",
            (
                f"- `{canonical['pdf'].name}`: {page_count} pages, "
                f"{canonical['pdf'].stat().st_size:,} bytes."
            ),
            "- `main.pdf` is byte-identical to `paper.pdf`.",
            f"- Reviewer: `{reviewer}`.",
            f"- Review time: `{reviewed_at}`.",
            (
                "- Every hash-bound rendered page was recorded as reviewed "
                "with zero layout issues（逐页复核通过）."
            ),
            "",
            "## Claim Boundary",
            "",
            (
                "This delivery audit covers the current Plan A internal "
                "paper and its rendered layout. It does not establish "
                "physical-sensor, RGB-D, external-blind, deployment, or "
                "complete-patent validation."
            ),
            "",
        ]
    )
    _atomic_text(manifest_path, manifest)
    _atomic_text(latex_report_path, latex_report)
    return {
        "state": "complete",
        "valid": True,
        "page_count": page_count,
        "main_tex_sha256": _sha256(canonical["main_tex"]),
        "docx_sha256": _sha256(canonical["docx"]),
        "pdf_sha256": _sha256(canonical["pdf"]),
        "manifest": manifest_path.as_posix(),
        "latex_report": latex_report_path.as_posix(),
    }


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    report = build_delivery_manifest(
        project_root=project_root,
        paper_root=args.paper_root,
        visual_qa_path=args.visual_qa,
        manifest_path=args.manifest,
        latex_report_path=args.latex_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
