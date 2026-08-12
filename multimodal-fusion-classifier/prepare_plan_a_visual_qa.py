"""Render the Plan A delivery PDF into a hash-bound visual review bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_ROOT = PROJECT_ROOT / "paper_rewriting_output"
FINAL_PAPER = PAPER_ROOT / "final_paper"
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_PDF = FINAL_PAPER / "paper.pdf"
DEFAULT_DOCX = FINAL_PAPER / "paper.zh.docx"
DEFAULT_MAIN = FINAL_PAPER / "main.tex"
DEFAULT_AUTOMATED_AUDIT = PAPER_ROOT / "plan_a_automated_build_audit.json"
DEFAULT_STATUS = (
    REPORT_ROOT / "training" / "plan_a_manuscript_refresh_status.json"
)
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "plan_a_visual_qa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare hash-bound page renders for Plan A visual QA"
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument(
        "--automated-audit",
        type=Path,
        default=DEFAULT_AUTOMATED_AUDIT,
    )
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=144)
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
        display = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display = resolved.as_posix()
    return {
        "path": display,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def prepare_visual_qa(
    pdf_path: Path = DEFAULT_PDF,
    docx_path: Path = DEFAULT_DOCX,
    main_path: Path = DEFAULT_MAIN,
    automated_audit_path: Path = DEFAULT_AUTOMATED_AUDIT,
    status_path: Path = DEFAULT_STATUS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    dpi: int = 144,
) -> dict[str, Any]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required; use the base Anaconda document environment"
        ) from exc
    if dpi < 96 or dpi > 300:
        raise ValueError("dpi must be between 96 and 300")
    paths = {
        "pdf": _resolve(pdf_path),
        "docx": _resolve(docx_path),
        "main": _resolve(main_path),
        "automated_audit": _resolve(automated_audit_path),
        "status": _resolve(status_path),
    }
    output_dir = _resolve(output_dir)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Plan A visual-QA inputs missing: " + ", ".join(missing)
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite visual-QA bundle: {output_dir}"
        )
    automated = _read_json(paths["automated_audit"])
    status = _read_json(paths["status"])
    if (
        automated.get("valid") is not True
        or automated.get("state")
        != "automated_build_complete_visual_qa_pending"
    ):
        raise ValueError("automated paper-build audit is not ready for visual QA")
    if status.get("state") != "awaiting_visual_qa":
        raise ValueError("manuscript refresh status is not awaiting_visual_qa")

    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    page_rows = []
    with fitz.open(paths["pdf"]) as document:
        if document.page_count <= 0:
            raise ValueError("delivery PDF contains no pages")
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )
            page_path = pages_dir / f"page-{index:03d}.png"
            pixmap.save(page_path)
            page_rows.append(
                {
                    "page_number": index,
                    "png_path": page_path.relative_to(output_dir).as_posix(),
                    "png_sha256": _sha256(page_path),
                    "width_px": pixmap.width,
                    "height_px": pixmap.height,
                    "reviewed": False,
                    "issues": [],
                }
            )
    manifest = {
        "schema_version": 1,
        "state": "awaiting_page_by_page_review",
        "valid": True,
        "visual_qa_complete": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "render_backend": "PyMuPDF",
        "dpi": dpi,
        "page_count": len(page_rows),
        "pages": page_rows,
        "source_bindings": {
            **{name: _binding(path) for name, path in paths.items()},
            "preparer_source": _binding(Path(__file__)),
        },
        "review_contract": {
            "inspect_every_page_at_100_percent": True,
            "required_checks": [
                "no clipped or overlapping text",
                "no missing glyphs or broken equations",
                "tables and figures remain inside page bounds",
                "captions stay associated with their figures or tables",
                "headers, footers, page numbers, and references are coherent",
                "no unintended blank pages or severe page-break gaps",
            ],
            "machine_render_is_not_visual_approval": True,
        },
    }
    _atomic_text(
        output_dir / "REVIEW_MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    decision = {
        "schema_version": 1,
        "state": "pending_visual_review",
        "reviewer": "",
        "reviewed_at": "",
        "overall_pass": False,
        "paper_pdf_sha256": manifest["source_bindings"]["pdf"]["sha256"],
        "page_count": len(page_rows),
        "pages": [
            {
                "page_number": row["page_number"],
                "png_sha256": row["png_sha256"],
                "reviewed": False,
                "issues": [],
            }
            for row in page_rows
        ],
        "notes": (
            "Fill only after every rendered page has been visually inspected."
        ),
    }
    _atomic_text(
        output_dir / "REVIEW_DECISION.template.json",
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def main() -> None:
    args = parse_args()
    report = prepare_visual_qa(
        args.pdf,
        args.docx,
        args.main,
        args.automated_audit,
        args.status,
        args.output_dir,
        dpi=args.dpi,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "visual_qa_complete": report["visual_qa_complete"],
                "page_count": report["page_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
