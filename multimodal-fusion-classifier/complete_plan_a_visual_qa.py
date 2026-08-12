"""Finalize Plan A visual QA from a page-by-page review decision."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_ROOT = PROJECT_ROOT / "paper_rewriting_output"
DEFAULT_MANIFEST = PAPER_ROOT / "plan_a_visual_qa" / "REVIEW_MANIFEST.json"
DEFAULT_DECISION = PAPER_ROOT / "plan_a_visual_qa" / "REVIEW_DECISION.json"
DEFAULT_OUTPUT = PAPER_ROOT / "plan_a_visual_qa.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize hash-bound Plan A page-by-page visual QA"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
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
        display = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display = resolved.as_posix()
    return {
        "path": display,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _declared_binding_current(binding: dict[str, Any]) -> bool:
    raw_path = Path(str(binding.get("path", "")))
    path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
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


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reviewed_at must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone")
    return parsed.isoformat()


def complete_visual_qa(
    manifest_path: Path = DEFAULT_MANIFEST,
    decision_path: Path = DEFAULT_DECISION,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    manifest_path = _resolve(manifest_path)
    decision_path = _resolve(decision_path)
    output_path = _resolve(output_path)
    for path in (manifest_path, decision_path):
        if not path.is_file():
            raise FileNotFoundError(f"visual-QA input does not exist: {path}")
    if output_path.exists() or output_path.with_suffix(".md").exists():
        raise FileExistsError("refusing to overwrite completed visual-QA audit")
    manifest = _read_json(manifest_path)
    decision = _read_json(decision_path)
    if (
        manifest.get("valid") is not True
        or manifest.get("state") != "awaiting_page_by_page_review"
        or manifest.get("visual_qa_complete") is not False
    ):
        raise ValueError("visual-QA render manifest is invalid")
    upstream_names = (
        "pdf",
        "docx",
        "main",
        "automated_audit",
        "status",
        "preparer_source",
    )
    stale_upstream = [
        name
        for name in upstream_names
        if not isinstance(manifest.get("source_bindings", {}).get(name), dict)
        or not _declared_binding_current(
            manifest["source_bindings"][name]
        )
    ]
    if stale_upstream:
        raise ValueError(
            "visual-QA upstream bindings changed: "
            + ", ".join(stale_upstream)
        )
    if (
        decision.get("schema_version") != 1
        or decision.get("state") != "complete"
        or decision.get("overall_pass") is not True
    ):
        raise ValueError("visual-QA decision is not a complete pass")
    reviewer = decision.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("visual-QA reviewer must be identified")
    reviewed_at = _timestamp(decision.get("reviewed_at"))
    if (
        decision.get("paper_pdf_sha256")
        != manifest["source_bindings"]["pdf"]["sha256"]
        or decision.get("page_count") != manifest.get("page_count")
    ):
        raise ValueError("visual-QA decision source identity mismatch")
    manifest_pages = {
        row["page_number"]: row for row in manifest.get("pages", [])
    }
    decision_pages = decision.get("pages")
    if (
        not isinstance(decision_pages, list)
        or len(decision_pages) != manifest.get("page_count")
        or {row.get("page_number") for row in decision_pages}
        != set(manifest_pages)
    ):
        raise ValueError("visual-QA page decision coverage mismatch")
    bundle_dir = manifest_path.parent
    for row in decision_pages:
        page_number = row["page_number"]
        source = manifest_pages[page_number]
        page_path = bundle_dir / source["png_path"]
        if (
            row.get("reviewed") is not True
            or row.get("issues") != []
            or row.get("png_sha256") != source["png_sha256"]
            or not page_path.is_file()
            or _sha256(page_path) != source["png_sha256"]
        ):
            raise ValueError(
                f"visual-QA page {page_number} is not a clean bound review"
            )
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "visual_qa_complete": True,
        "reviewer": reviewer.strip(),
        "reviewed_at": reviewed_at,
        "page_count": manifest["page_count"],
        "issue_count": 0,
        "paper_pdf_sha256": manifest["source_bindings"]["pdf"]["sha256"],
        "source_bindings": {
            "review_manifest": _binding(manifest_path),
            "review_decision": _binding(decision_path),
            "paper_pdf": manifest["source_bindings"]["pdf"],
            "paper_docx": manifest["source_bindings"]["docx"],
            "main_tex": manifest["source_bindings"]["main"],
            "automated_build_audit": manifest["source_bindings"][
                "automated_audit"
            ],
            "manuscript_refresh_status": manifest["source_bindings"]["status"],
            "visual_qa_preparer": manifest["source_bindings"][
                "preparer_source"
            ],
            "completer_source": _binding(Path(__file__)),
        },
        "claim_boundary": (
            "This audit proves that every hash-bound rendered page was marked "
            "reviewed with no recorded layout issue. It does not create "
            "external, hardware, or complete-patent evidence."
        ),
        "errors": [],
    }
    _atomic_text(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# 方案A逐页视觉复核",
        "",
        "- 状态：PASS。",
        f"- 页数：{report['page_count']}。",
        f"- 复核者：{report['reviewer']}。",
        "- 记录问题数：0。",
        "",
        "该结论只覆盖绑定版本的论文页面布局，不代表外部或硬件验证。",
        "",
    ]
    _atomic_text(output_path.with_suffix(".md"), "\n".join(lines))
    return report


def main() -> None:
    args = parse_args()
    report = complete_visual_qa(
        args.manifest,
        args.decision,
        args.output,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "visual_qa_complete": report["visual_qa_complete"],
                "page_count": report["page_count"],
                "issue_count": report["issue_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
