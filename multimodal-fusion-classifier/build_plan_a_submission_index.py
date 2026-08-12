"""Build an immutable submission index for the visually approved Plan A paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
PAPER_ROOT = PROJECT_ROOT / "paper_rewriting_output"
DEFAULT_OUTPUT = REPORT_ROOT / "PLAN_A_SUBMISSION_INDEX.json"

REPORT_FILES = {
    "base_final_evidence_audit": "FINAL_EVIDENCE_AUDIT.json",
    "paper_readiness_audit": "PLAN_A_PAPER_READINESS_AUDIT.json",
    "final_claim_evidence": "FINAL_CLAIM_EVIDENCE.json",
    "patent_requirement_matrix": "PATENT_REQUIREMENT_MATRIX.json",
    "physical_sensor_interface": "PHYSICAL_SENSOR_INTERFACE_AUDIT.json",
    "external_blind_readiness": "EXTERNAL_BLIND_READINESS_V1.json",
    "manuscript_governance": "MANUSCRIPT_RESULT_GOVERNANCE_V14.json",
    "full_regression_audit": "FULL_REGRESSION_AUDIT.json",
    "full_regression_audit_markdown": "FULL_REGRESSION_AUDIT.md",
    "component_queue_status": (
        "training/v3_loss_component_ablation_status.json"
    ),
    "unseen_queue_status": "training/unseen_sensor_conditions_status.json",
    "baseline_queue_status": "training/recent_robust_baseline_status.json",
    "manuscript_queue_status": (
        "training/plan_a_manuscript_refresh_status.json"
    ),
    "component_outcome": (
        "confirmatory-ablations/v3-loss-components/OUTCOME.json"
    ),
    "unseen_outcome": "unseen-sensor-conditions-v1/OUTCOME.json",
    "baseline_outcome": "recent-robust-baselines/OUTCOME.json",
    "baseline_protocol_amendment": (
        "recent-robust-baselines/PROTOCOL_AMENDMENT.json"
    ),
    "plan_a_outcome": "plan-a-confirmatory-outcome/OUTCOME.json",
}

PAPER_FILES = {
    "main_tex": "final_paper/main.tex",
    "bibliography": "final_paper/references.bib",
    "paper_docx": "final_paper/paper.zh.docx",
    "paper_pdf": "final_paper/paper.pdf",
    "paper_pdf_alias": "final_paper/main.pdf",
    "delivery_manifest": "final_artifact_manifest.md",
    "latex_report": "latex_report.md",
    "word_report": "word_report.zh.md",
    "artifact_check": "artifact_check.md",
    "citation_bank_check": "citation_bank_check.md",
    "citation_quality_audit": "citation_quality_audit.md",
    "integrity_audit": "integrity_audit.md",
    "progress": "progress.md",
    "revision_audit": "plan_a_confirmatory_revision_audit.json",
    "automated_build_audit": "plan_a_automated_build_audit.json",
    "automated_build_audit_markdown": "plan_a_automated_build_audit.md",
    "visual_qa": "plan_a_visual_qa.json",
    "visual_qa_markdown": "plan_a_visual_qa.md",
}

PROJECT_FILES = {
    "delivery_manifest_builder": "build_plan_a_delivery_manifest.py",
    "submission_index_builder": "build_plan_a_submission_index.py",
    "evidence_archive_builder": "build_evidence_archive.py",
    "evidence_archive_verifier": "verify_evidence_archive.py",
    "completion_audit_builder": "build_project_completion_audit_v1.py",
    "visual_qa_preparer": "prepare_plan_a_visual_qa.py",
    "visual_qa_completer": "complete_plan_a_visual_qa.py",
    "post_visual_finalizer": (
        "reports/qwen-material/training/"
        "run_plan_a_post_visual_finalize.ps1"
    ),
    "equivalent_priority_guard": (
        "reports/qwen-material/training/"
        "run_equivalent_training_priority_guard.ps1"
    ),
    "visual_qa_preparation_watcher": (
        "reports/qwen-material/training/"
        "run_plan_a_visual_qa_prepare_after_refresh.ps1"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable Plan A submission index"
    )
    parser.add_argument("--reports", type=Path, default=REPORT_ROOT)
    parser.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing immutable index and all source bindings",
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


def _binding(path: Path, *, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(project_root).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _declared_binding_current(
    binding: Mapping[str, Any],
    *,
    project_root: Path,
) -> bool:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    path = _resolve(Path(raw_path), project_root=project_root)
    try:
        path.relative_to(project_root)
    except ValueError:
        return False
    return (
        path.is_file()
        and path.stat().st_size == binding.get("size_bytes")
        and _sha256(path) == binding.get("sha256")
    )


def _validate_bound_report(
    report: Mapping[str, Any],
    *,
    project_root: Path,
) -> list[str]:
    bindings = report.get("source_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        return ["source bindings are missing"]
    return [
        f"source binding drift: {name}"
        for name, binding in bindings.items()
        if not isinstance(binding, Mapping)
        or not _declared_binding_current(
            binding,
            project_root=project_root,
        )
    ]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _require_complete(report: Mapping[str, Any], label: str) -> None:
    if report.get("state") != "complete" or report.get("valid") is not True:
        raise ValueError(f"{label} is not complete and valid")


def verify_plan_a_submission_index(
    index_path: Path = DEFAULT_OUTPUT,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    index_path = _resolve(index_path, project_root=project_root)
    markdown_path = index_path.with_suffix(".md")
    errors = []
    if not index_path.is_file():
        errors.append(f"Plan A submission index is missing: {index_path}")
        index: dict[str, Any] = {}
    else:
        index = _read_json(index_path)
    if not markdown_path.is_file():
        errors.append(f"Plan A submission Markdown is missing: {markdown_path}")
    if (
        index.get("schema_version") != 1
        or index.get("state") != "complete"
        or index.get("valid") is not True
        or index.get("scope")
        != "plan_a_visually_approved_internal_paper_release"
    ):
        errors.append("Plan A submission index header is invalid")
    for gate in (
        "internal_paper_evidence_ready",
        "plan_a_confirmatory_evidence_ready",
        "plan_a_visual_qa_complete",
        "paper_spine_final_gate_passed",
    ):
        if index.get("gates", {}).get(gate) is not True:
            errors.append(f"Plan A submission gate is not complete: {gate}")
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        errors.append("Plan A submission artifact bindings are missing")
        artifacts = {}
    stale = [
        name
        for name, binding in artifacts.items()
        if not isinstance(binding, Mapping)
        or not _declared_binding_current(
            binding,
            project_root=project_root,
        )
    ]
    errors.extend(f"Plan A submission binding drift: {name}" for name in stale)
    return {
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "index": index_path.as_posix(),
        "artifact_count": len(artifacts),
        "error_count": len(errors),
        "errors": errors,
    }


def build_plan_a_submission_index(
    *,
    project_root: Path = PROJECT_ROOT,
    reports_root: Path = REPORT_ROOT,
    paper_root: Path = PAPER_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    reports_root = _resolve(reports_root, project_root=project_root)
    paper_root = _resolve(paper_root, project_root=project_root)
    output_path = _resolve(output_path, project_root=project_root)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("Plan A submission index is immutable")

    report_paths = {
        name: reports_root / relative
        for name, relative in REPORT_FILES.items()
    }
    paper_paths = {
        name: paper_root / relative
        for name, relative in PAPER_FILES.items()
    }
    project_paths = {
        name: project_root / relative
        for name, relative in PROJECT_FILES.items()
    }
    paths = {**report_paths, **paper_paths, **project_paths}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Plan A submission inputs missing: " + ", ".join(missing)
        )

    documents = {
        name: _read_json(path)
        for name, path in paths.items()
        if path.suffix.lower() == ".json"
    }
    _require_complete(
        documents["base_final_evidence_audit"],
        "base final evidence audit",
    )
    readiness = documents["paper_readiness_audit"]
    if (
        readiness.get("state") != "complete"
        or readiness.get("valid") is not True
        or readiness.get("internal_paper_evidence_ready") is not True
        or readiness.get("source_evidence_audit_sha256")
        != _sha256(report_paths["base_final_evidence_audit"])
    ):
        raise ValueError("current paper-readiness audit is not complete")
    claims = documents["final_claim_evidence"]
    if (
        claims.get("state") != "complete"
        or claims.get("source_final_audit_sha256")
        != _sha256(report_paths["base_final_evidence_audit"])
    ):
        raise ValueError("final claim evidence is not bound to the base audit")
    if documents["patent_requirement_matrix"].get("valid") is not True:
        raise ValueError("patent requirement matrix is invalid")
    _require_complete(
        documents["physical_sensor_interface"],
        "physical sensor interface audit",
    )
    _require_complete(
        documents["external_blind_readiness"],
        "external blind readiness audit",
    )
    governance = documents["manuscript_governance"]
    if (
        governance.get("valid") is not True
        or governance.get("decision", {}).get("confirmed_option") != "A"
    ):
        raise ValueError("manuscript governance does not confirm option A")
    regression = documents["full_regression_audit"]
    if (
        regression.get("state") != "complete"
        or regression.get("valid") is not True
        or regression.get("combined", {}).get(
            "all_collected_tests_passed"
        )
        is not True
        or regression.get("combined", {}).get("unresolved_skips") != 0
    ):
        raise ValueError("full split-environment regression is not complete")

    for name in (
        "component_queue_status",
        "unseen_queue_status",
        "baseline_queue_status",
    ):
        if documents[name].get("state") != "complete":
            raise ValueError(f"confirmatory queue is incomplete: {name}")
    if (
        documents["manuscript_queue_status"].get("state")
        != "awaiting_visual_qa"
    ):
        raise ValueError("manuscript queue did not reach visual-QA handoff")
    for name in (
        "component_outcome",
        "unseen_outcome",
        "baseline_outcome",
        "plan_a_outcome",
        "revision_audit",
    ):
        _require_complete(documents[name], name)

    automated = documents["automated_build_audit"]
    if (
        automated.get("valid") is not True
        or automated.get("state")
        != "automated_build_complete_visual_qa_pending"
    ):
        raise ValueError("automated paper-build audit is invalid")
    visual = documents["visual_qa"]
    if (
        visual.get("state") != "complete"
        or visual.get("valid") is not True
        or visual.get("visual_qa_complete") is not True
        or visual.get("issue_count") != 0
    ):
        raise ValueError("Plan A visual QA is not a clean pass")
    visual_errors = _validate_bound_report(
        visual,
        project_root=project_root,
    )
    if visual_errors:
        raise ValueError("; ".join(visual_errors))
    nested_binding_errors = []
    for name, document in documents.items():
        if name == "visual_qa" or not document.get("source_bindings"):
            continue
        nested_binding_errors.extend(
            f"{name}: {error}"
            for error in _validate_bound_report(
                document,
                project_root=project_root,
            )
        )
    if nested_binding_errors:
        raise ValueError("; ".join(nested_binding_errors))

    if _sha256(paper_paths["paper_pdf"]) != _sha256(
        paper_paths["paper_pdf_alias"]
    ):
        raise ValueError("canonical PDF and delivery alias differ")
    if visual.get("paper_pdf_sha256") != _sha256(
        paper_paths["paper_pdf"]
    ):
        raise ValueError("visual QA is not bound to the indexed PDF")

    artifact_text = paper_paths["artifact_check"].read_text(
        encoding="utf-8-sig"
    )
    progress_text = paper_paths["progress"].read_text(encoding="utf-8-sig")
    integrity_text = paper_paths["integrity_audit"].read_text(
        encoding="utf-8-sig"
    )
    word_text = paper_paths["word_report"].read_text(encoding="utf-8-sig")
    if "Status: PASS" not in artifact_text:
        raise ValueError("PaperSpine artifact check is not PASS")
    if "Next stage: Complete" not in progress_text:
        raise ValueError("PaperSpine progress is not complete")
    if "BLOCKED" in integrity_text or "Status: FAIL" in integrity_text:
        raise ValueError("PaperSpine integrity audit is not clean")
    if "FAIL" in word_text or "BLOCKED" in word_text:
        raise ValueError("Word guard report is not clean")

    bindings = {
        name: _binding(path, project_root=project_root)
        for name, path in paths.items()
    }
    index = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "plan_a_visually_approved_internal_paper_release",
        "gates": {
            "internal_paper_evidence_ready": True,
            "plan_a_confirmatory_evidence_ready": True,
            "plan_a_visual_qa_complete": True,
            "paper_spine_final_gate_passed": True,
            "complete_patent_replication_ready": False,
            "physical_hardware_validation_ready": False,
            "external_blind_validation_ready": False,
            "rgbd_validation_ready": False,
        },
        "document": {
            "title": (
                "面向图像条件软传感器退化的可靠多模态垃圾分类方法"
            ),
            "language": "zh",
            "page_count": visual.get("page_count"),
            "visual_reviewer": visual.get("reviewer"),
            "visual_reviewed_at": visual.get("reviewed_at"),
            "canonical_pdf_sha256": _sha256(paper_paths["paper_pdf"]),
        },
        "artifacts": bindings,
        "boundaries": [
            "Qwen/rule-v1 sidecars are image-conditioned soft sensors.",
            "The fixed internal test set is not an external blind test.",
            "No physical 4x4 array or RGB-D acquisition has been validated.",
            "The release does not claim complete reproduction of the patent.",
            "Hardware deployment remains an independent evidence gate.",
        ],
        "errors": [],
    }
    _atomic_text(
        output_path,
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
    )
    gate_rows = [
        (name, value) for name, value in index["gates"].items()
    ]
    artifact_rows = [
        (
            f"| `{name}` | `{binding['path']}` | "
            f"`{binding['sha256']}` |"
        )
        for name, binding in bindings.items()
    ]
    markdown = "\n".join(
        [
            "# 方案A最终投稿证据索引",
            "",
            "本索引对应完成确认性实验、机器审计和逐页视觉复核后的内部论文版本。",
            "",
            "## 就绪门",
            "",
            "| 门 | 状态 |",
            "|---|---|",
            *[
                f"| `{name}` | `{str(value).lower()}` |"
                for name, value in gate_rows
            ],
            "",
            "## 绑定产物",
            "",
            "| 名称 | 路径 | SHA-256 |",
            "|---|---|---|",
            *artifact_rows,
            "",
            "## 主张边界",
            "",
            *[f"- {item}" for item in index["boundaries"]],
            "",
        ]
    )
    _atomic_text(markdown_path, markdown)
    return index


def main() -> None:
    args = parse_args()
    if args.verify_only:
        report = verify_plan_a_submission_index(
            args.output,
            project_root=PROJECT_ROOT,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["valid"] is not True:
            raise SystemExit(1)
        return
    report = build_plan_a_submission_index(
        project_root=PROJECT_ROOT,
        reports_root=args.reports,
        paper_root=args.paper_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "scope": report["scope"],
                "artifact_count": len(report["artifacts"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
