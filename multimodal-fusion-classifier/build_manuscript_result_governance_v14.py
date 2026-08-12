"""Record the visually approved Plan A manuscript release."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
PAPER_ROOT = PROJECT_ROOT / "paper_rewriting_output"
DEFAULT_OUTPUT = REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V14.json"
INPUTS = {
    "governance_v13": REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V13.json",
    "baseline_protocol_amendment": (
        REPORT_ROOT / "recent-robust-baselines" / "PROTOCOL_AMENDMENT.json"
    ),
    "baseline_outcome": (
        REPORT_ROOT / "recent-robust-baselines" / "OUTCOME.json"
    ),
    "confirmatory_outcome": (
        REPORT_ROOT / "plan-a-confirmatory-outcome" / "OUTCOME.json"
    ),
    "manuscript_revision_audit": (
        PAPER_ROOT / "plan_a_confirmatory_revision_audit.json"
    ),
    "automated_build_audit": (
        PAPER_ROOT / "plan_a_automated_build_audit.json"
    ),
    "visual_qa": PAPER_ROOT / "plan_a_visual_qa.json",
    "manuscript_refresh_status": (
        REPORT_ROOT / "training" / "plan_a_manuscript_refresh_status.json"
    ),
    "main_tex": PAPER_ROOT / "final_paper" / "main.tex",
    "paper_docx": PAPER_ROOT / "final_paper" / "paper.zh.docx",
    "paper_pdf": PAPER_ROOT / "final_paper" / "paper.pdf",
    "docx_styler": PAPER_ROOT / "final_paper" / "style_docx.py",
    "automated_build_auditor": (
        PROJECT_ROOT / "audit_plan_a_automated_paper_build.py"
    ),
    "completion_auditor": PROJECT_ROOT / "build_project_completion_audit_v1.py",
    "submission_index_builder": (
        PROJECT_ROOT / "build_plan_a_submission_index.py"
    ),
    "evidence_archive_builder": PROJECT_ROOT / "build_evidence_archive.py",
    "governance_v14_builder": Path(__file__),
}
EXPECTED_V13_DRIFT = {
    "automated_build_auditor",
    "completion_auditor",
    "submission_index_builder",
    "evidence_archive_builder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manuscript result governance v14"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


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
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _binding_current(binding: Mapping[str, Any]) -> bool:
    raw = binding.get("path")
    if not isinstance(raw, str) or not raw:
        return False
    path = _resolve(Path(raw))
    return (
        path.is_file()
        and path.stat().st_size == binding.get("size_bytes")
        and _sha256(path) == binding.get("sha256")
    )


def _binding_map_current(report: Mapping[str, Any]) -> bool:
    bindings = report.get("source_bindings")
    return (
        isinstance(bindings, Mapping)
        and bool(bindings)
        and all(
            isinstance(binding, Mapping) and _binding_current(binding)
            for binding in bindings.values()
        )
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def validate_visual_release_transition(
    governance_v13: Mapping[str, Any],
    amendment: Mapping[str, Any],
    baseline_outcome: Mapping[str, Any],
    confirmatory_outcome: Mapping[str, Any],
    automated_audit: Mapping[str, Any],
    visual_qa: Mapping[str, Any],
    manuscript_status: Mapping[str, Any],
) -> None:
    method_seeds = {"qmf": [42, 43, 44], "tmc": [42]}
    if (
        governance_v13.get("valid") is not True
        or governance_v13.get("version") != 13
        or governance_v13.get("decision", {}).get("confirmed_option") != "A"
        or governance_v13.get("baseline_reporting_contract", {}).get(
            "method_training_seeds"
        )
        != method_seeds
    ):
        raise ValueError("governance v13 identity mismatch")
    if (
        amendment.get("valid") is not True
        or amendment.get("method_training_seeds") != method_seeds
        or amendment.get("scope_reduction", {}).get(
            "single_run_reported_as_replication"
        )
        is not False
        or amendment.get("scope_reduction", {}).get(
            "partial_run_used_for_results"
        )
        is not False
    ):
        raise ValueError("baseline protocol amendment mismatch")
    if (
        baseline_outcome.get("valid") is not True
        or baseline_outcome.get("state") != "complete"
        or baseline_outcome.get("method_training_seeds") != method_seeds
        or baseline_outcome.get("run_count") != 4
    ):
        raise ValueError("baseline outcome mismatch")
    tmc_rows = [
        row
        for row in baseline_outcome.get("summary", [])
        if row.get("method") == "tmc"
    ]
    if (
        not tmc_rows
        or any(row.get("seed_count") != 1 for row in tmc_rows)
        or any(
            row.get("accuracy_sample_std") is not None
            or row.get("macro_f1_sample_std") is not None
            for row in tmc_rows
        )
    ):
        raise ValueError("TMC single-seed reporting contract mismatch")
    if (
        confirmatory_outcome.get("valid") is not True
        or confirmatory_outcome.get("state") != "complete"
    ):
        raise ValueError("confirmatory outcome mismatch")
    if (
        automated_audit.get("valid") is not True
        or automated_audit.get("state")
        != "automated_build_complete_visual_qa_pending"
        or automated_audit.get("visual_qa_complete") is not False
    ):
        raise ValueError("automated paper-build audit mismatch")
    if (
        visual_qa.get("valid") is not True
        or visual_qa.get("state") != "complete"
        or visual_qa.get("visual_qa_complete") is not True
        or visual_qa.get("issue_count") != 0
        or visual_qa.get("page_count") != 13
    ):
        raise ValueError("visual QA is not a clean 13-page pass")
    if manuscript_status.get("state") != "awaiting_visual_qa":
        raise ValueError("manuscript refresh did not reach visual-QA handoff")


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite governance v14")
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "governance v14 inputs missing: " + ", ".join(missing)
        )
    documents = {name: _read_json(path) for name, path in INPUTS.items()
                 if path.suffix == ".json"}
    governance_v13 = documents["governance_v13"]
    validate_visual_release_transition(
        governance_v13,
        documents["baseline_protocol_amendment"],
        documents["baseline_outcome"],
        documents["confirmatory_outcome"],
        documents["automated_build_audit"],
        documents["visual_qa"],
        documents["manuscript_refresh_status"],
    )
    for name in (
        "manuscript_revision_audit",
        "automated_build_audit",
        "visual_qa",
    ):
        if not _binding_map_current(documents[name]):
            raise ValueError(f"{name} source bindings are stale")

    old_bindings = governance_v13.get("source_bindings", {})
    if not isinstance(old_bindings, Mapping):
        raise ValueError("governance v13 source bindings are missing")
    drifted_v13 = sorted(
        name
        for name, binding in old_bindings.items()
        if not isinstance(binding, Mapping) or not _binding_current(binding)
    )
    if set(drifted_v13) != EXPECTED_V13_DRIFT:
        raise ValueError(
            "governance v13 drift is outside the visual-release transition: "
            + ", ".join(drifted_v13)
        )

    report = {
        "schema_version": 1,
        "version": 14,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": governance_v13["decision"],
        "current_paper": {
            "audited_draft_complete": True,
            "paper_spine_final_gate_passed": True,
            "external_submission_ready": False,
            "editorial_position": (
                "internally_complete_external_validation_pending"
            ),
            "confirmatory_revision_applied": True,
            "automated_build_audit_passed": True,
            "visual_qa_complete": True,
            "visual_qa_page_count": documents["visual_qa"]["page_count"],
            "reason": (
                "Plan A confirmatory results were incorporated and the "
                "hash-bound DOCX/PDF passed automated and page-by-page QA."
            ),
        },
        "governance_transition": {
            "supersedes_version": 13,
            "scope": "document_audit_correction_and_visual_release",
            "drifted_v13_bindings": drifted_v13,
            "model_or_loss_changed": False,
            "data_or_sample_order_changed": False,
            "completed_run_metrics_rewritten": False,
            "paper_content_metrics_changed": False,
            "partial_run_used_for_results": False,
            "single_run_reported_as_replication": False,
        },
        "baseline_reporting_contract": (
            governance_v13["baseline_reporting_contract"]
        ),
        "unchanged_evidence_boundaries": (
            governance_v13["unchanged_evidence_boundaries"]
        ),
        "claim_policy": {
            "allowed_now": [
                "QMF同输入适配报告三种子均值与样本标准差。",
                "TMC同输入适配仅报告seed42单种子参考，样本标准差不适用。",
                "可报告当前绑定版本的13页论文已完成自动审计和逐页视觉复核。",
            ],
            "not_allowed_now": [
                "把TMC seed42称为多次实验、均值或重复性证据。",
                "把部分TMC seed43或未启动seed44纳入结果。",
                "把内部固定测试、软sidecar或软件故障称为外部硬件证据。",
            ],
        },
        "source_bindings": {
            name: _binding(path) for name, path in INPUTS.items()
        },
        "errors": [],
    }
    _atomic_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(
        markdown,
        "\n".join(
            [
                "# 论文结果治理 v14：方案A视觉放行",
                "",
                "- 已继承v13的诚实统计范围：QMF n=3，TMC n=1参考。",
                "- 当前DOCX/PDF已通过自动内容审计和13页逐页视觉复核。",
                "- 本次只修正文档审计与发布绑定；模型、数据及指标未改写。",
                "- 内部固定测试仍不等于外部盲测，软sidecar仍不等于硬件测量。",
                "",
            ]
        ),
    )
    return report


def main() -> None:
    report = build(parse_args().output)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "version": report["version"],
                "confirmed_option": report["decision"]["confirmed_option"],
                "visual_qa_complete": report["current_paper"][
                    "visual_qa_complete"
                ],
                "visual_qa_page_count": report["current_paper"][
                    "visual_qa_page_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
