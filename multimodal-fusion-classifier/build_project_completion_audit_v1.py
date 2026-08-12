"""Build a current, scope-aware completion audit for the project."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from build_evidence_archive import PLAN_A_TRAINING_RUN_DIRS
from verify_evidence_archive import verify_evidence_archive


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
PAPER_ROOT = PROJECT_ROOT / "paper_rewriting_output"
FINAL_PAPER = PAPER_ROOT / "final_paper"
DEFAULT_OUTPUT = REPORT_ROOT / "PROJECT_COMPLETION_AUDIT_V1.json"

MANDATORY_SOURCES = {
    "qwen_stats": PROJECT_ROOT / "data" / "qwen-material" / "stats.json",
    "full_regression": REPORT_ROOT / "FULL_REGRESSION_AUDIT.json",
    "paper_readiness": REPORT_ROOT / "PAPER_READINESS_AUDIT.json",
    "patent_matrix": REPORT_ROOT / "PATENT_REQUIREMENT_MATRIX.json",
    "physical_interface": (
        REPORT_ROOT / "PHYSICAL_SENSOR_INTERFACE_AUDIT.json"
    ),
    "external_readiness": REPORT_ROOT / "EXTERNAL_BLIND_READINESS_V1.json",
    "governance_current": (
        REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V14.json"
    ),
    "component_status": (
        REPORT_ROOT / "training" / "v3_loss_component_ablation_status.json"
    ),
    "unseen_status": (
        REPORT_ROOT / "training" / "unseen_sensor_conditions_status.json"
    ),
    "recent_baseline_status": (
        REPORT_ROOT / "training" / "recent_robust_baseline_status.json"
    ),
    "manuscript_refresh_status": (
        REPORT_ROOT / "training" / "plan_a_manuscript_refresh_status.json"
    ),
    "main_tex": FINAL_PAPER / "main.tex",
    "paper_docx": FINAL_PAPER / "paper.zh.docx",
    "paper_pdf": FINAL_PAPER / "paper.pdf",
    "legacy_submission_index": REPORT_ROOT / "SUBMISSION_INDEX.json",
    "legacy_archive_manifest": (
        REPORT_ROOT / "submission-evidence" / "archive-manifest.json"
    ),
    "final_artifact_manifest": PAPER_ROOT / "final_artifact_manifest.md",
    "visual_qa_preparer": PROJECT_ROOT / "prepare_plan_a_visual_qa.py",
    "visual_qa_completer": PROJECT_ROOT / "complete_plan_a_visual_qa.py",
}

OPTIONAL_SOURCES = {
    "plan_a_paper_readiness": (
        REPORT_ROOT / "PLAN_A_PAPER_READINESS_AUDIT.json"
    ),
    "component_outcome": (
        REPORT_ROOT
        / "confirmatory-ablations"
        / "v3-loss-components"
        / "OUTCOME.json"
    ),
    "unseen_outcome": (
        REPORT_ROOT / "unseen-sensor-conditions-v1" / "OUTCOME.json"
    ),
    "recent_baseline_outcome": (
        REPORT_ROOT / "recent-robust-baselines" / "OUTCOME.json"
    ),
    "plan_a_confirmatory_outcome": (
        REPORT_ROOT / "plan-a-confirmatory-outcome" / "OUTCOME.json"
    ),
    "plan_a_revision_audit": (
        PAPER_ROOT / "plan_a_confirmatory_revision_audit.json"
    ),
    "plan_a_automated_build_audit": (
        PAPER_ROOT / "plan_a_automated_build_audit.json"
    ),
    "plan_a_visual_qa": PAPER_ROOT / "plan_a_visual_qa.json",
    "plan_a_submission_index": REPORT_ROOT / "PLAN_A_SUBMISSION_INDEX.json",
    "plan_a_archive_manifest": (
        REPORT_ROOT
        / "submission-evidence-plan-a"
        / "archive-manifest.json"
    ),
    "plan_a_evidence_manifest": (
        REPORT_ROOT
        / "submission-evidence-plan-a"
        / "evidence-manifest.json"
    ),
    "plan_a_archive": (
        REPORT_ROOT / "submission-evidence-plan-a" / "paper-evidence.zip"
    ),
}

QWEN_PAPER_GATES = (
    "qwen_dataset_ready",
    "full_regression_ready",
    "internal_evidence_ready",
    "patent_method_alignment_ready",
    "option_a_confirmed",
    "audited_draft_available",
    "plan_a_confirmatory_evidence_ready",
    "plan_a_automated_build_ready",
    "plan_a_visual_qa_complete",
    "submission_index_current",
    "evidence_archive_current",
)

GATE_ACTIONS = {
    "qwen_dataset_ready": "Repair and revalidate the frozen Qwen dataset.",
    "full_regression_ready": (
        "Run and audit the complete split-environment regression suite."
    ),
    "internal_evidence_ready": "Repair the internal paper evidence audit.",
    "patent_method_alignment_ready": (
        "Restore the bounded patent-alignment method evidence."
    ),
    "option_a_confirmed": "Confirm manuscript option A.",
    "audited_draft_available": "Restore the audited TeX/DOCX/PDF draft.",
    "plan_a_confirmatory_evidence_ready": (
        "Finish component, unseen-condition, and recent-baseline outcomes."
    ),
    "plan_a_automated_build_ready": (
        "Run the queued Plan A manuscript refresh and machine content audit."
    ),
    "plan_a_visual_qa_complete": (
        "Complete page-by-page DOCX/PDF visual QA and record its audit."
    ),
    "submission_index_current": (
        "Refresh the submission index after the final visual revision."
    ),
    "evidence_archive_current": (
        "Refresh and verify the evidence archive after the final revision."
    ),
}


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
    size_matches = (
        "size_bytes" not in binding
        or path.is_file()
        and path.stat().st_size == binding.get("size_bytes")
    )
    return (
        path.is_file()
        and size_matches
        and _sha256(path) == binding.get("sha256")
    )


def _binding_map_integrity(
    bindings: dict[str, Any],
) -> tuple[bool, list[str]]:
    drifted = [
        str(name)
        for name, binding in bindings.items()
        if not isinstance(binding, dict)
        or not _declared_binding_current(binding)
    ]
    return not drifted, drifted


def _paper_readiness_integrity(
    report: dict[str, Any],
) -> tuple[bool, list[str]]:
    drifted: list[str] = []
    for requirement_name, requirement in report.get(
        "requirements", {}
    ).items():
        if not isinstance(requirement, dict):
            drifted.append(str(requirement_name))
            continue
        for index, artifact in enumerate(requirement.get("artifacts", [])):
            if (
                not isinstance(artifact, dict)
                or not _declared_binding_current(artifact)
            ):
                artifact_path = (
                    artifact.get("path")
                    if isinstance(artifact, dict)
                    else None
                )
                drifted.append(
                    str(artifact_path)
                    if artifact_path
                    else f"{requirement_name}.artifacts[{index}]"
                )
    return not drifted, drifted


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _valid_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    value = _read_json(path)
    return value.get("valid") is True and value.get("state") == "complete"


def _mtime_current(candidate: Path, required: list[Path]) -> bool:
    if not candidate.is_file() or any(not path.is_file() for path in required):
        return False
    return candidate.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns for path in required
    )


def build_completion_audit(
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = _resolve(output)
    mandatory = {name: _resolve(path) for name, path in MANDATORY_SOURCES.items()}
    optional = {name: _resolve(path) for name, path in OPTIONAL_SOURCES.items()}
    missing = [name for name, path in mandatory.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "project completion audit inputs missing: " + ", ".join(missing)
        )

    stats = _read_json(mandatory["qwen_stats"])
    full_regression = _read_json(mandatory["full_regression"])
    paper_readiness_path = (
        optional["plan_a_paper_readiness"]
        if optional["plan_a_paper_readiness"].is_file()
        else mandatory["paper_readiness"]
    )
    paper_readiness = _read_json(paper_readiness_path)
    patent = _read_json(mandatory["patent_matrix"])
    physical = _read_json(mandatory["physical_interface"])
    external = _read_json(mandatory["external_readiness"])
    governance = _read_json(mandatory["governance_current"])
    statuses = {
        "component": _read_json(mandatory["component_status"]),
        "unseen": _read_json(mandatory["unseen_status"]),
        "recent_baseline": _read_json(
            mandatory["recent_baseline_status"]
        ),
        "manuscript_refresh": _read_json(
            mandatory["manuscript_refresh_status"]
        ),
    }
    paper_bindings_current, paper_binding_drift = (
        _paper_readiness_integrity(paper_readiness)
    )
    patent_bindings_current, patent_binding_drift = _binding_map_integrity(
        {
            **patent.get("bindings", {}),
            **patent.get("evidence_bindings", {}),
        }
    )
    physical_bindings_current, physical_binding_drift = (
        _binding_map_integrity(physical.get("source_bindings", {}))
    )
    external_bindings_current, external_binding_drift = (
        _binding_map_integrity(external.get("bindings", {}))
    )
    governance_bindings_current, governance_binding_drift = (
        _binding_map_integrity(governance.get("source_bindings", {}))
    )
    regression_bindings_current, regression_binding_drift = (
        _binding_map_integrity(
            full_regression.get("source_bindings", {})
        )
    )

    qwen_dataset_ready = (
        stats.get("samples") == 8654
        and stats.get("sensor_records") == 8654
        and stats.get("source_images") == 5990
        and stats.get("split_counts")
        == {"train": 7002, "val": 826, "test": 826}
        and stats.get("qwen_success_records") == 8639
        and stats.get("fallback_sensor_records") == 15
    )
    full_regression_ready = (
        full_regression.get("state") == "complete"
        and full_regression.get("valid") is True
        and full_regression.get("combined", {}).get(
            "all_collected_tests_passed"
        )
        is True
        and full_regression.get("combined", {}).get(
            "unresolved_skips"
        )
        == 0
        and regression_bindings_current
    )
    internal_evidence_ready = (
        paper_readiness.get("valid") is True
        and paper_readiness.get("internal_paper_evidence_ready") is True
        and paper_bindings_current
    )
    patent_method_alignment_ready = (
        patent.get("valid") is True
        and patent_bindings_current
        and patent.get("summary", {}).get(
            "architecture_method_section_ready_with_boundaries"
        )
        is True
    )
    option_a_confirmed = (
        governance.get("valid") is True
        and governance_bindings_current
        and governance.get("decision", {}).get("confirmed_option") == "A"
    )
    audited_draft_available = (
        governance.get("current_paper", {}).get("audited_draft_complete") is True
        and governance.get("current_paper", {}).get(
            "paper_spine_final_gate_passed"
        )
        is True
        and all(
            mandatory[name].is_file()
            for name in ("main_tex", "paper_docx", "paper_pdf")
        )
    )
    plan_a_confirmatory_evidence_ready = (
        all(
            statuses[name].get("state") == "complete"
            for name in ("component", "unseen", "recent_baseline")
        )
        and all(
            _valid_complete(optional[name])
            for name in (
                "component_outcome",
                "unseen_outcome",
                "recent_baseline_outcome",
            )
        )
    )
    plan_a_automated_build_ready = (
        statuses["manuscript_refresh"].get("state") == "awaiting_visual_qa"
        and _valid_complete(optional["plan_a_confirmatory_outcome"])
        and _valid_complete(optional["plan_a_revision_audit"])
        and (
            optional["plan_a_automated_build_audit"].is_file()
            and _read_json(optional["plan_a_automated_build_audit"]).get(
                "valid"
            )
            is True
        )
    )
    visual_report = (
        _read_json(optional["plan_a_visual_qa"])
        if optional["plan_a_visual_qa"].is_file()
        else {}
    )
    visual_report_bindings_current, visual_report_binding_drift = (
        _binding_map_integrity(visual_report.get("source_bindings", {}))
        if visual_report
        else (False, [])
    )
    plan_a_visual_qa_complete = (
        visual_report.get("valid") is True
        and visual_report.get("state") == "complete"
        and visual_report.get("visual_qa_complete") is True
        and visual_report_bindings_current
    )
    plan_a_index = (
        _read_json(optional["plan_a_submission_index"])
        if optional["plan_a_submission_index"].is_file()
        else {}
    )
    plan_a_index_bindings_current, plan_a_index_binding_drift = (
        _binding_map_integrity(plan_a_index.get("artifacts", {}))
        if plan_a_index
        else (False, [])
    )
    submission_index_current = (
        plan_a_visual_qa_complete
        and plan_a_index.get("state") == "complete"
        and plan_a_index.get("valid") is True
        and plan_a_index.get("gates", {}).get(
            "plan_a_visual_qa_complete"
        )
        is True
        and plan_a_index.get("gates", {}).get(
            "paper_spine_final_gate_passed"
        )
        is True
        and plan_a_index_bindings_current
    )
    plan_a_archive_paths_ready = all(
        optional[name].is_file()
        for name in (
            "plan_a_archive_manifest",
            "plan_a_evidence_manifest",
            "plan_a_archive",
        )
    )
    plan_a_archive_verification = (
        verify_evidence_archive(
            optional["plan_a_archive_manifest"].parent
        )
        if plan_a_archive_paths_ready
        else {"valid": False, "errors": ["Plan A archive is missing"]}
    )
    plan_a_evidence = (
        _read_json(optional["plan_a_evidence_manifest"])
        if plan_a_archive_paths_ready
        else {}
    )
    plan_a_archive_manifest = (
        _read_json(optional["plan_a_archive_manifest"])
        if plan_a_archive_paths_ready
        else {}
    )
    archived_entries = {
        entry.get("path"): entry
        for entry in plan_a_evidence.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    required_archived_paths = (
        "reports/qwen-material/PLAN_A_SUBMISSION_INDEX.json",
        "paper_rewriting_output/final_paper/main.tex",
        "paper_rewriting_output/final_paper/paper.zh.docx",
        "paper_rewriting_output/final_paper/paper.pdf",
        "paper_rewriting_output/final_artifact_manifest.md",
        "paper_rewriting_output/plan_a_visual_qa.json",
    )
    required_training_metadata_paths = tuple(
        f"{run_dir}/{filename}"
        for run_dir in PLAN_A_TRAINING_RUN_DIRS
        for filename in ("history.json", "training_summary.json")
    )
    plan_a_archive_bindings_current = all(
        path in archived_entries
        and _declared_binding_current(archived_entries[path])
        for path in required_archived_paths
    )
    plan_a_training_metadata_current = all(
        path in archived_entries
        and _declared_binding_current(archived_entries[path])
        for path in required_training_metadata_paths
    )
    evidence_archive_current = (
        submission_index_current
        and plan_a_archive_verification.get("valid") is True
        and plan_a_archive_manifest.get("plan_a_release_included") is True
        and plan_a_evidence.get("plan_a_release_included") is True
        and plan_a_archive_bindings_current
        and plan_a_training_metadata_current
    )

    gates = {
        "qwen_dataset_ready": qwen_dataset_ready,
        "full_regression_ready": full_regression_ready,
        "internal_evidence_ready": internal_evidence_ready,
        "patent_method_alignment_ready": patent_method_alignment_ready,
        "option_a_confirmed": option_a_confirmed,
        "audited_draft_available": audited_draft_available,
        "plan_a_confirmatory_evidence_ready": (
            plan_a_confirmatory_evidence_ready
        ),
        "plan_a_automated_build_ready": plan_a_automated_build_ready,
        "plan_a_visual_qa_complete": plan_a_visual_qa_complete,
        "submission_index_current": submission_index_current,
        "evidence_archive_current": evidence_archive_current,
    }
    blockers = [
        {
            "gate": name,
            "action": GATE_ACTIONS[name],
        }
        for name in QWEN_PAPER_GATES
        if gates[name] is not True
    ]
    qwen_paper_goal_ready = all(
        gates[name] is True for name in QWEN_PAPER_GATES
    )
    complete_patent_replication_ready = (
        patent.get("summary", {}).get("complete_patent_replication") is True
    )
    hardware_validation_ready = (
        physical_bindings_current
        and external_bindings_current
        and physical.get("hardware_validation_ready") is True
        and physical.get("physical_record_count", 0) > 0
    )
    external_deployment_ready = (
        external.get("formal_external_result_available") is True
        and external.get("submission_ready") is True
        and hardware_validation_ready
    )

    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "current_project_completion_audit_for_qwen_paper_option_a",
        "objective": (
            "接近专利结构并完成可写论文的Qwen软传感器方案A，同时独立披露"
            "完整专利复现和外部部署证据门。"
        ),
        "qwen_paper_goal_ready": qwen_paper_goal_ready,
        "qwen_paper_gates": gates,
        "blocking_count": len(blockers),
        "blockers": blockers,
        "queue_snapshot": {
            name: {
                "state": value.get("state"),
                "stage": value.get("stage"),
                "error": value.get("error"),
            }
            for name, value in statuses.items()
        },
        "source_integrity": {
            "paper_readiness_path": (
                paper_readiness_path.relative_to(PROJECT_ROOT).as_posix()
            ),
            "paper_readiness_bindings_current": paper_bindings_current,
            "paper_readiness_drift": paper_binding_drift,
            "patent_matrix_bindings_current": patent_bindings_current,
            "patent_matrix_drift": patent_binding_drift,
            "physical_interface_bindings_current": physical_bindings_current,
            "physical_interface_drift": physical_binding_drift,
            "external_readiness_bindings_current": external_bindings_current,
            "external_readiness_drift": external_binding_drift,
            "governance_v14_bindings_current": governance_bindings_current,
            "governance_v14_drift": governance_binding_drift,
            "full_regression_bindings_current": (
                regression_bindings_current
            ),
            "full_regression_drift": regression_binding_drift,
            "plan_a_visual_qa_bindings_current": (
                visual_report_bindings_current
            ),
            "plan_a_visual_qa_drift": visual_report_binding_drift,
            "plan_a_submission_index_bindings_current": (
                plan_a_index_bindings_current
            ),
            "plan_a_submission_index_drift": (
                plan_a_index_binding_drift
            ),
            "plan_a_archive_verification_valid": (
                plan_a_archive_verification.get("valid") is True
            ),
            "plan_a_archive_bindings_current": (
                plan_a_archive_bindings_current
            ),
            "plan_a_training_metadata_current": (
                plan_a_training_metadata_current
            ),
            "plan_a_training_metadata_file_count": len(
                required_training_metadata_paths
            ),
        },
        "independent_claim_gates": {
            "complete_patent_replication_ready": (
                complete_patent_replication_ready
            ),
            "hardware_validation_ready": hardware_validation_ready,
            "physical_record_count": physical.get("physical_record_count"),
            "external_blind_result_ready": external.get(
                "formal_external_result_available"
            )
            is True,
            "external_deployment_ready": external_deployment_ready,
            "rgbd_ready": physical.get("rgbd_ready") is True,
        },
        "superseded_artifact_state": {
            "legacy_paper_readiness_predates_governance_v14": (
                mandatory["paper_readiness"].stat().st_mtime_ns
                < mandatory["governance_current"].stat().st_mtime_ns
            ),
            "legacy_submission_index_current": False,
            "legacy_evidence_archive_current": False,
            "plan_a_submission_index_current": submission_index_current,
            "plan_a_evidence_archive_current": evidence_archive_current,
            "policy": (
                "保留旧产物不删除；方案A使用独立索引和独立证据归档。"
            ),
        },
        "claim_policy": {
            "allowed_now": [
                "现有Qwen软传感器内部论文稿和基础证据已完成审计。",
                "专利相关多尺度、文本、4x4网格与融合结构可在差异边界内讨论。",
            ],
            "not_allowed_now": [
                "方案A确认性修订和最终交付已经完成。",
                "完整复现原专利。",
                "已完成实体硬件、RGB-D或外部部署验证。",
            ],
        },
        "source_bindings": {
            **{name: _binding(path) for name, path in mandatory.items()},
            **{
                name: _binding(path)
                for name, path in optional.items()
                if path.is_file()
            },
            "builder_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# 项目总完成审计 v1",
        "",
        f"- Qwen方案A论文目标完成：`{str(qwen_paper_goal_ready).lower()}`。",
        f"- 未完成内部交付门：{len(blockers)}。",
        (
            "- 完整专利复现："
            f"`{str(complete_patent_replication_ready).lower()}`。"
        ),
        f"- 实体硬件验证：`{str(hardware_validation_ready).lower()}`。",
        f"- 外部部署证据：`{str(external_deployment_ready).lower()}`。",
        "",
        "## Qwen论文门",
        "",
        "| Gate | State |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{name}` | `{str(gates[name]).lower()}` |"
        for name in QWEN_PAPER_GATES
    )
    lines.extend(["", "## 当前阻塞", ""])
    if blockers:
        lines.extend(
            f"- `{item['gate']}`：{item['action']}" for item in blockers
        )
    else:
        lines.append("- 无。")
    lines.extend(
        [
            "",
            "硬件、RGB-D和外部盲测为独立声明门，不得由内部Qwen论文门外推。",
            "",
        ]
    )
    _atomic_text(output.with_suffix(".md"), "\n".join(lines))
    return report


def main() -> None:
    report = build_completion_audit()
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "qwen_paper_goal_ready": report["qwen_paper_goal_ready"],
                "blocking_count": report["blocking_count"],
                "independent_claim_gates": report[
                    "independent_claim_gates"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
