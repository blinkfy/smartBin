"""Record the queued Plan A manuscript-refresh chain without inventing results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V11.json"
INPUTS = {
    "governance_v10": REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V10.json",
    "confirmatory_builder": PROJECT_ROOT
    / "build_plan_a_confirmatory_addendum.py",
    "manuscript_updater": PROJECT_ROOT
    / "apply_plan_a_confirmatory_manuscript_update.py",
    "automated_build_auditor": PROJECT_ROOT
    / "audit_plan_a_automated_paper_build.py",
    "refresh_watcher": REPORT_ROOT
    / "training"
    / "run_plan_a_manuscript_refresh_after_baselines.ps1",
    "final_bibliography": PROJECT_ROOT
    / "paper_rewriting_output"
    / "final_paper"
    / "references.bib",
}
STATUS_PATHS = {
    "component": REPORT_ROOT
    / "training"
    / "v3_loss_component_ablation_status.json",
    "unseen": REPORT_ROOT
    / "training"
    / "unseen_sensor_conditions_status.json",
    "recent_baseline": REPORT_ROOT
    / "training"
    / "recent_robust_baseline_status.json",
    "manuscript_refresh": REPORT_ROOT
    / "training"
    / "plan_a_manuscript_refresh_status.json",
}
ADMISSIBLE_STATUS = {
    "waiting",
    "running",
    "complete",
    "awaiting_visual_qa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the option-A manuscript governance v11 addendum"
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
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def validate_execution_state(
    governance: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, Any]],
) -> None:
    if (
        governance.get("valid") is not True
        or governance.get("version") != 10
        or governance.get("decision", {}).get("confirmed_option") != "A"
        or governance.get("recent_baseline_extension", {}).get(
            "result_available"
        )
        is not False
    ):
        raise ValueError("governance v10 identity mismatch")
    missing = set(STATUS_PATHS) - set(statuses)
    if missing:
        raise ValueError("manuscript refresh statuses missing")
    invalid = {
        name: status.get("state")
        for name, status in statuses.items()
        if status.get("state") not in ADMISSIBLE_STATUS
    }
    if invalid:
        raise ValueError(f"inadmissible queued execution state: {invalid}")
    refresh = statuses["manuscript_refresh"]
    if (
        refresh.get("state") == "waiting"
        and refresh.get("stage") != "recent_robust_baseline_outcome"
    ):
        raise ValueError("manuscript refresh waiting-stage mismatch")


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite governance v11")
    missing_inputs = [
        name for name, path in INPUTS.items() if not path.is_file()
    ]
    missing_statuses = [
        name for name, path in STATUS_PATHS.items() if not path.is_file()
    ]
    if missing_inputs or missing_statuses:
        raise FileNotFoundError(
            "governance v11 inputs missing: "
            + ", ".join(missing_inputs + missing_statuses)
        )
    governance = _read_json(INPUTS["governance_v10"])
    statuses = {
        name: _read_json(path) for name, path in STATUS_PATHS.items()
    }
    validate_execution_state(governance, statuses)
    report = {
        "schema_version": 1,
        "version": 11,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": governance["decision"],
        "current_paper": {
            **governance["paper_artifact_state"],
            "confirmatory_revision_applied": False,
            "reason": "three queued confirmatory outcome audits are pending",
        },
        "queued_execution": {
            name: {
                "state": status.get("state"),
                "stage": status.get("stage"),
                "updated_at": status.get("updated_at"),
                "mutable_status_path": STATUS_PATHS[name]
                .relative_to(PROJECT_ROOT)
                .as_posix(),
            }
            for name, status in statuses.items()
        },
        "automatic_completion_contract": {
            "result_available_now": False,
            "canonical_paper_updated_now": False,
            "sequence": [
                "audit both three-seed auxiliary-loss component ablations",
                "evaluate full v3 under all eleven frozen unseen conditions",
                "train and audit QMF/TMC adaptations for seeds 42/43/44",
                "build complete neutral tables and a bound TeX addendum",
                "snapshot delivery artifacts and update canonical main.tex",
                "rebuild and machine-audit DOCX/PDF",
                "stop in awaiting_visual_qa state",
            ],
            "automatic_terminal_state": (
                "automated_build_complete_visual_qa_pending"
            ),
            "not_automatic": [
                "page-by-page visual inspection",
                "PaperSpine final-audit refresh",
                "final artifact-manifest and evidence-archive refresh",
            ],
        },
        "unchanged_external_gates": [
            "independent external blind data",
            "synchronized physical 4x4 sensor acquisition",
            "RGB-D input if complete patent reproduction is claimed",
        ],
        "claim_policy": {
            "allowed_now": [
                "方案A母稿已通过既有PaperSpine门，但新增确认性修订仍在排队。",
                "三条实验和一条论文重建链均已实现并自动接力。",
            ],
            "not_allowed_now": [
                "声称组件、未见条件或QMF/TMC已有正式结果。",
                "声称新增确认性修订后的Word/PDF已完成视觉复核。",
                "声称已有外部盲测、硬件或RGB-D验证。",
            ],
        },
        "source_bindings": {
            name: _binding(path) for name, path in INPUTS.items()
        },
        "errors": [],
    }
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        markdown,
        "\n".join(
            [
                "# 论文结果治理 v11：方案A自动结案链",
                "",
                "- 既有方案A母稿仍有效；新增确认性修订尚未写入规范稿。",
                "- 组件消融、11条未见条件、QMF/TMC直接基线和论文重建已串行接力。",
                "- 当前没有新增正式结果，不得提前写入优越性结论。",
                "- 自动链最终停在`awaiting_visual_qa`，逐页视觉复核仍由人工完成。",
                "",
                "## 自动顺序",
                "",
                "1. 两个辅助监督组件各三种子训练、评估与审计。",
                "2. 完整v3三种子在11种冻结条件下评估。",
                "3. QMF/TMC各三种子训练、clean/11条件评估与配对审计。",
                "4. 生成完整表格和LaTeX增量，快照并更新规范母稿。",
                "5. 重建Word/PDF并执行机器内容审计。",
                "6. 停在逐页视觉复核前，再刷新PaperSpine总关卡和交付清单。",
                "",
                "外部盲测、实体传感器与RGB-D边界保持不变。",
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
                "result_available_now": report[
                    "automatic_completion_contract"
                ]["result_available_now"],
                "manuscript_refresh_state": report["queued_execution"][
                    "manuscript_refresh"
                ]["state"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
