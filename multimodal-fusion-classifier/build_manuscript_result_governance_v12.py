"""Record the audited Plan A multiseed-reporting governance transition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V12.json"
INPUTS = {
    "governance_v11": REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V11.json",
    "confirmatory_builder": PROJECT_ROOT
    / "build_plan_a_confirmatory_addendum.py",
    "confirmatory_builder_tests": PROJECT_ROOT
    / "tests"
    / "test_plan_a_confirmatory_addendum.py",
    "completion_auditor": PROJECT_ROOT
    / "build_project_completion_audit_v1.py",
    "submission_index_builder": PROJECT_ROOT
    / "build_plan_a_submission_index.py",
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
        description="Build the option-A manuscript governance v12 transition"
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


def _binding_current(binding: Mapping[str, Any]) -> bool:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    path = _resolve(Path(raw_path))
    return (
        path.is_file()
        and path.stat().st_size == binding.get("size_bytes")
        and _sha256(path) == binding.get("sha256")
    )


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def validate_transition(
    governance: Mapping[str, Any],
    statuses: Mapping[str, Mapping[str, Any]],
    drifted_v11_bindings: Sequence[str],
) -> None:
    if (
        governance.get("valid") is not True
        or governance.get("version") != 11
        or governance.get("decision", {}).get("confirmed_option") != "A"
    ):
        raise ValueError("governance v11 identity mismatch")
    if list(drifted_v11_bindings) != ["confirmatory_builder"]:
        raise ValueError("governance v11 transition scope mismatch")
    if set(statuses) != set(STATUS_PATHS):
        raise ValueError("governance v12 status identity mismatch")
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
        raise FileExistsError("refusing to overwrite governance v12")
    required = {**INPUTS, "governance_v12_builder": Path(__file__)}
    missing = [
        name for name, path in {**required, **STATUS_PATHS}.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "governance v12 inputs missing: " + ", ".join(missing)
        )
    governance = _read_json(INPUTS["governance_v11"])
    statuses = {
        name: _read_json(path) for name, path in STATUS_PATHS.items()
    }
    old_bindings = governance.get("source_bindings", {})
    if not isinstance(old_bindings, Mapping):
        raise ValueError("governance v11 source bindings are missing")
    drifted = sorted(
        name
        for name, binding in old_bindings.items()
        if not isinstance(binding, Mapping) or not _binding_current(binding)
    )
    validate_transition(governance, statuses, drifted)
    historical_builder = old_bindings["confirmatory_builder"]
    report = {
        "schema_version": 1,
        "version": 12,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": governance["decision"],
        "current_paper": governance["current_paper"],
        "governance_transition": {
            "supersedes_version": 11,
            "scope": "reporting_only_multiseed_component_summary_upgrade",
            "historical_confirmatory_builder": historical_builder,
            "current_confirmatory_builder": _binding(
                INPUTS["confirmatory_builder"]
            ),
            "training_protocol_changed": False,
            "model_or_loss_changed": False,
            "data_or_seed_order_changed": False,
        },
        "component_reporting_contract": {
            "training_seed_count": 3,
            "reported_statistics": [
                "mean",
                "sample_standard_deviation",
                "full_v3_minus_ablation_mean",
                "full_v3_minus_ablation_sample_standard_deviation",
                "accuracy_positive_seed_count",
                "macro_f1_positive_seed_count",
            ],
            "aggregate_crosscheck_required": True,
            "direction_crosscheck_required": True,
            "silent_aggregate_drift_allowed": False,
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
        "unchanged_evidence_boundaries": [
            "reused internal fixed test split is not external blind evidence",
            "Qwen/rule sidecars are not synchronized physical measurements",
            "software faults are not hardware validation",
            "RGB input is not RGB-D",
        ],
        "claim_policy": {
            "allowed_now": [
                "方案A已经确认，确认性实验仍按冻结队列运行。",
                "组件结果完成后报告三种子均值、样本标准差和方向计数。",
            ],
            "not_allowed_now": [
                "在三条结果审计完成前填写确认性实验数值。",
                "把内部固定测试集、软sidecar或软件故障写成外部硬件证据。",
            ],
        },
        "source_bindings": {
            name: _binding(path) for name, path in required.items()
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
                "# 论文结果治理 v12：组件多种子报告口径",
                "",
                "- 方案A确认状态与外部证据边界保持不变。",
                "- v11保留为历史快照；v12只升级组件消融的结果呈现口径。",
                "- Accuracy、Macro F1及完整v3减消融差值均报告三种子均值与样本标准差。",
                "- 同时报告Accuracy/Macro F1正向种子数，并与逐种子记录交叉核验。",
                "- 训练配置、模型、损失、数据、种子和执行顺序均未改变。",
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
                "transition_scope": report["governance_transition"]["scope"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
