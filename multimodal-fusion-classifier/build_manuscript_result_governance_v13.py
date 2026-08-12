"""Record the Plan A mixed-seed baseline reporting amendment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_recent_robust_baselines import (
    AMENDED_METHOD_TRAINING_SEEDS,
    AMENDED_PROTOCOL_STATE,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V13.json"
INPUTS = {
    "governance_v12": REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V12.json",
    "baseline_protocol_amendment": (
        REPORT_ROOT
        / "recent-robust-baselines"
        / "PROTOCOL_AMENDMENT.json"
    ),
    "baseline_outcome": (
        REPORT_ROOT / "recent-robust-baselines" / "OUTCOME.json"
    ),
    "baseline_scope_builder": (
        PROJECT_ROOT / "build_recent_robust_baseline_scope_amendment.py"
    ),
    "baseline_auditor": PROJECT_ROOT / "audit_recent_robust_baselines.py",
    "confirmatory_builder": (
        PROJECT_ROOT / "build_plan_a_confirmatory_addendum.py"
    ),
    "confirmatory_builder_tests": (
        PROJECT_ROOT / "tests" / "test_plan_a_confirmatory_addendum.py"
    ),
    "manuscript_updater": (
        PROJECT_ROOT / "apply_plan_a_confirmatory_manuscript_update.py"
    ),
    "automated_build_auditor": (
        PROJECT_ROOT / "audit_plan_a_automated_paper_build.py"
    ),
    "completion_auditor": (
        PROJECT_ROOT / "build_project_completion_audit_v1.py"
    ),
    "submission_index_builder": (
        PROJECT_ROOT / "build_plan_a_submission_index.py"
    ),
    "evidence_archive_builder": (
        PROJECT_ROOT / "build_evidence_archive.py"
    ),
    "governance_v13_builder": Path(__file__),
}
STATUS_PATHS = {
    "component": (
        REPORT_ROOT
        / "training"
        / "v3_loss_component_ablation_status.json"
    ),
    "unseen": (
        REPORT_ROOT / "training" / "unseen_sensor_conditions_status.json"
    ),
    "recent_baseline": (
        REPORT_ROOT / "training" / "recent_robust_baseline_status.json"
    ),
    "manuscript_refresh": (
        REPORT_ROOT / "training" / "plan_a_manuscript_refresh_status.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manuscript result governance v13"
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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def validate_transition(
    governance_v12: Mapping[str, Any],
    amendment: Mapping[str, Any],
    baseline_outcome: Mapping[str, Any],
) -> None:
    expected = {
        method: list(values)
        for method, values in AMENDED_METHOD_TRAINING_SEEDS.items()
    }
    if (
        governance_v12.get("valid") is not True
        or governance_v12.get("version") != 12
        or governance_v12.get("decision", {}).get("confirmed_option") != "A"
    ):
        raise ValueError("governance v12 identity mismatch")
    if (
        amendment.get("valid") is not True
        or amendment.get("state") != AMENDED_PROTOCOL_STATE
        or amendment.get("method_training_seeds") != expected
        or amendment.get("run_count") != 4
        or amendment.get("scope_reduction", {}).get(
            "single_run_reported_as_replication"
        )
        is not False
        or amendment.get("scope_reduction", {}).get(
            "partial_run_used_for_results"
        )
        is not False
    ):
        raise ValueError("baseline scope amendment identity mismatch")
    if (
        baseline_outcome.get("valid") is not True
        or baseline_outcome.get("state") != "complete"
        or baseline_outcome.get("method_training_seeds") != expected
        or baseline_outcome.get("run_count") != 4
        or baseline_outcome.get("scope")
        != "matched_same_input_qmf_three_seed_tmc_single_seed_reference"
    ):
        raise ValueError("amended baseline outcome identity mismatch")
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


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite governance v13")
    missing = [
        name
        for name, path in {**INPUTS, **STATUS_PATHS}.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "governance v13 inputs missing: " + ", ".join(missing)
        )
    governance_v12 = _read_json(INPUTS["governance_v12"])
    amendment = _read_json(INPUTS["baseline_protocol_amendment"])
    outcome = _read_json(INPUTS["baseline_outcome"])
    validate_transition(governance_v12, amendment, outcome)

    old_bindings = governance_v12.get("source_bindings", {})
    if not isinstance(old_bindings, Mapping):
        raise ValueError("governance v12 source bindings are missing")
    drifted_v12 = sorted(
        name
        for name, binding in old_bindings.items()
        if not isinstance(binding, Mapping) or not _binding_current(binding)
    )
    allowed_drift = {
        "confirmatory_builder",
        "confirmatory_builder_tests",
        "completion_auditor",
        "submission_index_builder",
    }
    if not {"confirmatory_builder", "confirmatory_builder_tests"}.issubset(
        drifted_v12
    ) or not set(drifted_v12).issubset(allowed_drift):
        raise ValueError(
            "governance v12 drift is outside the mixed-seed transition: "
            + ", ".join(drifted_v12)
        )

    statuses = {
        name: _read_json(path) for name, path in STATUS_PATHS.items()
    }
    report = {
        "schema_version": 1,
        "version": 13,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": governance_v12["decision"],
        "current_paper": governance_v12["current_paper"],
        "governance_transition": {
            "supersedes_version": 12,
            "scope": (
                "user_directed_recent_baseline_compute_scope_reduction"
            ),
            "drifted_v12_bindings": drifted_v12,
            "model_or_loss_changed": False,
            "data_or_sample_order_changed": False,
            "completed_run_metrics_rewritten": False,
            "partial_run_used_for_results": False,
            "single_run_reported_as_replication": False,
        },
        "baseline_reporting_contract": {
            "method_training_seeds": amendment["method_training_seeds"],
            "qmf": {
                "seed_count": 3,
                "reported_statistics": ["mean", "sample_standard_deviation"],
                "evidence_level": "confirmatory_multiseed",
            },
            "tmc": {
                "seed_count": 1,
                "reported_statistics": [
                    "point_estimate",
                    "source_image_cluster_bootstrap_interval",
                ],
                "sample_standard_deviation": None,
                "evidence_level": "reference_single_seed",
            },
            "excluded_runs": amendment["excluded_runs"],
        },
        "queued_execution": {
            name: {
                "state": status.get("state"),
                "stage": status.get("stage"),
                "message": status.get("message"),
                "updated_at": status.get("updated_at"),
                "mutable_status_path": path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
            }
            for (name, path), status in zip(
                STATUS_PATHS.items(), statuses.values()
            )
        },
        "unchanged_evidence_boundaries": (
            governance_v12["unchanged_evidence_boundaries"]
        ),
        "claim_policy": {
            "allowed_now": [
                "QMF同输入适配报告三种子均值与样本标准差。",
                "TMC同输入适配仅报告seed42单种子参考和源图簇区间。",
                "披露TMC seed43在34轮被终止且从全部统计中排除。",
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
                "# 论文结果治理 v13：近期基线计算范围修订",
                "",
                "- 方案A保持不变；此次只缩减尚未完成的TMC重复训练。",
                "- QMF：seeds 42/43/44，n=3，报告均值和样本标准差。",
                "- TMC：仅seed42，n=1，报告点估计和源图簇区间，SD不适用。",
                "- TMC seed43在34轮终止并隔离；seed44未启动，均不进入统计。",
                "- 禁止把单次TMC运行声称为多次结果。",
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
                "method_training_seeds": report[
                    "baseline_reporting_contract"
                ]["method_training_seeds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
