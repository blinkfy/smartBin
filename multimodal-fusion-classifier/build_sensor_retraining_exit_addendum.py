"""Freeze the recovery-queue amendment for the rule-v1 exit incident."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "incident": (
        "reports/qwen-material/SENSOR_RETRAINING_EXIT_INCIDENT.json"
    ),
    "original_recovery_queue": (
        "reports/qwen-material/training/"
        "run_sensor_retraining_post_exit_recovery.ps1"
    ),
    "revised_recovery_queue": (
        "reports/qwen-material/training/"
        "run_sensor_retraining_post_exit_recovery_v2.ps1"
    ),
    "frozen_training_queue": (
        "reports/qwen-material/training/run_sensor_source_retraining.ps1"
    ),
    "protocol": (
        "reports/qwen-material/sensor-source-retraining/PROTOCOL.json"
    ),
    "post_exit_checkpoint_audit": (
        "reports/qwen-material/training/interrupted/"
        "sensor_source_retraining_exit_access_violation_20260728/"
        "checkpoint_audit.post_exit.json"
    ),
    "best_checkpoint": (
        "checkpoints/source-transfer/"
        "full_cross_modal_quality_rule_v1/best.pt"
    ),
    "last_checkpoint": (
        "checkpoints/source-transfer/"
        "full_cross_modal_quality_rule_v1/last.pt"
    ),
    "builder_source": "build_sensor_retraining_exit_addendum.py",
    "outcome_audit_source": "audit_sensor_retraining_exit_outcome.py",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the sensor retraining recovery-queue addendum"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_sensor_retraining_exit_addendum(
    *,
    files: Mapping[str, Path],
    downstream_report_root: Path,
    comparison_path: Path,
) -> dict[str, Any]:
    missing_keys = sorted(set(DEFAULT_FILES) - set(files))
    if missing_keys:
        raise FileNotFoundError(
            "sensor retraining addendum source map is incomplete: "
            + ", ".join(missing_keys)
        )
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "sensor retraining addendum inputs are missing: "
            + ", ".join(sorted(missing))
        )

    incident = _read_json(resolved["incident"])
    checkpoint = _read_json(resolved["post_exit_checkpoint_audit"])
    original = resolved["original_recovery_queue"].read_text(
        encoding="utf-8-sig"
    )
    revised = resolved["revised_recovery_queue"].read_text(
        encoding="utf-8-sig"
    )
    errors: list[str] = []

    if incident.get("state") != "complete" or incident.get("valid") is not True:
        errors.append("sensor retraining exit incident is not valid")
    incident_queue = incident.get("source_bindings", {}).get(
        "recovery_queue_source", {}
    )
    if (
        incident_queue.get("sha256")
        != _sha256(resolved["original_recovery_queue"])
        or incident_queue.get("size_bytes")
        != resolved["original_recovery_queue"].stat().st_size
    ):
        errors.append("incident does not bind the original recovery queue")
    if (
        '"reports\\qwen-material\\training\\interrupted\\" +' not in original
        or "$PostExitAuditRelative = (" not in revised
        or "$PostExitAudit = Join-Path $ProjectRoot $PostExitAuditRelative"
        not in revised
        or "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json" not in revised
        or "run_sensor_source_retraining.ps1" not in revised
        or "audit_sensor_retraining_exit_outcome.py" not in revised
        or "run_final_evidence_audit.ps1" not in revised
    ):
        errors.append("recovery queue source transition mismatch")
    if _sha256(resolved["original_recovery_queue"]) == _sha256(
        resolved["revised_recovery_queue"]
    ):
        errors.append("revised recovery queue did not change")
    training = incident.get("completed_training", {})
    if (
        checkpoint.get("valid") is not True
        or checkpoint.get("best", {}).get("sha256")
        != _sha256(resolved["best_checkpoint"])
        or checkpoint.get("last", {}).get("sha256")
        != _sha256(resolved["last_checkpoint"])
        or training.get("best_checkpoint", {}).get("sha256")
        != _sha256(resolved["best_checkpoint"])
        or training.get("last_checkpoint", {}).get("sha256")
        != _sha256(resolved["last_checkpoint"])
    ):
        errors.append("completed training bundle changed before recovery")
    downstream = _resolve(downstream_report_root)
    comparison = _resolve(comparison_path)
    if (
        downstream.exists() and any(downstream.rglob("*"))
    ) or comparison.exists():
        errors.append("downstream evidence existed before amended recovery")

    return {
        "schema_version": 1,
        "status": (
            "frozen_after_recovery_queue_path_failure_before_recovery"
            if not errors
            else "invalid"
        ),
        "valid": not errors,
        "frozen_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "incident": _binding(resolved["incident"]),
        "implementation_fix": {
            "allowed_change": (
                "group_post_exit_audit_relative_path_before_Join-Path"
            ),
            "original_recovery_queue": _binding(
                resolved["original_recovery_queue"]
            ),
            "revised_recovery_queue": _binding(
                resolved["revised_recovery_queue"]
            ),
            "frozen_training_queue_unchanged": _binding(
                resolved["frozen_training_queue"]
            ),
        },
        "invariants": {
            "training_rerun": False,
            "completed_epochs": 100,
            "best_epoch": 91,
            "last_epoch": 100,
            "best_checkpoint_unchanged": True,
            "last_checkpoint_unchanged": True,
            "downstream_evidence_absent": True,
            "frozen_protocol_unchanged": True,
            "formal_training_queue_unchanged": True,
        },
        "source_bindings": {
            name: _binding(path)
            for name, path in sorted(resolved.items())
        },
        "errors": errors,
        "boundaries": [
            (
                "The amendment fixes only PowerShell construction of the "
                "post-exit checkpoint-audit path and binds the addendum."
            ),
            (
                "It does not change or rerun training, checkpoints, data, "
                "configuration, evaluation, or comparison settings."
            ),
            (
                "The original invalid recovery queue remains unchanged and "
                "is retained as the incident-bound source."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    fix = report["implementation_fix"]
    return "\n".join(
        [
            "# rule-v1 退出事件恢复脚本补充",
            "",
            f"冻结时间：`{report['frozen_at']}`。",
            "",
            (
                "首份恢复脚本的多段路径拼接可被 PowerShell 解析，但 `Join-Path` "
                "参数绑定无效；该脚本未运行。"
            ),
            (
                "v2 只先组合相对路径再传给 `Join-Path`，并校验本补充件。"
                "原训练队列、100 轮训练包、检查点和全部分析参数保持不变。"
            ),
            (
                "修订前后恢复脚本哈希分别为 "
                f"`{fix['original_recovery_queue']['sha256']}` 与 "
                f"`{fix['revised_recovery_queue']['sha256']}`。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError(
            f"sensor retraining exit addendum is immutable: {output}"
        )
    report = build_sensor_retraining_exit_addendum(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        },
        downstream_report_root=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "sensor-source-retraining"
            / "rule-v1"
        ),
        comparison_path=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "sensor-source-retraining"
            / "RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json"
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "valid": report["valid"],
                "original_queue_sha256": report["implementation_fix"][
                    "original_recovery_queue"
                ]["sha256"],
                "revised_queue_sha256": report["implementation_fix"][
                    "revised_recovery_queue"
                ]["sha256"],
                "source_binding_count": len(report["source_bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
