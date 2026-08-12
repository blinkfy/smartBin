"""Freeze the UTF-16 log recovery amendment before outcome retry."""

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
    "first_addendum": (
        "reports/qwen-material/"
        "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json"
    ),
    "encoding_repair": (
        "reports/qwen-material/"
        "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json"
    ),
    "archived_utf16_log": (
        "reports/qwen-material/training/interrupted/"
        "sensor_retraining_outcome_log_encoding_20260728/"
        "inner_pipeline.utf16.log"
    ),
    "repaired_utf8_log": (
        "reports/qwen-material/training/"
        "sensor_source_retraining.log"
    ),
    "v2_recovery_queue": (
        "reports/qwen-material/training/"
        "run_sensor_retraining_post_exit_recovery_v2.ps1"
    ),
    "v3_recovery_queue": (
        "reports/qwen-material/training/"
        "run_sensor_retraining_post_exit_recovery_v3.ps1"
    ),
    "archived_v2_status": (
        "reports/qwen-material/training/interrupted/"
        "sensor_retraining_outcome_log_encoding_20260728/"
        "recovery_v2.failed.status.json"
    ),
    "archived_v2_stderr": (
        "reports/qwen-material/training/interrupted/"
        "sensor_retraining_outcome_log_encoding_20260728/"
        "recovery_v2.outcome.stderr.log"
    ),
    "inner_status": (
        "reports/qwen-material/training/"
        "sensor_source_retraining.status.json"
    ),
    "formal_checkpoint_audit": (
        "reports/qwen-material/sensor-source-retraining/"
        "rule-v1/CHECKPOINT_AUDIT.json"
    ),
    "evaluation": (
        "reports/qwen-material/sensor-source-retraining/"
        "rule-v1/test/evaluation.json"
    ),
    "comparison": (
        "reports/qwen-material/sensor-source-retraining/"
        "RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json"
    ),
    "best_checkpoint": (
        "checkpoints/source-transfer/"
        "full_cross_modal_quality_rule_v1/best.pt"
    ),
    "last_checkpoint": (
        "checkpoints/source-transfer/"
        "full_cross_modal_quality_rule_v1/last.pt"
    ),
    "outcome_audit_source": "audit_sensor_retraining_exit_outcome.py",
    "repair_source": "repair_sensor_retraining_log_encoding.py",
    "builder_source": "build_sensor_retraining_encoding_addendum.py",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the sensor retraining log-encoding addendum"
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


def build_sensor_retraining_encoding_addendum(
    *,
    files: Mapping[str, Path],
    outcome_path: Path,
    strict_output_paths: list[Path],
) -> dict[str, Any]:
    missing_keys = sorted(set(DEFAULT_FILES) - set(files))
    if missing_keys:
        raise FileNotFoundError(
            "sensor log addendum source map is incomplete: "
            + ", ".join(missing_keys)
        )
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "sensor log addendum inputs are missing: "
            + ", ".join(sorted(missing))
        )

    incident = _read_json(resolved["incident"])
    first = _read_json(resolved["first_addendum"])
    repair = _read_json(resolved["encoding_repair"])
    status = _read_json(resolved["archived_v2_status"])
    inner = _read_json(resolved["inner_status"])
    checkpoint = _read_json(resolved["formal_checkpoint_audit"])
    evaluation = _read_json(resolved["evaluation"])
    comparison = _read_json(resolved["comparison"])
    stderr = resolved["archived_v2_stderr"].read_text(
        encoding="utf-8-sig"
    )
    repaired_text = resolved["repaired_utf8_log"].read_text(
        encoding="utf-8"
    )
    v3 = resolved["v3_recovery_queue"].read_text(encoding="utf-8-sig")
    errors: list[str] = []

    if incident.get("valid") is not True or first.get("valid") is not True:
        errors.append("prior sensor retraining incident chain is invalid")
    if (
        first.get("implementation_fix", {})
        .get("revised_recovery_queue", {})
        .get("sha256")
        != _sha256(resolved["v2_recovery_queue"])
    ):
        errors.append("first addendum does not bind the v2 recovery queue")
    if (
        status.get("state") != "failed"
        or "outcome_audit failed" not in str(status.get("message", ""))
        or "UnicodeDecodeError" not in stderr
        or "invalid start byte" not in stderr
    ):
        errors.append("archived v2 outcome failure mismatch")
    if (
        repair.get("state") != "complete"
        or repair.get("valid") is not True
        or repair.get("decoded_text_equal") is not True
        or repair.get("original", {}).get("sha256")
        != _sha256(resolved["archived_utf16_log"])
        or repair.get("repaired", {}).get("sha256")
        != _sha256(resolved["repaired_utf8_log"])
    ):
        errors.append("log encoding repair is not valid")
    if (
        "SKIP rule_v1_training (verified complete)" not in repaired_text
        or "DONE rule_v1_checkpoint_audit" not in repaired_text
        or "DONE rule_v1_test" not in repaired_text
        or "DONE rule_v1_vs_qwen_comparison" not in repaired_text
    ):
        errors.append("repaired log lacks completed downstream stages")
    if inner.get("state") != "complete" or inner.get("stage") != "all":
        errors.append("inner sensor-source pipeline is not complete")
    if (
        checkpoint.get("valid") is not True
        or checkpoint.get("training_summary_completed_epochs") != 100
        or checkpoint.get("best", {}).get("sha256")
        != _sha256(resolved["best_checkpoint"])
        or checkpoint.get("last", {}).get("sha256")
        != _sha256(resolved["last_checkpoint"])
        or evaluation.get("metrics", {}).get("sample_count") != 826
        or comparison.get("paired_bootstrap", {}).get("repeats") != 5000
    ):
        errors.append("completed downstream evidence mismatch")
    if (
        "run_sensor_source_retraining.ps1" in v3
        or "audit_sensor_retraining_exit_outcome.py" not in v3
        or "audit_manuscript_stable_sections.py" not in v3
        or "run_final_evidence_audit.ps1" not in v3
    ):
        errors.append("v3 queue does not start at the outcome audit")
    if _resolve(outcome_path).exists():
        errors.append("sensor retraining outcome existed before v3 retry")
    unexpected_strict = [
        _display_path(_resolve(path))
        for path in strict_output_paths
        if _resolve(path).exists()
    ]
    if unexpected_strict:
        errors.append(
            "strict outputs existed before v3 retry: "
            + ", ".join(unexpected_strict)
        )

    return {
        "schema_version": 1,
        "status": (
            "frozen_after_outcome_log_encoding_failure_before_retry"
            if not errors
            else "invalid"
        ),
        "valid": not errors,
        "frozen_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "incident": _binding(resolved["incident"]),
        "first_addendum": _binding(resolved["first_addendum"]),
        "failure": {
            "stage": "outcome_audit_log_read",
            "exception": "UnicodeDecodeError",
            "cause": "powershell_utf16_log_read_as_utf8",
            "inner_pipeline_complete": True,
            "outcome_created": False,
        },
        "implementation_fix": {
            "allowed_change": (
                "normalize_completed_pipeline_log_utf16_to_utf8"
            ),
            "encoding_repair": _binding(resolved["encoding_repair"]),
            "archived_utf16_log": _binding(
                resolved["archived_utf16_log"]
            ),
            "repaired_utf8_log": _binding(
                resolved["repaired_utf8_log"]
            ),
            "v2_recovery_queue": _binding(
                resolved["v2_recovery_queue"]
            ),
            "v3_recovery_queue": _binding(
                resolved["v3_recovery_queue"]
            ),
            "outcome_audit_source_unchanged": _binding(
                resolved["outcome_audit_source"]
            ),
        },
        "invariants": {
            "training_rerun": False,
            "checkpoint_audit_rerun": False,
            "fixed_test_rerun": False,
            "paired_comparison_rerun": False,
            "decoded_log_text_equal": True,
            "completed_epochs": 100,
            "test_sample_count": 826,
            "paired_bootstrap_repeats": 5000,
            "strict_outputs_absent": True,
        },
        "source_bindings": {
            name: _binding(path)
            for name, path in sorted(resolved.items())
        },
        "errors": errors,
        "boundaries": [
            (
                "The amendment changes only the byte encoding of a completed "
                "PowerShell provenance log; decoded text is identical."
            ),
            (
                "The v3 queue starts at outcome auditing and cannot launch "
                "training, checkpoint audit, fixed-test evaluation, or the "
                "paired comparison again."
            ),
            (
                "Models, checkpoints, metadata, predictions, metrics, data, "
                "and all frozen analytical settings remain unchanged."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    fix = report["implementation_fix"]
    return "\n".join(
        [
            "# rule-v1 结案日志编码补充",
            "",
            f"冻结时间：`{report['frozen_at']}`。",
            "",
            (
                "v2 已完成训练跳过、检查点审计、固定 test 和配对比较，但结案"
                "审计把 PowerShell UTF-16 日志按 UTF-8 读取而退出。"
            ),
            (
                "原日志已逐字节归档，规范化后的 UTF-8 日志保持解码文本完全"
                "一致；v3 从结案审计开始，不再调用任何训练或评估阶段。"
            ),
            (
                "UTF-16/UTF-8 日志哈希分别为 "
                f"`{fix['archived_utf16_log']['sha256']}` 与 "
                f"`{fix['repaired_utf8_log']['sha256']}`。"
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
            f"sensor retraining encoding addendum is immutable: {output}"
        )
    reports = PROJECT_ROOT / "reports" / "qwen-material"
    report = build_sensor_retraining_encoding_addendum(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        },
        outcome_path=reports / "SENSOR_RETRAINING_EXIT_OUTCOME.json",
        strict_output_paths=[
            reports / "FINAL_EVIDENCE_AUDIT.json",
            reports / "PAPER_READINESS_AUDIT.json",
            reports / "paper-tables",
            reports / "FINAL_CLAIM_EVIDENCE.json",
            reports / "SUBMISSION_INDEX.json",
            reports / "submission-evidence",
        ],
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
                "source_binding_count": len(report["source_bindings"]),
                "v3_queue_sha256": report["implementation_fix"][
                    "v3_recovery_queue"
                ]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
