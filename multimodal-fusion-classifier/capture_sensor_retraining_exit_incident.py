"""Capture the rule-v1 trainer crash after complete artifact persistence."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "training"
    / "interrupted"
    / "sensor_source_retraining_exit_access_violation_20260728"
)
DEFAULT_FILES = {
    "protocol": (
        "reports/qwen-material/sensor-source-retraining/PROTOCOL.json"
    ),
    "resolved_config": (
        "reports/qwen-material/sensor-source-retraining/"
        "resolved_rule_v1_config.json"
    ),
    "queue_source": (
        "reports/qwen-material/training/run_sensor_source_retraining.ps1"
    ),
    "recovery_queue_source": (
        "reports/qwen-material/training/"
        "run_sensor_retraining_post_exit_recovery.ps1"
    ),
    "archived_failed_status": (
        "reports/qwen-material/training/interrupted/"
        "sensor_source_retraining_exit_access_violation_20260728/"
        "sensor_source_retraining.failed.status.json"
    ),
    "archived_continuation_status": (
        "reports/qwen-material/training/interrupted/"
        "sensor_source_retraining_exit_access_violation_20260728/"
        "source_repair_continuation.failed.status.json"
    ),
    "archived_training_stdout": (
        "reports/qwen-material/training/interrupted/"
        "sensor_source_retraining_exit_access_violation_20260728/"
        "training.stdout.log"
    ),
    "archived_training_stderr": (
        "reports/qwen-material/training/interrupted/"
        "sensor_source_retraining_exit_access_violation_20260728/"
        "training.stderr.log"
    ),
    "archived_history": (
        "reports/qwen-material/training/interrupted/"
        "sensor_source_retraining_exit_access_violation_20260728/"
        "history.json"
    ),
    "archived_training_summary": (
        "reports/qwen-material/training/interrupted/"
        "sensor_source_retraining_exit_access_violation_20260728/"
        "training_summary.json"
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
    "train_source": "train.py",
    "training_source": "mmfc/training.py",
    "model_source": "mmfc/model.py",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SENSOR_RETRAINING_EXIT_INCIDENT.json"
)
EXIT_CODE = -1073741819
EXIT_CODE_HEX = "0xC0000005"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the post-completion rule-v1 trainer exit"
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


def _protocol_hash(protocol: Mapping[str, Any], relative_path: str) -> str:
    for entry in protocol.get("input_hashes", []):
        if Path(str(entry["path"])).as_posix() == relative_path:
            return str(entry["sha256"]).lower()
    return ""


def build_sensor_retraining_exit_incident(
    *,
    files: Mapping[str, Path],
    archive_root: Path,
    downstream_report_root: Path,
    comparison_path: Path,
) -> dict[str, Any]:
    missing_keys = sorted(set(DEFAULT_FILES) - set(files))
    if missing_keys:
        raise FileNotFoundError(
            "sensor retraining incident source map is incomplete: "
            + ", ".join(missing_keys)
        )
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "sensor retraining incident inputs are missing: "
            + ", ".join(sorted(missing))
        )
    archive = _resolve(archive_root)
    if not archive.is_dir():
        raise FileNotFoundError(f"incident archive is missing: {archive}")

    protocol = _read_json(resolved["protocol"])
    status = _read_json(resolved["archived_failed_status"])
    continuation = _read_json(resolved["archived_continuation_status"])
    history = _read_json(resolved["archived_history"])
    summary = _read_json(resolved["archived_training_summary"])
    checkpoint = _read_json(resolved["post_exit_checkpoint_audit"])
    queue = resolved["queue_source"].read_text(encoding="utf-8-sig")
    recovery = resolved["recovery_queue_source"].read_text(
        encoding="utf-8-sig"
    )
    history_rows = list(history.get("history", []))
    summary_rows = list(summary.get("history", []))
    errors: list[str] = []

    if (
        protocol.get("status") != "frozen_before_rule_v1_retraining"
        or protocol.get("protocol")
        != "independent_sensor_source_retraining"
    ):
        errors.append("sensor-source retraining protocol is not frozen")
    if (
        status.get("state") != "failed"
        or str(EXIT_CODE) not in str(status.get("message", ""))
        or continuation.get("state") != "failed"
        or "sensor_source_retraining failed" not in str(
            continuation.get("message", "")
        )
    ):
        errors.append("archived failure statuses do not bind the exit")
    if resolved["archived_training_stderr"].stat().st_size != 0:
        errors.append("trainer stderr is not empty")
    if (
        summary.get("completed_epochs") != 100
        or summary.get("checkpoints_saved") is not True
        or len(history_rows) != 100
        or len(summary_rows) != 100
        or history_rows[-1].get("epoch") != 100
        or summary_rows[-1].get("epoch") != 100
    ):
        errors.append("training artifacts do not contain 100 complete epochs")
    if (
        checkpoint.get("valid") is not True
        or checkpoint.get("training_summary_completed_epochs") != 100
        or checkpoint.get("history_json_rows") != 100
        or checkpoint.get("best", {}).get("epoch") != 91
        or checkpoint.get("last", {}).get("epoch") != 100
        or checkpoint.get("best", {}).get("sha256")
        != _sha256(resolved["best_checkpoint"])
        or checkpoint.get("last", {}).get("sha256")
        != _sha256(resolved["last_checkpoint"])
    ):
        errors.append("post-exit checkpoint validation mismatch")
    if (
        _protocol_hash(
            protocol,
            "reports/qwen-material/training/"
            "run_sensor_source_retraining.ps1",
        )
        != _sha256(resolved["queue_source"])
        or _protocol_hash(protocol, "train.py")
        != _sha256(resolved["train_source"])
        or _protocol_hash(protocol, "mmfc/training.py")
        != _sha256(resolved["training_source"])
        or _protocol_hash(protocol, "mmfc/model.py")
        != _sha256(resolved["model_source"])
    ):
        errors.append("frozen training source hashes mismatch")
    if (
        "SKIP rule_v1_training (verified complete)" not in queue
        or "Test-TrainingComplete" not in queue
        or "run_sensor_source_retraining.ps1" not in recovery
        or "audit_sensor_retraining_exit_outcome.py" not in recovery
        or "run_final_evidence_audit.ps1" not in recovery
    ):
        errors.append("recovery queue does not preserve the frozen skip path")

    downstream_root = _resolve(downstream_report_root)
    comparison = _resolve(comparison_path)
    if (
        downstream_root.exists() and any(downstream_root.rglob("*"))
    ) or comparison.exists():
        errors.append("downstream rule-v1 evidence existed before recovery")

    archived_files = sorted(path for path in archive.rglob("*") if path.is_file())
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "rule_v1_training_post_completion_process_exit",
        "captured_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "failure": {
            "stage": "rule_v1_training_process_exit",
            "exit_code": EXIT_CODE,
            "exit_code_hex": EXIT_CODE_HEX,
            "python_traceback": False,
            "stderr_empty": (
                resolved["archived_training_stderr"].stat().st_size == 0
            ),
            "failure_after_training_summary_persisted": True,
        },
        "completed_training": {
            "completed_epochs": summary.get("completed_epochs"),
            "history_rows": len(history_rows),
            "checkpoints_saved": summary.get("checkpoints_saved"),
            "best_epoch": checkpoint.get("best", {}).get("epoch"),
            "last_epoch": checkpoint.get("last", {}).get("epoch"),
            "config_fingerprint": checkpoint.get("expected", {}).get(
                "config_fingerprint"
            ),
            "best_checkpoint": _binding(resolved["best_checkpoint"]),
            "last_checkpoint": _binding(resolved["last_checkpoint"]),
            "post_exit_checkpoint_audit": _binding(
                resolved["post_exit_checkpoint_audit"]
            ),
        },
        "recovery": {
            "decision": (
                "reuse_valid_complete_training_bundle_and_skip_retraining"
            ),
            "queue_behavior": (
                "rerun_original_frozen_queue_via_Test-TrainingComplete"
            ),
            "downstream_evidence_absent_before_recovery": not (
                (downstream_root.exists() and any(downstream_root.rglob("*")))
                or comparison.exists()
            ),
            "recovery_queue": _binding(
                resolved["recovery_queue_source"]
            ),
        },
        "source_bindings": {
            name: _binding(path)
            for name, path in sorted(resolved.items())
        },
        "archive": {
            "path": _display_path(archive),
            "file_count": len(archived_files),
            "bindings": {
                path.relative_to(archive).as_posix(): _binding(path)
                for path in archived_files
            },
        },
        "errors": errors,
        "boundaries": [
            (
                "The nonzero Windows process exit occurred after all 100 "
                "epoch records, both checkpoints, and the complete training "
                "summary were persisted."
            ),
            (
                "The completed training bundle is reused only after an "
                "independent checkpoint audit loads best.pt and last.pt and "
                "verifies their epochs, schemas, fingerprints, and hashes."
            ),
            (
                "Recovery uses the original frozen queue's verified skip "
                "branch; it does not resume, warm-start, or retrain the model."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    training = report["completed_training"]
    return "\n".join(
        [
            "# rule-v1 重训完成后进程退出事件",
            "",
            f"捕获时间：`{report['captured_at']}`。",
            "",
            (
                "训练子进程在完整写出 100 轮 history、training summary、best.pt "
                "和第 100 轮 last.pt 后，以 Windows 访问冲突类退出码 "
                f"`{report['failure']['exit_code_hex']}` 结束；stderr 为空且没有 "
                "Python traceback。"
            ),
            (
                "独立检查点审计已成功加载两份权重，best 位于第 "
                f"{training['best_epoch']} 轮，last 位于第 "
                f"{training['last_epoch']} 轮，配置指纹和状态结构一致。"
            ),
            (
                "恢复只重跑原冻结队列并触发其 `Test-TrainingComplete` 跳过分支，"
                "继续检查点审计、固定 test 与配对统计；不恢复、不暖启动、也不"
                "重新训练。"
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
            f"sensor retraining exit incident is immutable: {output}"
        )
    report = build_sensor_retraining_exit_incident(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        },
        archive_root=ARCHIVE_ROOT,
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
                "completed_epochs": report["completed_training"][
                    "completed_epochs"
                ],
                "best_epoch": report["completed_training"]["best_epoch"],
                "last_epoch": report["completed_training"]["last_epoch"],
                "archive_file_count": report["archive"]["file_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
