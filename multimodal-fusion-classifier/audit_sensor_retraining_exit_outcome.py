"""Audit recovery from the rule-v1 trainer's post-completion exit."""

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
    "incident_addendum": (
        "reports/qwen-material/"
        "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json"
    ),
    "protocol": (
        "reports/qwen-material/sensor-source-retraining/PROTOCOL.json"
    ),
    "queue_source": (
        "reports/qwen-material/training/run_sensor_source_retraining.ps1"
    ),
    "recovery_queue_source": (
        "reports/qwen-material/training/"
        "run_sensor_retraining_post_exit_recovery_v2.ps1"
    ),
    "inner_status": (
        "reports/qwen-material/training/"
        "sensor_source_retraining.status.json"
    ),
    "inner_pipeline_log": (
        "reports/qwen-material/training/sensor_source_retraining.log"
    ),
    "training_stdout": (
        "reports/qwen-material/training/"
        "sensor_source_retraining.train.stdout.log"
    ),
    "archived_training_stdout": (
        "reports/qwen-material/training/interrupted/"
        "sensor_source_retraining_exit_access_violation_20260728/"
        "training.stdout.log"
    ),
    "history": (
        "checkpoints/source-transfer/"
        "full_cross_modal_quality_rule_v1/history.json"
    ),
    "training_summary": (
        "checkpoints/source-transfer/"
        "full_cross_modal_quality_rule_v1/training_summary.json"
    ),
    "best_checkpoint": (
        "checkpoints/source-transfer/"
        "full_cross_modal_quality_rule_v1/best.pt"
    ),
    "last_checkpoint": (
        "checkpoints/source-transfer/"
        "full_cross_modal_quality_rule_v1/last.pt"
    ),
    "formal_checkpoint_audit": (
        "reports/qwen-material/sensor-source-retraining/"
        "rule-v1/CHECKPOINT_AUDIT.json"
    ),
    "evaluation": (
        "reports/qwen-material/sensor-source-retraining/"
        "rule-v1/test/evaluation.json"
    ),
    "predictions": (
        "reports/qwen-material/sensor-source-retraining/"
        "rule-v1/test/predictions.jsonl"
    ),
    "comparison": (
        "reports/qwen-material/sensor-source-retraining/"
        "RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json"
    ),
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SENSOR_RETRAINING_EXIT_OUTCOME.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the post-completion rule-v1 recovery"
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


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def build_sensor_retraining_exit_outcome(
    *,
    files: Mapping[str, Path],
) -> dict[str, Any]:
    missing_keys = sorted(set(DEFAULT_FILES) - set(files))
    if missing_keys:
        raise FileNotFoundError(
            "sensor retraining outcome source map is incomplete: "
            + ", ".join(missing_keys)
        )
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "sensor retraining outcome inputs are missing: "
            + ", ".join(sorted(missing))
        )

    incident = _read_json(resolved["incident"])
    addendum = _read_json(resolved["incident_addendum"])
    status = _read_json(resolved["inner_status"])
    history = _read_json(resolved["history"])
    summary = _read_json(resolved["training_summary"])
    checkpoint = _read_json(resolved["formal_checkpoint_audit"])
    evaluation = _read_json(resolved["evaluation"])
    comparison = _read_json(resolved["comparison"])
    pipeline_log = resolved["inner_pipeline_log"].read_text(
        encoding="utf-8-sig"
    )
    errors: list[str] = []

    if (
        incident.get("state") != "complete"
        or incident.get("valid") is not True
        or incident.get("recovery", {}).get("decision")
        != "reuse_valid_complete_training_bundle_and_skip_retraining"
    ):
        errors.append("post-completion exit incident is not valid")
    if (
        addendum.get("status")
        != "frozen_after_recovery_queue_path_failure_before_recovery"
        or addendum.get("valid") is not True
        or addendum.get("incident", {}).get("sha256")
        != _sha256(resolved["incident"])
        or addendum.get("implementation_fix", {})
        .get("revised_recovery_queue", {})
        .get("sha256")
        != _sha256(resolved["recovery_queue_source"])
    ):
        errors.append("post-completion exit incident addendum is not valid")
    if status.get("state") != "complete" or status.get("stage") != "all":
        errors.append("recovered sensor-source pipeline is not complete")
    if "SKIP rule_v1_training (verified complete)" not in pipeline_log:
        errors.append("recovery log does not prove the training skip")

    incident_training = incident.get("completed_training", {})
    best_sha = _sha256(resolved["best_checkpoint"])
    last_sha = _sha256(resolved["last_checkpoint"])
    if (
        checkpoint.get("valid") is not True
        or checkpoint.get("training_summary_completed_epochs") != 100
        or checkpoint.get("history_json_rows") != 100
        or checkpoint.get("best", {}).get("epoch") != 91
        or checkpoint.get("last", {}).get("epoch") != 100
        or checkpoint.get("best", {}).get("sha256") != best_sha
        or checkpoint.get("last", {}).get("sha256") != last_sha
        or incident_training.get("best_checkpoint", {}).get("sha256")
        != best_sha
        or incident_training.get("last_checkpoint", {}).get("sha256")
        != last_sha
    ):
        errors.append("recovered checkpoint identity mismatch")
    if (
        summary.get("completed_epochs") != 100
        or summary.get("checkpoints_saved") is not True
        or len(history.get("history", [])) != 100
        or len(summary.get("history", [])) != 100
    ):
        errors.append("recovered training metadata mismatch")
    archived_summary = incident.get("source_bindings", {}).get(
        "archived_training_summary", {}
    )
    archived_history = incident.get("source_bindings", {}).get(
        "archived_history", {}
    )
    if (
        archived_summary.get("sha256")
        != _sha256(resolved["training_summary"])
        or archived_history.get("sha256") != _sha256(resolved["history"])
        or _sha256(resolved["training_stdout"])
        != _sha256(resolved["archived_training_stdout"])
    ):
        errors.append("training artifacts changed during skip recovery")

    metrics = evaluation.get("metrics", {})
    evaluation_config = evaluation.get("evaluation_config", {})
    if (
        evaluation.get("dataset", {}).get("split") != "test"
        or metrics.get("sample_count") != 826
        or evaluation_config.get("bootstrap_repeats") != 2000
        or _jsonl_count(resolved["predictions"]) != 826
    ):
        errors.append("rule-v1 fixed-test evaluation mismatch")
    if (
        comparison.get("reference", {}).get("name") != "qwen_retrained"
        or comparison.get("candidate", {}).get("name")
        != "rule_v1_retrained"
        or comparison.get("paired_bootstrap", {}).get("repeats") != 5000
        or comparison.get("paired_bootstrap", {}).get("sample_count") != 826
        or comparison.get("paired_bootstrap", {}).get("cluster_count") != 570
    ):
        errors.append("rule-v1 versus Qwen comparison mismatch")

    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "rule_v1_post_completion_exit_recovery_outcome",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "incident": _binding(resolved["incident"]),
        "recovery": {
            "training_rerun": False,
            "training_skip_verified": (
                "SKIP rule_v1_training (verified complete)" in pipeline_log
            ),
            "completed_epochs": summary.get("completed_epochs"),
            "best_epoch": checkpoint.get("best", {}).get("epoch"),
            "last_epoch": checkpoint.get("last", {}).get("epoch"),
            "best_checkpoint_sha256": best_sha,
            "last_checkpoint_sha256": last_sha,
            "test_sample_count": metrics.get("sample_count"),
            "paired_bootstrap_repeats": comparison.get(
                "paired_bootstrap", {}
            ).get("repeats"),
        },
        "source_bindings": {
            name: _binding(path)
            for name, path in sorted(resolved.items())
        },
        "errors": errors,
        "boundaries": [
            (
                "The recovery did not launch train.py because the original "
                "frozen queue verified the complete 100-epoch artifact set."
            ),
            (
                "Checkpoint identity and training metadata remain byte-for-"
                "byte equal to the incident snapshot."
            ),
            (
                "Only checkpoint audit, fixed-test evaluation, and paired "
                "comparison were executed after the process-exit incident."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    recovery = report["recovery"]
    return "\n".join(
        [
            "# rule-v1 重训完成后退出事件结案",
            "",
            f"生成时间：`{report['generated_at']}`。",
            "",
            (
                "恢复队列确认 100 轮训练包完整后触发原冻结脚本的训练跳过分支，"
                "没有再次启动 train.py。"
            ),
            (
                f"best/last 分别绑定第 {recovery['best_epoch']} 与 "
                f"{recovery['last_epoch']} 轮；随后完成固定 test 和 "
                f"{recovery['paired_bootstrap_repeats']} 次源图聚类配对 "
                "bootstrap。"
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
            f"sensor retraining exit outcome is immutable: {output}"
        )
    report = build_sensor_retraining_exit_outcome(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        }
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
                **report["recovery"],
                "source_binding_count": len(report["source_bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
