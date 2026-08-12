"""Audit the clean restart outcome after the formal queue interruption."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "snapshot": "reports/qwen-material/FORMAL_QUEUE_INTERRUPTION.json",
    "addendum": (
        "reports/qwen-material/FORMAL_QUEUE_INTERRUPTION_ADDENDUM.json"
    ),
    "current_integrity": (
        "reports/qwen-material/SOURCE_IMAGE_INTEGRITY_CURRENT.json"
    ),
    "config": "configs/ablations/full_no_reconstruction.json",
    "history": "checkpoints/ablations/full_no_reconstruction/history.json",
    "training_summary": (
        "checkpoints/ablations/full_no_reconstruction/training_summary.json"
    ),
    "best_checkpoint": (
        "checkpoints/ablations/full_no_reconstruction/best.pt"
    ),
    "last_checkpoint": (
        "checkpoints/ablations/full_no_reconstruction/last.pt"
    ),
    "checkpoint_audit": (
        "reports/qwen-material/ablations/full_no_reconstruction/"
        "CHECKPOINT_AUDIT.json"
    ),
    "evaluation": (
        "reports/qwen-material/ablations/full_no_reconstruction/"
        "evaluation.json"
    ),
    "predictions": (
        "reports/qwen-material/ablations/full_no_reconstruction/"
        "predictions.jsonl"
    ),
    "restart_stdout": (
        "reports/qwen-material/training/"
        "source_repair_continuation."
        "full_no_reconstruction_restart_v2.stdout.log"
    ),
    "restart_stderr": (
        "reports/qwen-material/training/"
        "source_repair_continuation."
        "full_no_reconstruction_restart_v2.stderr.log"
    ),
    "continuation_script": (
        "reports/qwen-material/training/run_source_repair_continuation.ps1"
    ),
    "archived_history": (
        "checkpoints/interrupted/"
        "full_no_reconstruction_epoch68_20260727/history.json"
    ),
    "archived_best": (
        "checkpoints/interrupted/"
        "full_no_reconstruction_epoch68_20260727/best.pt"
    ),
    "archived_last": (
        "checkpoints/interrupted/"
        "full_no_reconstruction_epoch68_20260727/last.pt"
    ),
    "train_manifest": "data/qwen-material/train.jsonl",
    "test_manifest": "data/qwen-material/test.jsonl",
    "sensor_manifest": "data/qwen-material/sensors.jsonl",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "FORMAL_QUEUE_INTERRUPTION_OUTCOME.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the full_no_reconstruction clean restart outcome"
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


def _jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def build_interruption_outcome(
    *,
    files: Mapping[str, Path],
    expected_partial_epochs: int = 68,
    expected_total_epochs: int = 100,
    expected_test_samples: int = 826,
) -> dict[str, Any]:
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "formal interruption outcome inputs are missing: "
            + ", ".join(sorted(missing))
        )

    snapshot = _read_json(resolved["snapshot"])
    addendum = _read_json(resolved["addendum"])
    current_integrity = _read_json(resolved["current_integrity"])
    history = _read_json(resolved["history"])
    training_summary = _read_json(resolved["training_summary"])
    checkpoint_audit = _read_json(resolved["checkpoint_audit"])
    evaluation = _read_json(resolved["evaluation"])
    errors: list[str] = []

    partial = snapshot.get("partial_training", {})
    partial_fingerprint = partial.get("config_fingerprint")
    if snapshot.get("valid") is not True:
        errors.append("formal interruption snapshot is not valid")
    if (
        partial.get("experiment") != "full_no_reconstruction"
        or partial.get("completed_epochs") != expected_partial_epochs
        or partial.get("last_completed_epoch") != expected_partial_epochs
    ):
        errors.append("formal interruption snapshot epoch contract mismatch")

    failed_resume = addendum.get("failed_resume", {})
    if addendum.get("valid") is not True:
        errors.append("formal interruption addendum is not valid")
    if addendum.get("snapshot", {}).get("sha256") != _sha256(
        resolved["snapshot"]
    ):
        errors.append("formal interruption snapshot hash mismatch")
    if (
        failed_resume.get("decision")
        != "restart_same_formal_config_from_epoch_1"
        or failed_resume.get("last_checkpoint_zip_readable") is not False
        or failed_resume.get("best_checkpoint_zip_readable") is not True
        or failed_resume.get("optimizer_state_available_from_best") is not False
    ):
        errors.append("formal interruption restart decision mismatch")

    archived_files = addendum.get("archive", {}).get("files", {})
    for outcome_name, addendum_name in (
        ("archived_history", "history"),
        ("archived_best", "best_checkpoint"),
        ("archived_last", "last_checkpoint"),
    ):
        expected = archived_files.get(addendum_name, {})
        path = resolved[outcome_name]
        if (
            expected.get("size_bytes") != path.stat().st_size
            or expected.get("sha256") != _sha256(path)
        ):
            errors.append(f"archived partial artifact drift: {outcome_name}")

    snapshot_config = snapshot.get("source_bindings", {}).get("config", {})
    if snapshot_config.get("sha256") != _sha256(resolved["config"]):
        errors.append("formal configuration changed after interruption")

    integrity_summary = current_integrity.get("summary", {})
    raw_inventory = integrity_summary.get("raw_inventory_sha256")
    effective_inventory = integrity_summary.get(
        "effective_inventory_sha256"
    )
    if (
        current_integrity.get("valid") is not True
        or integrity_summary.get("decode_failure_count") != 0
        or integrity_summary.get("documented_recovery_count") != 0
        or raw_inventory != effective_inventory
        or effective_inventory
        != snapshot.get("input_state", {}).get("effective_inventory_sha256")
    ):
        errors.append("repaired source inventory contract mismatch")

    rows = history.get("history")
    observed_epochs = (
        [row.get("epoch") for row in rows]
        if isinstance(rows, list)
        else []
    )
    if observed_epochs != list(range(1, expected_total_epochs + 1)):
        errors.append("restart history is not a contiguous 1..100 run")
    if history.get("resumed_from") not in (None, ""):
        errors.append("formal restart history unexpectedly records a resume")
    config_fingerprint = history.get("config_fingerprint")
    if config_fingerprint != partial_fingerprint:
        errors.append("formal restart config fingerprint changed")

    if (
        training_summary.get("start_epoch") != 0
        or training_summary.get("completed_epochs") != expected_total_epochs
        or training_summary.get("checkpoints_saved") is not True
        or training_summary.get("resumed_from") not in (None, "")
        or training_summary.get("config_fingerprint") != config_fingerprint
    ):
        errors.append("formal restart training summary contract mismatch")

    if checkpoint_audit.get("valid") is not True:
        errors.append("formal restart checkpoint audit is not valid")
    if (
        checkpoint_audit.get("history_json_rows") != expected_total_epochs
        or checkpoint_audit.get("training_summary_completed_epochs")
        != expected_total_epochs
    ):
        errors.append("formal restart checkpoint audit epoch mismatch")
    expected_checkpoint = checkpoint_audit.get("expected", {})
    best = checkpoint_audit.get("best", {})
    last = checkpoint_audit.get("last", {})
    if expected_checkpoint.get("config_fingerprint") != config_fingerprint:
        errors.append("checkpoint audit config fingerprint mismatch")
    if best.get("sha256") != _sha256(resolved["best_checkpoint"]):
        errors.append("best checkpoint hash mismatch")
    if last.get("sha256") != _sha256(resolved["last_checkpoint"]):
        errors.append("last checkpoint hash mismatch")
    best_epoch = best.get("epoch")
    if (
        not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= expected_total_epochs
        or last.get("epoch") != expected_total_epochs
    ):
        errors.append("best/last checkpoint epoch contract mismatch")

    restart_stdout = resolved["restart_stdout"].read_text(
        encoding="utf-8-sig"
    )
    if "[mmfc-image-recovery]" in restart_stdout:
        errors.append("clean restart unexpectedly activated image recovery")
    if resolved["restart_stderr"].stat().st_size != 0:
        errors.append("clean restart stderr is not empty")

    metrics = evaluation.get("metrics", {})
    dataset = evaluation.get("dataset", {})
    evaluation_checkpoint = evaluation.get("checkpoint", {})
    evaluation_config = evaluation.get("evaluation_config", {})
    if (
        dataset.get("split") != "test"
        or dataset.get("evaluated_samples") != expected_test_samples
        or dataset.get("full_split_samples") != expected_test_samples
        or metrics.get("sample_count") != expected_test_samples
        or _jsonl_rows(resolved["predictions"]) != expected_test_samples
    ):
        errors.append("formal restart test sample contract mismatch")
    if dataset.get("sensor_degradation_enabled") is not False:
        errors.append("formal restart clean test uses sensor degradation")
    if evaluation_config.get("bootstrap_repeats") != 2000:
        errors.append("formal restart evaluation bootstrap mismatch")
    if evaluation_checkpoint.get("epoch") != best_epoch:
        errors.append("formal restart evaluation did not use audited best")
    metric_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "ece",
        "negative_log_likelihood",
        "multiclass_brier_score",
    )
    for name in metric_names:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"formal restart metric is invalid: {name}")

    source_bindings = {
        name: _binding(path) for name, path in sorted(resolved.items())
    }
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "completed_full_no_reconstruction_clean_restart_outcome",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "interruption": {
            "partial_completed_epochs": expected_partial_epochs,
            "partial_best_epoch": partial.get("best_checkpoint_epoch"),
            "partial_last_zip_readable": failed_resume.get(
                "last_checkpoint_zip_readable"
            ),
            "partial_best_optimizer_state_available": failed_resume.get(
                "optimizer_state_available_from_best"
            ),
            "decision": failed_resume.get("decision"),
        },
        "restart": {
            "start_epoch": training_summary.get("start_epoch"),
            "completed_epochs": training_summary.get("completed_epochs"),
            "resumed_from": training_summary.get("resumed_from"),
            "config_fingerprint": config_fingerprint,
            "history_contiguous": observed_epochs
            == list(range(1, expected_total_epochs + 1)),
            "image_recovery_marker_present": (
                "[mmfc-image-recovery]" in restart_stdout
            ),
            "stderr_empty": resolved["restart_stderr"].stat().st_size == 0,
            "effective_inventory_sha256": effective_inventory,
        },
        "training": {
            "experiment": "full_no_reconstruction",
            "best_epoch": best_epoch,
            "best_val_accuracy": best.get("best_val_accuracy"),
            "best_checkpoint_sha256": best.get("sha256"),
            "last_epoch": last.get("epoch"),
            "last_checkpoint_sha256": last.get("sha256"),
            "checkpoint_audit_valid": checkpoint_audit.get("valid"),
        },
        "test": {
            "sample_count": metrics.get("sample_count"),
            **{name: metrics.get(name) for name in metric_names},
            "checkpoint_epoch": evaluation_checkpoint.get("epoch"),
            "bootstrap_repeats": evaluation_config.get(
                "bootstrap_repeats"
            ),
        },
        "source_bindings": source_bindings,
        "errors": errors,
        "boundaries": [
            (
                "The corrupt epoch-68 last checkpoint and readable epoch-67 "
                "best checkpoint remain archived and are not used for the "
                "formal result."
            ),
            (
                "The formal variant restarted from epoch 1 with the unchanged "
                "registered configuration and repaired source inventory; it "
                "did not warm-start from the partial best checkpoint."
            ),
            (
                "This audit closes the interruption provenance chain for one "
                "training seed; it does not estimate training-seed variance."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    interruption = report["interruption"]
    restart = report["restart"]
    training = report["training"]
    test = report["test"]
    return "\n".join(
        [
            "# 正式消融中断重训结果审计",
            "",
            f"生成时间：`{report['generated_at']}`。",
            "",
            (
                f"`full_no_reconstruction` 的原部分运行记录到第 "
                f"{interruption['partial_completed_epochs']} 轮；因 last 检查点"
                "不可读且可读 best 缺少完整恢复状态，未用于正式结果。"
            ),
            (
                f"同一冻结配置从第 {restart['start_epoch'] + 1} 轮重新启动，"
                f"连续完成 {restart['completed_epochs']} 轮；stderr 为空，"
                "未激活图像恢复 shim。"
            ),
            (
                f"最佳轮次为 {training['best_epoch']}，最佳检查点 SHA-256 为 "
                f"`{training['best_checkpoint_sha256']}`。"
            ),
            "",
            (
                f"固定 test 共 {test['sample_count']} 条，Accuracy 为 "
                f"{100.0 * float(test['accuracy']):.2f}%，Macro F1 为 "
                f"{100.0 * float(test['macro_f1']):.2f}%，ECE 为 "
                f"{100.0 * float(test['ece']):.2f}%。"
            ),
            "",
            "## 证据边界",
            "",
            *[f"- {boundary}" for boundary in report["boundaries"]],
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError(
            f"formal interruption outcome is immutable: {output}"
        )
    report = build_interruption_outcome(
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
    output.with_suffix(".md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "valid": report["valid"],
                "completed_epochs": report["restart"]["completed_epochs"],
                "best_epoch": report["training"]["best_epoch"],
                "test_accuracy": report["test"]["accuracy"],
                "test_macro_f1": report["test"]["macro_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
