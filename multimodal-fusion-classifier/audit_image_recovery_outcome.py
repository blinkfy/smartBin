"""Audit the completed training outcome after the image-read recovery."""

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
        "reports/qwen-material/IMAGE_READ_RECOVERY_INCIDENT.json"
    ),
    "addendum": (
        "reports/qwen-material/IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.json"
    ),
    "recovery_audit": (
        "reports/qwen-material/IMAGE_READ_RECOVERY_AUDIT.json"
    ),
    "source_integrity": (
        "reports/qwen-material/SOURCE_IMAGE_INTEGRITY_AUDIT.json"
    ),
    "history": "checkpoints/ablations/full_no_degradation/history.json",
    "checkpoint_audit": (
        "reports/qwen-material/ablations/full_no_degradation/"
        "CHECKPOINT_AUDIT.json"
    ),
    "evaluation": (
        "reports/qwen-material/ablations/full_no_degradation/"
        "evaluation.json"
    ),
    "resume_stdout": (
        "reports/qwen-material/training/"
        "image_recovery_chain.full_no_degradation_resume_v2.stdout.log"
    ),
    "resume_stderr": (
        "reports/qwen-material/training/"
        "image_recovery_chain.full_no_degradation_resume_v2.stderr.log"
    ),
    "runtime_shim": "runtime_recovery/sitecustomize.py",
    "recovery_chain": (
        "reports/qwen-material/training/run_image_recovery_chain.ps1"
    ),
    "config": "configs/ablations/full_no_degradation.json",
    "train_manifest": "data/qwen-material/train.jsonl",
    "test_manifest": "data/qwen-material/test.jsonl",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "IMAGE_READ_RECOVERY_OUTCOME.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the completed full_no_degradation recovery"
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


def _resume_event(path: Path) -> dict[str, Any] | None:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event") == "resumed":
            return value
    return None


def build_recovery_outcome(
    *,
    files: Mapping[str, Path],
    expected_partial_epochs: int = 5,
    expected_total_epochs: int = 100,
    expected_test_samples: int = 826,
) -> dict[str, Any]:
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "recovery outcome inputs are missing: "
            + ", ".join(sorted(missing))
        )

    incident = _read_json(resolved["incident"])
    addendum = _read_json(resolved["addendum"])
    recovery_audit = _read_json(resolved["recovery_audit"])
    source_integrity = _read_json(resolved["source_integrity"])
    history = _read_json(resolved["history"])
    checkpoint_audit = _read_json(resolved["checkpoint_audit"])
    evaluation = _read_json(resolved["evaluation"])
    resume_event = _resume_event(resolved["resume_stdout"])
    errors: list[str] = []

    partial = incident.get("snapshot", {}).get("partial_training", {})
    if (
        incident.get("valid") is not True
        or partial.get("completed_epochs") != expected_partial_epochs
    ):
        errors.append("incident snapshot does not bind the expected partial run")

    attempt_2 = addendum.get("resume_attempt_2", {})
    functional = addendum.get("functional_revision", {})
    if addendum.get("valid") is not True:
        errors.append("recovery addendum is not valid")
    if attempt_2.get("completed_epochs_before") != expected_partial_epochs:
        errors.append("attempt 2 does not start from the expected epoch")
    if functional.get("sha256") != _sha256(resolved["runtime_shim"]):
        errors.append("functional recovery shim hash mismatch")
    if functional.get("recovery_chain_sha256") != _sha256(
        resolved["recovery_chain"]
    ):
        errors.append("recovery chain hash mismatch")

    if recovery_audit.get("valid") is not True:
        errors.append("image recovery audit is not valid")
    if source_integrity.get("valid") is not True:
        errors.append("source image integrity audit is not valid")
    integrity_summary = source_integrity.get("summary", {})
    if (
        integrity_summary.get("decode_failure_count") != 1
        or integrity_summary.get("documented_recovery_count") != 1
    ):
        errors.append("source image integrity recovery count mismatch")

    rows = history.get("history")
    expected_epochs = list(range(1, expected_total_epochs + 1))
    observed_epochs = (
        [row.get("epoch") for row in rows]
        if isinstance(rows, list)
        else []
    )
    if observed_epochs != expected_epochs:
        errors.append("final history is not a contiguous 1..100 run")
    if checkpoint_audit.get("valid") is not True:
        errors.append("final checkpoint audit is not valid")
    if checkpoint_audit.get("history_json_rows") != expected_total_epochs:
        errors.append("checkpoint audit history length mismatch")
    if (
        checkpoint_audit.get("training_summary_completed_epochs")
        != expected_total_epochs
    ):
        errors.append("training summary epoch count mismatch")

    best = checkpoint_audit.get("best", {})
    last = checkpoint_audit.get("last", {})
    expected = checkpoint_audit.get("expected", {})
    config_fingerprint = history.get("config_fingerprint")
    if expected.get("config_fingerprint") != config_fingerprint:
        errors.append("history and checkpoint config fingerprints differ")
    if last.get("epoch") != expected_total_epochs:
        errors.append("last checkpoint is not the target epoch")
    best_epoch = best.get("epoch")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= expected_total_epochs:
        errors.append("best checkpoint epoch is invalid")

    if resume_event is None:
        errors.append("resume event is missing from attempt 2 stdout")
        resume_event = {}
    else:
        if resume_event.get("completed_epochs") != expected_partial_epochs:
            errors.append("resume event partial epoch count mismatch")
        if resume_event.get("target_epochs") != expected_total_epochs:
            errors.append("resume event target epoch count mismatch")
    if resolved["resume_stderr"].stat().st_size != 0:
        errors.append("attempt 2 stderr is not empty")

    metrics = evaluation.get("metrics", {})
    dataset = evaluation.get("dataset", {})
    evaluation_checkpoint = evaluation.get("checkpoint", {})
    if dataset.get("split") != "test":
        errors.append("recovery outcome evaluation is not the test split")
    if (
        dataset.get("evaluated_samples") != expected_test_samples
        or dataset.get("full_split_samples") != expected_test_samples
        or metrics.get("sample_count") != expected_test_samples
    ):
        errors.append("recovery outcome test sample count mismatch")
    if dataset.get("sensor_degradation_enabled") is not False:
        errors.append("recovery outcome clean evaluation uses degradation")
    if evaluation_checkpoint.get("epoch") != best_epoch:
        errors.append("evaluation did not use the audited best checkpoint")

    source_bindings = {
        name: _binding(path) for name, path in sorted(resolved.items())
    }
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "completed_full_no_degradation_image_recovery_outcome",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "recovery": {
            "completed_epochs_before_resume": expected_partial_epochs,
            "completed_epochs_after_resume": len(observed_epochs),
            "resume_event": resume_event,
            "runtime_shim_sha256": _sha256(resolved["runtime_shim"]),
            "recovery_chain_sha256": _sha256(resolved["recovery_chain"]),
            "resume_stderr_empty": (
                resolved["resume_stderr"].stat().st_size == 0
            ),
        },
        "training": {
            "experiment": "full_no_degradation",
            "config_fingerprint": config_fingerprint,
            "best_epoch": best_epoch,
            "best_val_accuracy": best.get("best_val_accuracy"),
            "best_checkpoint_sha256": best.get("sha256"),
            "last_epoch": last.get("epoch"),
            "last_checkpoint_sha256": last.get("sha256"),
            "checkpoint_audit_valid": checkpoint_audit.get("valid"),
        },
        "test": {
            "sample_count": metrics.get("sample_count"),
            "accuracy": metrics.get("accuracy"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "ece": metrics.get("ece"),
            "negative_log_likelihood": metrics.get(
                "negative_log_likelihood"
            ),
            "multiclass_brier_score": metrics.get(
                "multiclass_brier_score"
            ),
            "checkpoint_epoch": evaluation_checkpoint.get("epoch"),
        },
        "source_bindings": source_bindings,
        "errors": errors,
        "boundaries": [
            (
                "The runtime shim redirected exactly one training image to a "
                "hash-locked same-name mirror; validation and test inputs were "
                "not redirected."
            ),
            (
                "The incident snapshot and resume event support continuation "
                "from five completed epochs, but no pre-incident per-image hash "
                "exists for reconstructing the earlier bytes of that image."
            ),
            (
                "This audit establishes the recovered run's provenance and "
                "completion; it does not make the recovered training input "
                "byte-identical to variants trained before the incident."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    recovery = report["recovery"]
    training = report["training"]
    test = report["test"]
    return "\n".join(
        [
            "# 训练图像恢复结果审计",
            "",
            f"生成时间：`{report['generated_at']}`。",
            "",
            (
                f"`full_no_degradation` 从已绑定的第 "
                f"{recovery['completed_epochs_before_resume']} 轮状态恢复，"
                f"连续完成至第 {recovery['completed_epochs_after_resume']} 轮；"
                f"恢复进程 stderr 为空。"
            ),
            (
                f"检查点审计有效，最佳轮次为 {training['best_epoch']}，"
                f"最佳检查点 SHA-256 为 "
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
            f"image recovery outcome is immutable: {output}"
        )
    report = build_recovery_outcome(
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
                "completed_epochs": report["recovery"][
                    "completed_epochs_after_resume"
                ],
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
