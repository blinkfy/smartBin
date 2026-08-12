"""Capture an immutable evidence snapshot for the training image incident."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "formal_status": (
        "reports/qwen-material/training/formal_ablations.status.json"
    ),
    "history": "checkpoints/ablations/full_no_degradation/history.json",
    "best_checkpoint": (
        "checkpoints/ablations/full_no_degradation/best.pt"
    ),
    "last_checkpoint": (
        "checkpoints/ablations/full_no_degradation/last.pt"
    ),
    "config": "configs/ablations/full_no_degradation.json",
    "train_manifest": "data/qwen-material/train.jsonl",
    "formal_run_manifest": (
        "reports/qwen-material/ablations/FORMAL_RUN_MANIFEST.json"
    ),
    "recovery_audit": (
        "reports/qwen-material/IMAGE_READ_RECOVERY_AUDIT.json"
    ),
    "runtime_shim": "runtime_recovery/sitecustomize.py",
    "recovery_chain": (
        "reports/qwen-material/training/run_image_recovery_chain.ps1"
    ),
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "IMAGE_READ_RECOVERY_INCIDENT.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the failed-stage and partial-checkpoint snapshot"
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


def build_incident_snapshot(
    *,
    files: Mapping[str, Path],
    expected_completed_epochs: int = 5,
) -> dict[str, Any]:
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "incident inputs are missing: " + ", ".join(sorted(missing))
        )

    status = _read_json(resolved["formal_status"])
    history = _read_json(resolved["history"])
    recovery_audit = _read_json(resolved["recovery_audit"])
    history_rows = history.get("history")
    if not isinstance(history_rows, list) or not history_rows:
        raise ValueError("partial history is missing epoch rows")
    completed_epochs = len(history_rows)
    last_completed_epoch = int(history_rows[-1]["epoch"])
    errors: list[str] = []
    if status.get("state") != "failed":
        errors.append("formal queue status was not failed at capture")
    if status.get("stage") != "pipeline":
        errors.append("formal queue failure stage was not pipeline")
    if "full_no_degradation_training failed" not in str(
        status.get("message", "")
    ):
        errors.append("formal queue failure message is unexpected")
    if completed_epochs != expected_completed_epochs:
        errors.append(
            "partial history epoch count mismatch: "
            f"{completed_epochs} != {expected_completed_epochs}"
        )
    if last_completed_epoch != expected_completed_epochs:
        errors.append(
            "partial history last epoch mismatch: "
            f"{last_completed_epoch} != {expected_completed_epochs}"
        )
    if recovery_audit.get("valid") is not True:
        errors.append("image recovery audit was not valid at capture")

    source_bindings = {
        name: _binding(path) for name, path in sorted(resolved.items())
    }
    snapshot_payload = {
        "formal_status": status,
        "partial_training": {
            "experiment": history.get("experiment"),
            "config_fingerprint": history.get("config_fingerprint"),
            "completed_epochs": completed_epochs,
            "last_completed_epoch": last_completed_epoch,
            "best_val_accuracy_after_partial_run": history.get(
                "best_val_accuracy"
            ),
            "best_checkpoint_epoch_after_partial_run": history.get(
                "best_checkpoint_epoch"
            ),
        },
        "recovery_mapping": {
            "source": recovery_audit.get("source"),
            "fallback": recovery_audit.get("fallback"),
            "comparison": recovery_audit.get("comparison"),
            "manifest_occurrences": recovery_audit.get(
                "manifest_occurrences"
            ),
        },
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(
            snapshot_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "immutable_training_image_recovery_incident_snapshot",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "snapshot_sha256": snapshot_sha256,
        "snapshot": snapshot_payload,
        "source_bindings": source_bindings,
        "recovery_invariants": {
            "training_core_modified": False,
            "training_config_modified": False,
            "data_manifest_modified": False,
            "original_image_modified": False,
            "resume_checkpoint_role": "last.pt",
            "resume_starts_after_completed_epoch": expected_completed_epochs,
        },
        "errors": errors,
        "boundaries": [
            (
                "The formal status file is transient and will later be "
                "overwritten by queue recovery; its failed payload is embedded "
                "verbatim in this immutable snapshot."
            ),
            (
                "Checkpoint hashes identify the exact partial bundle selected "
                "for resume, but checkpoint binaries are excluded from the "
                "paper evidence archive."
            ),
            (
                "The recovery redirects one train image only and does not "
                "change validation or test inputs."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    partial = report["snapshot"]["partial_training"]
    status = report["snapshot"]["formal_status"]
    source = report["snapshot"]["recovery_mapping"]["source"]
    fallback = report["snapshot"]["recovery_mapping"]["fallback"]
    return "\n".join(
        [
            "# 训练图像恢复事件快照",
            "",
            f"捕获时间：`{report['captured_at']}`。",
            "",
            (
                f"正式队列状态为 `{status['state']}`，失败信息为 "
                f"`{status['message']}`。"
            ),
            (
                f"`full_no_degradation` 已完整保存 "
                f"{partial['completed_epochs']} 轮，最后完成轮次为 "
                f"{partial['last_completed_epoch']}；恢复角色为 `last.pt`。"
            ),
            "",
            (
                f"损坏源 SHA-256：`{source['sha256']}`；镜像 SHA-256："
                f"`{fallback['sha256']}`。"
            ),
            (
                "训练核心、配置、数据清单和原图均未由恢复流程改写；"
                "只在运行时把一个 train 图像读取映射到哈希锁定镜像。"
            ),
            "",
            "本文件绑定失败状态、部分检查点、配置、训练清单、正式运行清单、"
            "恢复审计、shim 和恢复链脚本的 SHA-256。检查点本体不进入论文证据 ZIP。",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError(
            f"incident snapshot already exists and is immutable: {output}"
        )
    report = build_incident_snapshot(
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
                "completed_epochs": report["snapshot"][
                    "partial_training"
                ]["completed_epochs"],
                "source_count": len(report["source_bindings"]),
                "snapshot_sha256": report["snapshot_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
