"""Capture the partial formal-ablation bundle before a second exact resume."""

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
    "recovery_chain_status": (
        "reports/qwen-material/training/image_recovery_chain.status.json"
    ),
    "history": (
        "checkpoints/ablations/full_no_reconstruction/history.json"
    ),
    "best_checkpoint": (
        "checkpoints/ablations/full_no_reconstruction/best.pt"
    ),
    "last_checkpoint": (
        "checkpoints/ablations/full_no_reconstruction/last.pt"
    ),
    "config": "configs/ablations/full_no_reconstruction.json",
    "train_manifest": "data/qwen-material/train.jsonl",
    "training_stdout": (
        "reports/qwen-material/training/full_no_reconstruction.stdout.log"
    ),
    "training_stderr": (
        "reports/qwen-material/training/full_no_reconstruction.stderr.log"
    ),
    "source_transition": (
        "reports/qwen-material/SOURCE_IMAGE_REPAIR_TRANSITION.json"
    ),
    "current_integrity": (
        "reports/qwen-material/SOURCE_IMAGE_INTEGRITY_CURRENT.json"
    ),
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "FORMAL_QUEUE_INTERRUPTION.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture full_no_reconstruction at its last saved epoch"
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


def build_interruption_snapshot(
    *,
    files: Mapping[str, Path],
    expected_completed_epochs: int = 68,
) -> dict[str, Any]:
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "formal interruption inputs are missing: "
            + ", ".join(sorted(missing))
        )
    history = _read_json(resolved["history"])
    transition = _read_json(resolved["source_transition"])
    current_integrity = _read_json(resolved["current_integrity"])
    rows = history.get("history")
    observed_epochs = (
        [row.get("epoch") for row in rows]
        if isinstance(rows, list)
        else []
    )
    errors: list[str] = []
    if observed_epochs != list(range(1, expected_completed_epochs + 1)):
        errors.append("partial history is not contiguous through epoch 68")
    if transition.get("valid") is not True:
        errors.append("source repair transition is not valid")
    if (
        current_integrity.get("valid") is not True
        or current_integrity.get("expected_recovery_state") != "repaired"
    ):
        errors.append("current repaired source inventory is not valid")
    if resolved["training_stderr"].stat().st_size != 0:
        errors.append("partial training stderr is not empty")
    stdout = resolved["training_stdout"].read_text(encoding="utf-8-sig")
    if "[mmfc-image-recovery] active:" not in stdout:
        errors.append("partial run does not record the recovery shim activation")

    bindings = {
        name: {
            "path": _display_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in sorted(resolved.items())
    }
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "full_no_reconstruction_partial_resume_snapshot",
        "captured_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "partial_training": {
            "experiment": "full_no_reconstruction",
            "completed_epochs": len(observed_epochs),
            "last_completed_epoch": (
                observed_epochs[-1] if observed_epochs else None
            ),
            "best_checkpoint_epoch": history.get("best_checkpoint_epoch"),
            "best_val_accuracy": history.get("best_val_accuracy"),
            "config_fingerprint": history.get("config_fingerprint"),
            "resume_checkpoint_role": "last.pt",
            "training_summary_exists": (
                resolved["history"].with_name("training_summary.json").is_file()
            ),
        },
        "input_state": {
            "first_68_epochs_used_hash_locked_redirection": True,
            "subsequent_epochs_use_repaired_source_without_shim": True,
            "effective_image_bytes_unchanged_across_resume": True,
            "transition_cause": transition.get("transition", {}).get("cause"),
            "effective_inventory_sha256": current_integrity.get(
                "summary", {}
            ).get("effective_inventory_sha256"),
        },
        "source_bindings": bindings,
        "errors": errors,
        "boundaries": [
            (
                "The first 68 epochs used the activated redirection shim; the "
                "repaired source now equals the same locked fallback bytes."
            ),
            (
                "This snapshot binds the partial checkpoint before resume but "
                "does not infer why the source path changed on disk."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    partial = report["partial_training"]
    state = report["input_state"]
    return "\n".join(
        [
            "# 正式消融队列中断快照",
            "",
            f"捕获时间：`{report['captured_at']}`。",
            "",
            (
                f"`full_no_reconstruction` 已连续保存 "
                f"{partial['completed_epochs']} 轮，最后轮次为 "
                f"{partial['last_completed_epoch']}，恢复角色为 `last.pt`。"
            ),
            (
                "前 68 轮通过已激活 shim 读取锁定镜像；当前源图已与该镜像"
                "逐字节相同，因此续训将移除 shim，模型实际看到的该样本字节不变。"
            ),
            (
                "修复态有效输入清单指纹为 "
                f"`{state['effective_inventory_sha256']}`。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError(
            f"interruption snapshot already exists and is immutable: {output}"
        )
    report = build_interruption_snapshot(
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
        render_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "valid": report["valid"],
                "completed_epochs": report["partial_training"][
                    "completed_epochs"
                ],
                "source_binding_count": len(report["source_bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
