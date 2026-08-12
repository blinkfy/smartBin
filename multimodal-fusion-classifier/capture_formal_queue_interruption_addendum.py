"""Bind the archived partial run and failed exact-resume attempt."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent
SNAPSHOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "FORMAL_QUEUE_INTERRUPTION.json"
)
ARCHIVE_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "interrupted"
    / "full_no_reconstruction_epoch68_20260727"
)
FAILED_STATUS = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "training"
    / "source_repair_continuation.status.json"
)
FAILED_STDERR = FAILED_STATUS.with_name(
    "source_repair_continuation.full_no_reconstruction_resume.stderr.log"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "FORMAL_QUEUE_INTERRUPTION_ADDENDUM.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind the archived epoch-68 bundle and failed resume"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _zip_readable(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def build_interruption_addendum(
    *,
    snapshot_path: Path,
    archive_dir: Path,
    failed_status_path: Path,
    failed_stderr_path: Path,
) -> dict[str, Any]:
    snapshot = _read_json(snapshot_path)
    status = _read_json(failed_status_path)
    errors: list[str] = []
    if snapshot.get("valid") is not True:
        errors.append("interruption snapshot is not valid")
    if (
        status.get("state") != "failed"
        or "full_no_reconstruction_resume failed" not in str(
            status.get("message", "")
        )
    ):
        errors.append("failed exact-resume status is missing")
    archived = {
        name: archive_dir / filename
        for name, filename in {
            "history": "history.json",
            "best_checkpoint": "best.pt",
            "last_checkpoint": "last.pt",
        }.items()
    }
    for name, path in archived.items():
        if not path.is_file():
            errors.append(f"archived partial file is missing: {name}")
            continue
        expected_hash = snapshot.get("source_bindings", {}).get(
            name, {}
        ).get("sha256")
        if _sha256(path) != expected_hash:
            errors.append(f"archived partial file hash mismatch: {name}")
    best_readable = (
        _zip_readable(archived["best_checkpoint"])
        if archived["best_checkpoint"].is_file()
        else False
    )
    last_readable = (
        _zip_readable(archived["last_checkpoint"])
        if archived["last_checkpoint"].is_file()
        else False
    )
    if not best_readable:
        errors.append("archived best.pt is not a readable ZIP checkpoint")
    if last_readable:
        errors.append("archived last.pt unexpectedly passed ZIP validation")
    stderr = failed_stderr_path.read_text(encoding="utf-8-sig")
    if "failed finding central directory" not in stderr:
        errors.append("failed resume stderr does not identify ZIP corruption")

    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "archived_partial_checkpoint_and_failed_resume_addendum",
        "captured_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "snapshot": {
            "path": _display_path(snapshot_path),
            "sha256": _sha256(snapshot_path),
        },
        "failed_resume": {
            "status": status,
            "status_sha256": _sha256(failed_status_path),
            "stderr_sha256": _sha256(failed_stderr_path),
            "last_checkpoint_zip_readable": last_readable,
            "best_checkpoint_zip_readable": best_readable,
            "optimizer_state_available_from_best": False,
            "decision": "restart_same_formal_config_from_epoch_1",
        },
        "archive": {
            "path": _display_path(archive_dir),
            "files": {
                name: {
                    "path": _display_path(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for name, path in archived.items()
                if path.is_file()
            },
        },
        "errors": errors,
        "boundaries": [
            (
                "The epoch-68 last.pt is retained under checkpoints/interrupted "
                "and is not used for scientific results."
            ),
            (
                "The readable epoch-67 best.pt lacks optimizer, scheduler, and "
                "RNG state, so warm-start continuation was rejected."
            ),
            (
                "The formal full_no_reconstruction variant is restarted from "
                "epoch 1 with the unchanged registered configuration."
            ),
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    failed = report["failed_resume"]
    return "\n".join(
        [
            "# 正式消融队列中断补充记录",
            "",
            f"捕获时间：`{report['captured_at']}`。",
            "",
            "第 68 轮 `last.pt` 未形成可读 ZIP 中央目录，精确续训在加载前失败；"
            "第 67 轮 `best.pt` 可读但不含优化器、调度器和 RNG 状态。",
            (
                "因此正式决策为：`"
                f"{failed['decision']}`，部分目录原样保存在 `"
                f"{report['archive']['path']}`，不进入论文结果。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError(
            f"interruption addendum already exists: {output}"
        )
    report = build_interruption_addendum(
        snapshot_path=SNAPSHOT,
        archive_dir=ARCHIVE_DIR,
        failed_status_path=FAILED_STATUS,
        failed_stderr_path=FAILED_STDERR,
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
                "decision": report["failed_resume"]["decision"],
                "archived_file_count": len(report["archive"]["files"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
