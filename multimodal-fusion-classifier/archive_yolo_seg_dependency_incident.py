"""Mirror the YOLO-seg dependency incident sources before log-path reuse."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INCIDENT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "YOLO_SEG_DEPENDENCY_INCIDENT.json"
)
DEFAULT_CHECKPOINT_ARCHIVE = (
    PROJECT_ROOT
    / "checkpoints"
    / "interrupted"
    / "yolo_seg_missing_polars_epoch1_20260728"
)
DEFAULT_MIRROR = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "training"
    / "interrupted"
    / "yolo_seg_missing_polars_epoch1_20260728"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive immutable sources for the YOLO dependency incident"
    )
    parser.add_argument("--incident", type=Path, default=DEFAULT_INCIDENT)
    parser.add_argument(
        "--checkpoint-archive", type=Path, default=DEFAULT_CHECKPOINT_ARCHIVE
    )
    parser.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR)
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


def archive_incident_sources(
    *,
    incident_path: Path,
    checkpoint_archive: Path,
    mirror_root: Path,
) -> dict[str, Any]:
    incident_file = _resolve(incident_path)
    checkpoint_dir = _resolve(checkpoint_archive)
    mirror = _resolve(mirror_root)
    if mirror.exists():
        raise FileExistsError(f"incident mirror already exists: {mirror}")
    incident = _read_json(incident_file)
    if incident.get("valid") is not True:
        raise ValueError("YOLO-seg dependency incident is not valid")
    source_bindings = incident.get("source_bindings")
    archive_bindings = incident.get("archive_bindings")
    if not isinstance(source_bindings, dict) or not source_bindings:
        raise ValueError("incident source bindings are missing")
    if not isinstance(archive_bindings, dict) or not archive_bindings:
        raise ValueError("incident checkpoint archive bindings are missing")

    copy_plan: list[tuple[str, Path, Path, Mapping[str, Any]]] = []
    errors: list[str] = []
    for name, binding in sorted(source_bindings.items()):
        if not isinstance(binding, Mapping):
            errors.append(f"invalid source binding: {name}")
            continue
        relative = Path(str(binding.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe source binding path: {name}")
            continue
        source = _resolve(relative)
        target = mirror / "sources" / relative
        if not source.is_file():
            errors.append(f"bound source is missing: {name}")
            continue
        if _sha256(source) != binding.get("sha256"):
            errors.append(f"bound source hash drift: {name}")
            continue
        copy_plan.append((name, source, target, binding))

    for relative_name, binding in sorted(archive_bindings.items()):
        path = checkpoint_dir / Path(relative_name)
        if not path.is_file():
            errors.append(f"checkpoint archive file is missing: {relative_name}")
        elif _sha256(path) != binding.get("sha256"):
            errors.append(f"checkpoint archive hash drift: {relative_name}")
    if errors:
        raise ValueError("; ".join(errors))

    mirror.mkdir(parents=True)
    mirrored_bindings: dict[str, dict[str, Any]] = {}
    for name, source, target, original_binding in copy_plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        mirrored = _binding(target)
        if (
            mirrored["sha256"] != original_binding.get("sha256")
            or mirrored["size_bytes"] != original_binding.get("size_bytes")
        ):
            raise RuntimeError(f"mirrored source verification failed: {name}")
        mirrored_bindings[name] = {
            "original_path": original_binding.get("path"),
            **mirrored,
        }

    return {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "yolo_seg_dependency_incident_source_mirror",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "incident": _binding(incident_file),
        "checkpoint_archive": {
            "path": _display_path(checkpoint_dir),
            "file_count": len(archive_bindings),
            "bindings": archive_bindings,
        },
        "source_mirror": {
            "path": _display_path(mirror),
            "file_count": len(mirrored_bindings),
            "bindings": mirrored_bindings,
        },
        "restart_authorization": {
            "decision": "reuse_fixed_runtime_log_paths_after_immutable_mirror",
            "same_frozen_protocol_required": True,
            "clean_checkpoint_directory_required": True,
        },
        "errors": [],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    mirror = report["source_mirror"]
    archive = report["checkpoint_archive"]
    return "\n".join(
        [
            "# YOLO-seg 依赖中断证据归档补充",
            "",
            f"归档时间：`{report['captured_at']}`。",
            "",
            (
                f"中断快照绑定的 {mirror['file_count']} 个状态、日志、协议和"
                f"环境文件已逐哈希镜像到 `{mirror['path']}`。"
            ),
            (
                f"失败运行的 {archive['file_count']} 个文件继续保存在 "
                f"`{archive['path']}`；其中没有模型权重。"
            ),
            (
                "完成该不可变镜像后，后台编排可以复用固定日志路径，并按相同"
                "冻结协议从第 1 轮干净重启。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError(f"incident addendum is immutable: {output}")
    report = archive_incident_sources(
        incident_path=args.incident,
        checkpoint_archive=args.checkpoint_archive,
        mirror_root=args.mirror,
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
                "mirrored_file_count": report["source_mirror"]["file_count"],
                "checkpoint_archive_file_count": report["checkpoint_archive"][
                    "file_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
