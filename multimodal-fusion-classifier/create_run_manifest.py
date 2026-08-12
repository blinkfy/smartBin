"""Freeze source, data, weight, and configuration identities for one run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from mmfc.data import write_json
from mmfc.training import (
    PROJECT_ROOT,
    _training_config_fingerprint,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a SHA-256 reproducibility manifest for a training run"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-file", type=Path, action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    try:
        display_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(path.resolve())
    return {
        "path": display_path,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest input is not a file: {resolved}")
        seen.add(resolved)
        result.append(sha256_file(resolved))
    return result


def build_run_manifest(
    config: dict[str, Any],
    *,
    config_path: Path,
    extra_files: Iterable[Path] = (),
) -> dict[str, Any]:
    data = config.get("data", {})
    data_paths = [
        resolve_project_path(data[key])
        for key in (
            "train_manifest",
            "val_manifest",
            "test_manifest",
            "sensor_manifest",
        )
    ]
    source_paths = [PROJECT_ROOT / "train.py", *sorted((PROJECT_ROOT / "mmfc").glob("*.py"))]
    weight_paths: list[Path] = []
    yolo_weights = config.get("model", {}).get("yolo_weights")
    if yolo_weights:
        weight_paths.append(resolve_project_path(yolo_weights))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "experiment": config.get("experiment"),
        "configuration": {
            **sha256_file(config_path.resolve()),
            "training_fingerprint": _training_config_fingerprint(config),
        },
        "files": {
            "data": inventory(data_paths),
            "source": inventory(source_paths),
            "weights": inventory(weight_paths),
            "extra": inventory(extra_files),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.cuda.is_available()
                else None
            ),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "notes": [
            "The CLIP model identifier is recorded in the configuration; its "
            "local cache files are not copied into this manifest.",
            "Checkpoint hashes are recorded separately by audit_checkpoints.py "
            "after training completes.",
        ],
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    report = build_run_manifest(
        config,
        config_path=config_path,
        extra_files=[resolve_project_path(path) for path in args.extra_file],
    )
    output = resolve_project_path(args.output)
    write_json(output, report)
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
