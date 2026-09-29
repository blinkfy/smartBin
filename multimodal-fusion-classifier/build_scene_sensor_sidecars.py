"""Build annotation-free full-image 4x4 sensor sidecars."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

from PIL import Image

from mmfc.model import PROJECT_ROOT
from mmfc.scene_sensors import (
    QwenSceneSensorEnhancer,
    scene_record_usable,
    simulate_scene_rule_sensor,
)
from mmfc.sensors import qwen_api_key


DEFAULT_IMAGES_ROOT = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "images"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "scene-sensor-sidecars"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build label-free full-image sensor sidecars"
    )
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("rule", "qwen"), default="rule")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--model", default="qwen3.5-flash")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _portable_path(path: Path) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for source_id in sorted(records):
            handle.write(
                json.dumps(
                    records[source_id],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")
    temporary.replace(path)


def discover_images(images_root: Path) -> list[dict[str, str]]:
    images_root = _resolve(images_root)
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for split in SPLITS:
        split_root = images_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"image split is missing: {split_root}")
        for path in sorted(
            (item for item in split_root.iterdir() if item.is_file()),
            key=lambda item: item.name,
        ):
            source_id = path.stem
            if not source_id or source_id in seen:
                raise ValueError(f"duplicate or empty source image id: {path}")
            seen.add(source_id)
            records.append(
                {
                    "source_image_id": source_id,
                    "split": split,
                    "image_path": _portable_path(path),
                }
            )
    return sorted(records, key=lambda item: item["source_image_id"])


def _load_resume(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            source_id = value.get("source_image_id")
            if not isinstance(source_id, str) or source_id in records:
                raise ValueError(
                    f"invalid resume source id at line {line_number}"
                )
            records[source_id] = value
    return records


def _build_one(
    image_record: Mapping[str, str],
    *,
    mode: str,
    seed: int,
    enhancer: QwenSceneSensorEnhancer | None,
) -> dict[str, Any]:
    path = _resolve(Path(image_record["image_path"]))
    with Image.open(path) as image:
        rule = simulate_scene_rule_sensor(
            image,
            image_record["source_image_id"],
            seed=seed,
        )
        record = (
            enhancer.enhance(image, rule)
            if mode == "qwen" and enhancer is not None
            else rule
        )
        record["image_width"] = int(image.width)
        record["image_height"] = int(image.height)
    record["split"] = image_record["split"]
    record["image_path"] = image_record["image_path"]
    return record


def build(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.workers) < 1 or int(args.workers) > 32:
        raise ValueError("--workers must be in range 1..32")
    if int(args.checkpoint_every) < 1 or int(args.progress_every) < 1:
        raise ValueError("checkpoint and progress intervals must be positive")
    if args.limit is not None and int(args.limit) < 1:
        raise ValueError("--limit must be positive")
    if float(args.timeout) <= 0.0 or int(args.retries) < 0:
        raise ValueError("timeout must be positive and retries non-negative")

    images_root = _resolve(args.images_root)
    output = _resolve(args.output)
    observations_path = output / "observations.jsonl"
    image_records = discover_images(images_root)
    if args.limit is not None:
        image_records = image_records[: int(args.limit)]
    selected = {record["source_image_id"]: record for record in image_records}
    api_key = qwen_api_key(args.api_key_env) if args.mode == "qwen" else None
    if args.mode == "qwen" and not api_key and not args.dry_run:
        raise RuntimeError(
            f"Qwen mode requires API credentials in {args.api_key_env}"
        )
    if args.dry_run:
        return {
            "state": "dry_run",
            "mode": args.mode,
            "source_image_count": len(image_records),
            "network_requests": 0,
        }

    existing = _load_resume(observations_path) if args.resume else {}
    completed = {
        source_id: record
        for source_id, record in existing.items()
        if source_id in selected
        and record.get("split") == selected[source_id]["split"]
        and record.get("image_path") == selected[source_id]["image_path"]
        and scene_record_usable(
            record,
            mode=args.mode,
            qwen_model=args.model,
        )
    }
    pending = [
        record
        for record in image_records
        if record["source_image_id"] not in completed
    ]
    enhancer = (
        QwenSceneSensorEnhancer(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            retries=int(args.retries),
            timeout=float(args.timeout),
        )
        if args.mode == "qwen"
        else None
    )
    started = time.monotonic()
    new_completed = 0
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.workers)
    ) as executor:
        futures = {
            executor.submit(
                _build_one,
                record,
                mode=args.mode,
                seed=int(args.seed),
                enhancer=enhancer,
            ): record
            for record in pending
        }
        for future in concurrent.futures.as_completed(futures):
            image_record = futures[future]
            source_id = image_record["source_image_id"]
            try:
                completed[source_id] = future.result()
            except Exception as exc:
                errors.append(
                    {
                        "source_image_id": source_id,
                        "error": f"{type(exc).__name__}: {exc}"[:500],
                    }
                )
            new_completed += 1
            if (
                new_completed % int(args.checkpoint_every) == 0
                or new_completed == len(pending)
            ):
                _atomic_jsonl(observations_path, completed)
            if (
                new_completed % int(args.progress_every) == 0
                or new_completed == len(pending)
            ):
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = new_completed / elapsed
                remaining = len(pending) - new_completed
                eta = remaining / rate if rate > 0 else 0.0
                print(
                    f"scene-sensors {new_completed}/{len(pending)} "
                    f"rate={rate:.2f}/s eta={eta:.1f}s errors={len(errors)}",
                    file=sys.stderr,
                    flush=True,
                )

    _atomic_jsonl(observations_path, completed)
    records = list(completed.values())
    qwen_success = sum(
        record.get("source", {}).get("enhancement_status") == "success"
        for record in records
    )
    fallback_count = sum(
        str(record.get("source", {}).get("enhancement_status", "")).startswith(
            "fallback"
        )
        for record in records
    )
    report = {
        "schema_version": 1,
        "state": "complete" if not errors else "complete_with_errors",
        "valid": not errors and len(completed) == len(image_records),
        "mode": args.mode,
        "model": args.model if args.mode == "qwen" else None,
        "source_contract": {
            "full_image_only": True,
            "annotation_files_read": False,
            "label_files_read": False,
            "bbox_inputs": False,
            "polygon_inputs": False,
            "class_inputs": False,
        },
        "source_image_count": len(image_records),
        "split_counts": {
            split: sum(
                record["split"] == split for record in image_records
            )
            for split in SPLITS
        },
        "resumed_success_count": len(completed) - (
            len(pending) - len(errors)
        ),
        "new_attempt_count": len(pending),
        "qwen_success_count": qwen_success,
        "fallback_count": fallback_count,
        "error_count": len(errors),
        "errors": errors,
        "output": {
            "observations": _portable_path(observations_path),
            "size_bytes": observations_path.stat().st_size,
            "sha256": _sha256(observations_path),
        },
        "inputs": {
            "images_root": _portable_path(images_root),
            "split_inventory": [
                {
                    "source_image_id": record["source_image_id"],
                    "split": record["split"],
                    "image_path": record["image_path"],
                }
                for record in image_records
            ],
        },
        "runtime": {
            "workers": int(args.workers),
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": datetime.now().astimezone().isoformat(),
        },
        "limitations": [
            "Rule observations are deterministic full-image software sensors.",
            "Qwen observations, when generated, remain image-conditioned estimates.",
            "Neither mode is a synchronized physical sensor measurement.",
        ],
    }
    _atomic_json(output / "BUILD_REPORT.json", report)
    return report


def main() -> None:
    report = build(parse_args())
    print(
        json.dumps(
            {
                "state": report.get("state"),
                "valid": report.get("valid"),
                "mode": report.get("mode"),
                "source_image_count": report.get("source_image_count"),
                "split_counts": report.get("split_counts"),
                "qwen_success_count": report.get("qwen_success_count"),
                "fallback_count": report.get("fallback_count"),
                "error_count": report.get("error_count"),
                "output": report.get("output"),
                "runtime": report.get("runtime"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report.get("valid") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
