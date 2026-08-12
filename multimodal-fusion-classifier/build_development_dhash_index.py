"""Build a filename-free dHash index for external-blind leakage checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mmfc.external_blind import (
    difference_hash,
    read_json,
    read_jsonl,
    sha256_file,
)
from mmfc.training import PROJECT_ROOT, _resolve_image_path


INTEGRITY = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SOURCE_IMAGE_INTEGRITY_CURRENT.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "external-blind-governance"
    / "development_dhash_index.jsonl"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "EXTERNAL_BLIND_DEVELOPMENT_DHASH_INDEX.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the development dHash leakage index"
    )
    parser.add_argument("--integrity", type=Path, default=INTEGRITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build(
    integrity_path: Path = INTEGRITY,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    integrity_path = integrity_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    integrity = read_json(integrity_path)
    if (
        integrity.get("valid") is not True
        or integrity.get("summary", {}).get("source_image_count") != 5990
    ):
        raise ValueError("development source integrity is invalid")
    image_by_source: dict[str, Path] = {}
    for binding in integrity["manifest_bindings"]:
        manifest = PROJECT_ROOT / binding["path"]
        if sha256_file(manifest) != binding["sha256"]:
            raise ValueError("development manifest binding drifted")
        for record in read_jsonl(manifest):
            source_id = str(record["source_image_id"])
            path = _resolve_image_path(
                record["image_path"], manifest.parent
            ).resolve()
            previous = image_by_source.setdefault(source_id, path)
            if previous != path:
                raise ValueError(
                    f"source image {source_id} maps to multiple paths"
                )
    integrity_by_id = {
        str(item["source_image_id"]): item
        for item in integrity["images"]
    }
    if set(image_by_source) != set(integrity_by_id):
        raise ValueError("development source ids do not match integrity audit")

    records: list[dict[str, Any]] = []
    for source_id in sorted(image_by_source):
        path = image_by_source[source_id]
        expected_sha = str(integrity_by_id[source_id]["effective_sha256"])
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"development image drifted: {source_id}")
        records.append(
            {
                "schema_version": 1,
                "source_image_id": source_id,
                "sha256": actual_sha,
                "dhash": difference_hash(path),
            }
        )
    _atomic_jsonl(output_path, records)
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "PIL_EXIF_RGB_to_grayscale_9x8_difference_hash_64bit",
        "source_image_count": len(records),
        "filename_persisted": False,
        "original_image_write_performed": False,
        "bindings": {
            "integrity": {
                "path": integrity_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": integrity_path.stat().st_size,
                "sha256": sha256_file(integrity_path),
            },
            "output": {
                "path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            },
            "builder_source": {
                "path": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__)),
            },
            "hash_source": {
                "path": (
                    PROJECT_ROOT / "mmfc" / "external_blind.py"
                ).relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": (
                    PROJECT_ROOT / "mmfc" / "external_blind.py"
                ).stat().st_size,
                "sha256": sha256_file(
                    PROJECT_ROOT / "mmfc" / "external_blind.py"
                ),
            },
        },
    }
    _atomic_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = build(args.integrity, args.output, args.report)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "source_image_count": report["source_image_count"],
                "filename_persisted": report["filename_persisted"],
                "output_sha256": report["bindings"]["output"]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
