"""Validate an external blind-confirmation bundle before evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mmfc.external_blind import (
    sha256_file,
    validate_external_blind_bundle,
)
from mmfc.training import PROJECT_ROOT


DEFAULT_BUNDLE = PROJECT_ROOT / "data" / "external-blind-v1"
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "external-blind-v1"
    / "PROTOCOL.json"
)
DEFAULT_INTEGRITY = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SOURCE_IMAGE_INTEGRITY_CURRENT.json"
)
DEFAULT_DHASH = (
    PROJECT_ROOT
    / "data"
    / "external-blind-governance"
    / "development_dhash_index.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the preregistered external blind intake"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--development-integrity",
        type=Path,
        default=DEFAULT_INTEGRITY,
    )
    parser.add_argument(
        "--development-dhash-index",
        type=Path,
        default=DEFAULT_DHASH,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _resolve(args.bundle)
    protocol = _resolve(args.protocol)
    integrity = _resolve(args.development_integrity)
    dhash = _resolve(args.development_dhash_index)
    result = validate_external_blind_bundle(
        bundle,
        protocol_path=protocol,
        development_integrity_path=integrity,
        development_dhash_index_path=dhash,
    )
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["validator"] = {
        "path": Path(__file__).resolve().relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "size_bytes": Path(__file__).stat().st_size,
        "sha256": sha256_file(Path(__file__)),
    }
    output = (
        _resolve(args.output)
        if args.output is not None
        else bundle / "VALIDATION_REPORT.json"
    )
    _atomic_json(output, result)
    result["report_path"] = output.as_posix()
    return result


def main() -> None:
    result = validate(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "error_count": result["error_count"],
                "warning_count": result["warning_count"],
                "sample_count": result.get("sample_count", 0),
                "source_image_count": result.get(
                    "source_image_count", 0
                ),
                "report_path": result["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
