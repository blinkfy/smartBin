"""Validate full-image soft-sensor sidecars and their no-annotation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mmfc.model import PROJECT_ROOT
from mmfc.sensors import PROPERTY_RANGES, contains_leakage_key


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "scene-sensor-sidecars"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate full-image scene sensor sidecars"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_grid(value: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(row, list) or len(row) != 4 for row in value
        )
    ):
        return False
    try:
        values = [float(item) for row in value for item in row]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in values)


def validate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    report_path = output / "BUILD_REPORT.json"
    observations_path = output / "observations.jsonl"
    errors: list[str] = []
    if not report_path.is_file() or not observations_path.is_file():
        return {
            "schema_version": 1,
            "valid": False,
            "error_count": 1,
            "errors": ["build report or observations are missing"],
        }
    build = json.loads(report_path.read_text(encoding="utf-8-sig"))
    records: list[dict[str, Any]] = []
    with observations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: root is not an object")
                continue
            records.append(value)
    seen: set[str] = set()
    split_counts = {"train": 0, "val": 0, "test": 0}
    for index, record in enumerate(records):
        source_id = record.get("source_image_id")
        prefix = f"record {index}"
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{prefix}: invalid source image id")
            continue
        if source_id in seen:
            errors.append(f"{prefix}: duplicate source image id {source_id}")
        seen.add(source_id)
        split = record.get("split")
        if split not in split_counts:
            errors.append(f"{prefix}: invalid split")
        else:
            split_counts[split] += 1
        image_path = record.get("image_path")
        identity_path = Path(str(image_path))
        if not identity_path.is_absolute():
            identity_path = PROJECT_ROOT / identity_path
        path = _resolve(Path(str(image_path)))
        if not path.is_file() or identity_path.stem != source_id:
            errors.append(f"{prefix}: image identity mismatch")
        if contains_leakage_key(record):
            errors.append(f"{prefix}: classification leakage key found")
        if not _valid_grid(record.get("grid")) or not _valid_grid(
            record.get("normalized_grid")
        ):
            errors.append(f"{prefix}: invalid 4x4 grid")
        properties = record.get("properties")
        if not isinstance(properties, Mapping):
            errors.append(f"{prefix}: properties are missing")
        else:
            for name, (lower, upper) in PROPERTY_RANGES.items():
                try:
                    value = float(properties[name])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{prefix}: invalid property {name}")
                    continue
                if not math.isfinite(value) or not lower <= value <= upper:
                    errors.append(f"{prefix}: out-of-range property {name}")
        source = record.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("observation_scope") != "full_image"
            or source.get("full_image_only") is not True
            or source.get("annotation_inputs") != []
        ):
            errors.append(f"{prefix}: source contract is invalid")
    source_contract = build.get("source_contract", {})
    if not all(
        (
            source_contract.get("full_image_only") is True,
            source_contract.get("annotation_files_read") is False,
            source_contract.get("label_files_read") is False,
            source_contract.get("bbox_inputs") is False,
            source_contract.get("polygon_inputs") is False,
            source_contract.get("class_inputs") is False,
        )
    ):
        errors.append("build source contract is invalid")
    if len(records) != build.get("source_image_count"):
        errors.append("record count differs from build report")
    if split_counts != build.get("split_counts"):
        errors.append("split counts differ from build report")
    if (
        observations_path.stat().st_size
        != build.get("output", {}).get("size_bytes")
        or _sha256(observations_path)
        != build.get("output", {}).get("sha256")
    ):
        errors.append("observation file binding mismatch")
    result = {
        "schema_version": 1,
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "mode": build.get("mode"),
        "source_image_count": len(records),
        "split_counts": split_counts,
        "qwen_success_count": build.get("qwen_success_count"),
        "fallback_count": build.get("fallback_count"),
        "source_contract": source_contract,
        "observations_sha256": _sha256(observations_path),
    }
    path = output / "VALIDATION_REPORT.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return result


def main() -> None:
    result = validate(parse_args().output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
