"""Build a deterministic within-split Qwen sidecar derangement."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from evaluate_external_sensor_conditions import derange_sensor_records
from mmfc.model import PROJECT_ROOT


DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "scene-qwen-sensor-sidecars" / "observations.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "scene-qwen-sensor-sidecars"
    / "SHUFFLED_OBSERVATIONS.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the formal within-split Qwen negative control"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            records.append(value)
    return records


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(
    path: Path, records: list[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(
            records, key=lambda item: str(item["source_image_id"])
        ):
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")
    temporary.replace(path)


def _payload_key(record: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "grid": record["grid"],
            "normalized_grid": record["normalized_grid"],
            "properties": record["properties"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def validate_derangement(
    original: list[Mapping[str, Any]],
    shuffled: list[Mapping[str, Any]],
) -> dict[str, Any]:
    original_by_id = {
        str(record["source_image_id"]): record for record in original
    }
    shuffled_by_id = {
        str(record["source_image_id"]): record for record in shuffled
    }
    errors: list[str] = []
    if len(original_by_id) != len(original):
        errors.append("original observations contain duplicate source ids")
    if set(original_by_id) != set(shuffled_by_id):
        errors.append("source image identity set changed")
    split_counts = {"train": 0, "val": 0, "test": 0}
    self_assignments = 0
    split_violations = 0
    for source_id, record in shuffled_by_id.items():
        split = str(record.get("split", ""))
        if split in split_counts:
            split_counts[split] += 1
        else:
            errors.append(f"invalid split for {source_id}")
        source = record.get("source", {})
        donor_id = str(
            source.get("diagnostic_donor_source_image_id", "")
        )
        if donor_id == source_id:
            self_assignments += 1
        donor = original_by_id.get(donor_id)
        if donor is None or donor.get("split") != split:
            split_violations += 1
    payload_preserved = Counter(map(_payload_key, original)) == Counter(
        map(_payload_key, shuffled)
    )
    if self_assignments:
        errors.append("derangement contains self assignments")
    if split_violations:
        errors.append("derangement crosses split boundaries")
    if not payload_preserved:
        errors.append("sensor payload multiset changed")
    return {
        "valid": not errors,
        "errors": errors,
        "source_image_count": len(shuffled),
        "split_counts": split_counts,
        "self_assignment_count": self_assignments,
        "split_violation_count": split_violations,
        "sensor_payload_multiset_preserved": payload_preserved,
    }


def build(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    input_path = _resolve(input_path)
    output_path = _resolve(output_path)
    validation_path = input_path.parent / "VALIDATION_REPORT.json"
    build_path = input_path.parent / "BUILD_REPORT.json"
    for path in (input_path, validation_path, build_path):
        if not path.is_file():
            raise FileNotFoundError(f"Qwen data input is missing: {path}")
    validation = _read_json(validation_path)
    build_report = _read_json(build_path)
    if (
        validation.get("valid") is not True
        or validation.get("mode") != "qwen"
        or validation.get("source_image_count") != 5990
        or validation.get("split_counts")
        != {"train": 4823, "val": 597, "test": 570}
        or validation.get("observations_sha256") != _sha256(input_path)
        or build_report.get("mode") != "qwen"
        or build_report.get("output", {}).get("sha256")
        != _sha256(input_path)
    ):
        raise ValueError("Qwen sidecars do not pass the formal data gate")

    original = _read_jsonl(input_path)
    shuffled = derange_sensor_records(original)
    audit = validate_derangement(original, shuffled)
    if not audit["valid"]:
        raise ValueError("; ".join(audit["errors"]))
    _atomic_jsonl(output_path, shuffled)
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "method": "within_split_one_position_cyclic_derangement",
        "source_image_count": audit["source_image_count"],
        "split_counts": audit["split_counts"],
        "self_assignment_count": audit["self_assignment_count"],
        "split_violation_count": audit["split_violation_count"],
        "split_membership_preserved": True,
        "sensor_payload_multiset_preserved": audit[
            "sensor_payload_multiset_preserved"
        ],
        "bindings": {
            "original_observations": _binding(input_path),
            "shuffled_observations": _binding(output_path),
            "qwen_build": _binding(build_path),
            "qwen_validation": _binding(validation_path),
            "builder_source": _binding(Path(__file__)),
        },
    }
    _atomic_json(output_path.parent / "SHUFFLE_REPORT.json", report)
    return report


def main() -> None:
    args = parse_args()
    result = build(args.input, args.output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "source_image_count": result["source_image_count"],
                "self_assignment_count": result["self_assignment_count"],
                "sensor_payload_multiset_preserved": result[
                    "sensor_payload_multiset_preserved"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
