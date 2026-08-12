"""Build a deterministic negative control for distilled scene sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_external_sensor_conditions import derange_sensor_records
from mmfc.model import PROJECT_ROOT
from prepare_qwen_sensor_derangement import (
    _atomic_json,
    _atomic_jsonl,
    _binding,
    _read_json,
    _read_jsonl,
    _resolve,
    _sha256,
    validate_derangement,
)


MODE = "object-qwen-scene-distilled"
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "scene-qwen-distilled-sidecars"
    / "observations.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "scene-qwen-distilled-sidecars"
    / "SHUFFLED_OBSERVATIONS.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the distilled-scene within-split negative control"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _valid_source_contract(value: Any) -> bool:
    return isinstance(value, dict) and all(
        value.get(key) is expected
        for key, expected in {
            "full_image_only": True,
            "annotation_files_read": False,
            "label_files_read": False,
            "bbox_inputs": False,
            "polygon_inputs": False,
            "class_inputs": False,
        }.items()
    )


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
            raise FileNotFoundError(f"distilled sidecar input is missing: {path}")

    validation = _read_json(validation_path)
    build_report = _read_json(build_path)
    expected_splits = {"train": 4823, "val": 597, "test": 570}
    if (
        validation.get("valid") is not True
        or validation.get("mode") != MODE
        or validation.get("source_image_count") != 5990
        or validation.get("split_counts") != expected_splits
        or validation.get("observations_sha256") != _sha256(input_path)
        or not _valid_source_contract(validation.get("source_contract"))
        or build_report.get("valid") is not True
        or build_report.get("mode") != MODE
        or build_report.get("output", {}).get("sha256") != _sha256(input_path)
        or not _valid_source_contract(build_report.get("source_contract"))
    ):
        raise ValueError("distilled sidecars do not pass the data gate")

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
        "mode": MODE,
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
            "distilled_build": _binding(build_path),
            "distilled_validation": _binding(validation_path),
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
