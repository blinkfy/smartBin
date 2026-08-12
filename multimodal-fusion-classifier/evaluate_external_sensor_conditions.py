"""Evaluate a deterministic shuffled-sidecar negative control."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from train_yolo_seg import _json_safe, _runtime_environment


DEFAULT_RUN_NAME = "yolo11s-seg-joint-external-rule-smoke"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-external"
    / DEFAULT_RUN_NAME
    / "weights"
    / "best.pt"
)
DEFAULT_OBSERVATIONS = (
    PROJECT_ROOT / "data" / "scene-sensor-sidecars" / "observations.jsonl"
)
DEFAULT_DATA = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "dataset.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
    / DEFAULT_RUN_NAME
    / "conditions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate shuffled external scene-sensor condition"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--sensor-observations", type=Path, default=DEFAULT_OBSERVATIONS
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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


def _atomic_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
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


def derange_sensor_records(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rotate sensor values within each split with no self assignments."""

    by_split: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        split = str(record.get("split", ""))
        source_id = str(record.get("source_image_id", ""))
        if split not in {"train", "val", "test"} or not source_id:
            raise ValueError("record has invalid split or source id")
        by_split.setdefault(split, []).append(record)
    transformed: list[dict[str, Any]] = []
    for split, split_records in sorted(by_split.items()):
        ordered = sorted(
            split_records, key=lambda item: str(item["source_image_id"])
        )
        if len(ordered) < 2:
            raise ValueError(f"split {split} cannot be deranged")
        donors = ordered[1:] + ordered[:1]
        for destination, donor in zip(ordered, donors):
            result = deepcopy(dict(destination))
            result["grid"] = deepcopy(donor["grid"])
            result["normalized_grid"] = deepcopy(
                donor["normalized_grid"]
            )
            result["properties"] = deepcopy(donor["properties"])
            result["source"] = {
                **dict(destination["source"]),
                "diagnostic_condition": "within_split_cyclic_derangement",
                "diagnostic_donor_source_image_id": donor[
                    "source_image_id"
                ],
            }
            transformed.append(result)
    if any(
        record["source_image_id"]
        == record["source"]["diagnostic_donor_source_image_id"]
        for record in transformed
    ):
        raise ValueError("derangement contains a self assignment")
    return sorted(
        transformed, key=lambda item: str(item["source_image_id"])
    )


def _metric_deltas(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(reference[key])
        for key in (
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(M)",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
        )
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    from mmfc.joint_seg_external import (
        ExternalSensorSegmentationValidator,
    )

    checkpoint = _resolve(args.checkpoint)
    observations = _resolve(args.sensor_observations)
    data = _resolve(args.data)
    output = _resolve(args.output)
    for path in (checkpoint, observations, data):
        if not path.is_file():
            raise FileNotFoundError(f"condition input is missing: {path}")
    if args.image_size <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("invalid image, batch, or workers value")
    report_run = output.parent
    evaluation_path = report_run / "evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError("base external-sensor evaluation is missing")
    base = _read_json(evaluation_path)
    if (
        base.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint)
        or base.get("sensor_observations", {}).get("sha256")
        != sha256_file(observations)
    ):
        raise ValueError("base evaluation input bindings differ")

    original = _read_jsonl(observations)
    shuffled = derange_sensor_records(original)
    shuffled_path = output / "SHUFFLED_OBSERVATIONS.jsonl"
    _atomic_jsonl(shuffled_path, shuffled)
    test_records = [
        record for record in shuffled if record.get("split") == "test"
    ]
    if len(test_records) != 570:
        raise ValueError("shuffled test split does not contain 570 images")

    validators = []

    def validator_factory(*factory_args: Any, **factory_kwargs: Any):
        validator = ExternalSensorSegmentationValidator(
            *factory_args,
            **factory_kwargs,
            sensor_observations=shuffled_path,
        )
        validators.append(validator)
        return validator

    model = YOLO(str(checkpoint))
    metrics = model.val(
        validator=validator_factory,
        data=str(data),
        split="test",
        imgsz=int(args.image_size),
        batch=int(args.batch_size),
        device=str(args.device),
        workers=int(args.workers),
        project=str(output),
        name="test-shuffled-sensor",
        exist_ok=False,
        plots=False,
        save_json=False,
        verbose=True,
    )
    validator = validators[0]
    if validator.external_sensor_sample_count != 570:
        raise ValueError("shuffled evaluation did not consume 570 sidecars")
    auxiliary = model.model.joint_fusion.last_aux
    if not bool((auxiliary["external_sensor_used"] == 1).all()):
        raise ValueError("shuffled external sensor was not used")

    external = base["external_sensor"]["metrics"]
    fallback = base["rgb_fallback"]["metrics"]
    shuffled_metrics = _json_safe(metrics.results_dict)
    result = {
        "schema_version": 1,
        "valid": True,
        "evidence_role": (
            "integration_smoke_negative_control"
            if base.get("sensor_observations", {}).get("mode") == "rule"
            else "formal_negative_control_candidate"
        ),
        "formal_paper_metric": False,
        "condition": {
            "name": "within_split_cyclic_derangement",
            "self_assignment_count": 0,
            "source_image_count": len(shuffled),
            "test_source_image_count": len(test_records),
            "split_preserved": True,
            "sensor_distribution_preserved": True,
        },
        "metrics": {
            "external_matched": external,
            "rgb_fallback": fallback,
            "external_shuffled": shuffled_metrics,
            "matched_minus_shuffled": _metric_deltas(
                external, shuffled_metrics
            ),
            "matched_minus_fallback": _metric_deltas(
                external, fallback
            ),
        },
        "speed_ms_per_image": _json_safe(metrics.speed),
        "runtime": _runtime_environment(),
        "bindings": {
            "checkpoint": {
                "path": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            },
            "original_observations": {
                "path": observations.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": observations.stat().st_size,
                "sha256": sha256_file(observations),
            },
            "shuffled_observations": {
                "path": shuffled_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": shuffled_path.stat().st_size,
                "sha256": sha256_file(shuffled_path),
            },
            "base_evaluation": {
                "path": evaluation_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": evaluation_path.stat().st_size,
                "sha256": sha256_file(evaluation_path),
            },
            "evaluator_source": {
                "path": Path(__file__).resolve().relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "size_bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "claim_boundaries": [
            "The source checkpoint has one training epoch.",
            "The sensor source is scene-rule-v1.",
            "This negative control is excluded from formal paper tables.",
        ],
    }
    _atomic_json(output / "CONDITION_DIAGNOSTICS.json", result)
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "matched_minus_shuffled": result["metrics"][
                    "matched_minus_shuffled"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
