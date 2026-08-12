"""Evaluate the v4 contrastive reliability model under sensor conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_reliable_external_conditions import (
    CONDITIONS,
    DEFAULT_DATA,
    DEFAULT_OBSERVATIONS,
    DEFAULT_SHUFFLED,
    _atomic_json,
    _binding,
    _final_gate_summary,
    _gate_collector,
    _metric_deltas,
    _read_json,
    _resolve,
)
from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from train_contrastive_reliable_external_sensor_joint_seg import DEFAULT_NAME
from train_yolo_seg import _json_safe, _runtime_environment


PROTOCOL_NAME = (
    "joint_text_external_scene_sensor_contrastive_yolo_segment_v4"
)
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-contrastive"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-contrastive"
    / DEFAULT_NAME
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT = REPORT_ROOT / DEFAULT_NAME / "RELIABILITY_CONDITIONS.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v4 contrastive sensor conditions"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--sensor-observations", type=Path, default=DEFAULT_OBSERVATIONS
    )
    parser.add_argument(
        "--shuffled-observations", type=Path, default=DEFAULT_SHUFFLED
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_run_bindings(
    protocol: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    checkpoint_sha256: str,
    observations_sha256: str,
) -> None:
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or protocol.get("formal_paper_metric") is not False
        or evaluation.get("valid") is not True
        or evaluation.get("formal_paper_metric") is not False
        or evaluation.get("checkpoint", {}).get("sha256")
        != checkpoint_sha256
        or evaluation.get("sensor_observations", {}).get("sha256")
        != observations_sha256
    ):
        raise ValueError("contrastive run bindings are inconsistent")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    from mmfc.joint_seg_external import load_scene_sensor_index
    from mmfc.joint_seg_reliable import (
        ReliableExternalSensorSegmentationValidator,
    )

    checkpoint = _resolve(args.checkpoint)
    observations = _resolve(args.sensor_observations)
    shuffled = _resolve(args.shuffled_observations)
    data = _resolve(args.data)
    output = _resolve(args.output)
    report_run = output.parent
    protocol_path = report_run / "PROTOCOL.json"
    evaluation_path = report_run / "evaluation.json"
    for path in (
        checkpoint,
        observations,
        shuffled,
        data,
        protocol_path,
        evaluation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"contrastive input is missing: {path}")
    if args.batch_size <= 0 or args.workers < 0 or args.image_size <= 0:
        raise ValueError("invalid evaluation runtime parameters")

    protocol = _read_json(protocol_path)
    base_evaluation = _read_json(evaluation_path)
    validate_run_bindings(
        protocol,
        base_evaluation,
        checkpoint_sha256=sha256_file(checkpoint),
        observations_sha256=sha256_file(observations),
    )
    matched_index = load_scene_sensor_index(observations)
    shuffled_index = load_scene_sensor_index(shuffled)
    condition_specs = {
        **{
            name: {
                "observations": matched_index,
                "degradation": degradation,
            }
            for name, degradation in CONDITIONS.items()
        },
        "shuffled": {
            "observations": shuffled_index,
            "degradation": {"enabled": False},
        },
    }

    condition_results: dict[str, Any] = {}
    for name, specification in condition_specs.items():
        validators = []

        def validator_factory(*factory_args: Any, **factory_kwargs: Any):
            validator = ReliableExternalSensorSegmentationValidator(
                *factory_args,
                **factory_kwargs,
                sensor_observations=specification["observations"],
                degradation=specification["degradation"],
                degradation_seed=int(args.seed),
            )
            validators.append(validator)
            return validator

        model = YOLO(str(checkpoint))
        captured, hook = _gate_collector(model)
        try:
            metrics = model.val(
                validator=validator_factory,
                data=str(data),
                split="test",
                imgsz=int(args.image_size),
                batch=int(args.batch_size),
                device=str(args.device),
                workers=int(args.workers),
                project=str(report_run / "condition-runs"),
                name=name,
                exist_ok=False,
                plots=False,
                save_json=False,
                verbose=False,
            )
        finally:
            hook.remove()
        if validators[0].external_sensor_sample_count != 570:
            raise ValueError(f"condition {name} did not consume 570 images")
        condition_results[name] = {
            "metrics": _json_safe(metrics.results_dict),
            "speed_ms_per_image": _json_safe(metrics.speed),
            "gate": _final_gate_summary(captured, sample_count=570),
            "degradation": specification["degradation"],
        }

    fallback_model = YOLO(str(checkpoint))
    captured, hook = _gate_collector(fallback_model)
    try:
        fallback_metrics = fallback_model.val(
            data=str(data),
            split="test",
            imgsz=int(args.image_size),
            batch=int(args.batch_size),
            device=str(args.device),
            workers=int(args.workers),
            project=str(report_run / "condition-runs"),
            name="rgb_fallback",
            exist_ok=False,
            plots=False,
            save_json=False,
            verbose=False,
        )
    finally:
        hook.remove()
    condition_results["rgb_fallback"] = {
        "metrics": _json_safe(fallback_metrics.results_dict),
        "speed_ms_per_image": _json_safe(fallback_metrics.speed),
        "gate": _final_gate_summary(captured, sample_count=570),
        "degradation": None,
    }

    clean = condition_results["clean_matched"]
    comparisons = {
        f"clean_minus_{name}": _metric_deltas(
            clean["metrics"], result["metrics"]
        )
        for name, result in condition_results.items()
        if name != "clean_matched"
    }
    clean_raw = clean["gate"]["sensor_raw_reliability"]["mean"]
    gate_ordering = {
        name: clean_raw
        > result["gate"]["sensor_raw_reliability"]["mean"]
        for name, result in condition_results.items()
        if name not in {"clean_matched", "rgb_fallback", "missing"}
    }
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "evidence_role": (
            "rule_sidecar_contrastive_reliability_development"
        ),
        "test_source_image_count": 570,
        "conditions": condition_results,
        "comparisons": comparisons,
        "gate_ordering_clean_above_observed_corruption": gate_ordering,
        "gate_ordering_all_passed": all(gate_ordering.values()),
        "runtime": _runtime_environment(),
        "bindings": {
            "checkpoint": _binding(checkpoint),
            "observations": _binding(observations),
            "shuffled_observations": _binding(shuffled),
            "protocol": _binding(protocol_path),
            "base_evaluation": _binding(evaluation_path),
            "evaluator_source": _binding(Path(__file__)),
        },
        "claim_boundaries": protocol["claim_boundaries"],
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "gate_ordering_all_passed": result[
                    "gate_ordering_all_passed"
                ],
                "clean_minus_shuffled": result["comparisons"][
                    "clean_minus_shuffled"
                ],
                "clean_minus_rgb_fallback": result["comparisons"][
                    "clean_minus_rgb_fallback"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
