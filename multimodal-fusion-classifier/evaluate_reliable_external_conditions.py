"""Evaluate reliability-aware joint segmentation under sensor conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import torch

from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from train_reliable_external_sensor_joint_seg import DEFAULT_NAME
from train_yolo_seg import _json_safe, _runtime_environment


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-reliable"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-reliable"
    / DEFAULT_NAME
    / "weights"
    / "best.pt"
)
DEFAULT_OBSERVATIONS = (
    PROJECT_ROOT / "data" / "scene-sensor-sidecars" / "observations.jsonl"
)
DEFAULT_SHUFFLED = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
    / "yolo11s-seg-joint-external-rule-smoke"
    / "conditions"
    / "SHUFFLED_OBSERVATIONS.jsonl"
)
DEFAULT_DATA = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "dataset.yaml"
)
DEFAULT_OUTPUT = REPORT_ROOT / DEFAULT_NAME / "RELIABILITY_CONDITIONS.json"


CONDITIONS: dict[str, dict[str, Any]] = {
    "clean_matched": {"enabled": False},
    "missing": {
        "enabled": True,
        "missing_probability": 1.0,
        "quality_scale": 0.25,
    },
    "noise_0_15": {
        "enabled": True,
        "noise_probability": 1.0,
        "noise_std_range": [0.15, 0.15],
        "quality_scale": 0.25,
    },
    "bias_drift_0_25": {
        "enabled": True,
        "bias_probability": 1.0,
        "bias_strength": 0.25,
        "bias_min_strength": 0.25,
        "quality_scale": 0.25,
    },
    "stuck": {
        "enabled": True,
        "stuck_probability": 1.0,
        "quality_scale": 0.25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v3 reliability-aware sensor conditions"
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


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def summarize_gate(values: torch.Tensor) -> dict[str, float]:
    samples = values.detach().float().reshape(-1).cpu().tolist()
    if not samples:
        raise ValueError("gate diagnostic tensor is empty")
    return {
        "mean": mean(samples),
        "population_standard_deviation": pstdev(samples),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _metric_deltas(
    candidate: Mapping[str, float],
    reference: Mapping[str, float],
) -> dict[str, float]:
    keys = (
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95(M)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
    )
    return {
        key: float(candidate[key]) - float(reference[key]) for key in keys
    }


def _gate_collector(model: Any):
    captured: dict[str, list[torch.Tensor]] = {
        "sensor_raw_reliability": [],
        "sensor_reliability": [],
        "sensor_consistency": [],
        "sensor_discrepancy": [],
        "sensor_available": [],
    }

    def hook(module: Any, inputs: Any, output: Any) -> None:
        auxiliary = module.last_aux
        for name in captured:
            value = auxiliary.get(name)
            if isinstance(value, torch.Tensor):
                captured[name].append(value.detach().float().cpu())

    handle = model.model.joint_fusion.register_forward_hook(hook)
    return captured, handle


def _final_gate_summary(
    captured: Mapping[str, list[torch.Tensor]],
    *,
    sample_count: int,
) -> dict[str, dict[str, float]]:
    result = {}
    for name, pieces in captured.items():
        if not pieces:
            raise ValueError(f"gate diagnostic was not captured: {name}")
        values = torch.cat(pieces, dim=0)
        if values.shape[0] < sample_count:
            raise ValueError(f"gate diagnostic sample count is short: {name}")
        result[name] = summarize_gate(values[-sample_count:])
    return result


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
            raise FileNotFoundError(f"reliability input is missing: {path}")
    if args.batch_size <= 0 or args.workers < 0 or args.image_size <= 0:
        raise ValueError("invalid evaluation runtime parameters")
    protocol = _read_json(protocol_path)
    base_evaluation = _read_json(evaluation_path)
    if (
        protocol.get("protocol")
        != "joint_text_external_scene_sensor_reliability_yolo_segment_v3"
        or protocol.get("formal_paper_metric") is not False
        or base_evaluation.get("valid") is not True
        or base_evaluation.get("checkpoint", {}).get("sha256")
        != sha256_file(checkpoint)
        or base_evaluation.get("sensor_observations", {}).get("sha256")
        != sha256_file(observations)
    ):
        raise ValueError("reliability run bindings are inconsistent")

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
        validator = validators[0]
        if validator.external_sensor_sample_count != 570:
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
        if name not in {"clean_matched", "rgb_fallback"}
    }
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "evidence_role": "rule_sidecar_reliability_development",
        "test_source_image_count": 570,
        "conditions": condition_results,
        "comparisons": comparisons,
        "gate_ordering_clean_above_condition": gate_ordering,
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
        "claim_boundaries": [
            "This is a single-seed five-epoch rule-sidecar development result.",
            "The robustness conditions are software simulations.",
            "The observations are not physical sensor measurements.",
            "The result is excluded from the preregistered formal Qwen table.",
        ],
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
