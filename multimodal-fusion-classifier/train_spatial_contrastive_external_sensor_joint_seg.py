"""Train v5 spatial reliability using the frozen v4 training machinery."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import mmfc.joint_seg_reliable as reliable_module
from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
import train_contrastive_reliable_external_sensor_joint_seg as v4


PARENT_NAME = (
    "yolo11s-seg-joint-reliable-contrastive-rule-dev-seed42"
)
DEFAULT_BASE_WEIGHTS = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-contrastive"
    / PARENT_NAME
    / "weights"
    / "best.pt"
)
DEFAULT_PROJECT = (
    PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg-spatial"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-spatial"
)
DEFAULT_NAME = (
    "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
)
PROTOCOL_NAME = (
    "joint_text_external_scene_sensor_spatial_yolo_segment_v5"
)


def parse_args():
    args = v4.parse_args()
    if Path(args.base_weights) == v4.DEFAULT_BASE_WEIGHTS:
        args.base_weights = DEFAULT_BASE_WEIGHTS
    if Path(args.project) == v4.DEFAULT_PROJECT:
        args.project = DEFAULT_PROJECT
    if Path(args.report_dir) == v4.DEFAULT_REPORT:
        args.report_dir = DEFAULT_REPORT
    if args.name == v4.DEFAULT_NAME:
        args.name = DEFAULT_NAME
    return args


def _protocol(
    args,
    paths: dict[str, Path],
    train_args: dict[str, Any],
    degradation: dict[str, Any],
) -> dict[str, Any]:
    build = json.loads(
        paths["sensor_build"].read_text(encoding="utf-8-sig")
    )
    source_paths = (
        paths["data"],
        paths["base_weights"],
        paths["sensor_observations"],
        paths["sensor_validation"],
        paths["sensor_build"],
        paths["text_prototypes"],
        Path(__file__).resolve(),
        PROJECT_ROOT / "train_contrastive_reliable_external_sensor_joint_seg.py",
        PROJECT_ROOT / "mmfc" / "joint_seg_spatial.py",
        PROJECT_ROOT / "mmfc" / "joint_seg_contrastive.py",
        PROJECT_ROOT / "mmfc" / "joint_seg_reliable.py",
        PROJECT_ROOT / "mmfc" / "joint_seg_external.py",
        PROJECT_ROOT / "mmfc" / "joint_seg.py",
        PROJECT_ROOT / "mmfc" / "model.py",
        PROJECT_ROOT / "mmfc" / "training.py",
    )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_protocol_frozen_before_training",
        "formal_paper_metric": False,
        "run_name": args.name,
        "seed": int(args.seed),
        "parent_checkpoint": {
            "path": paths["base_weights"].resolve().as_posix(),
            "sha256": sha256_file(paths["base_weights"]),
        },
        "architecture": {
            "base": "v4 contrastive reliability-aware joint head",
            "image_backbone_inputs": ["RGB"],
            "transport_channels": [
                "RGB",
                "4x4 sensor raster",
                "sensor availability",
            ],
            "rgb_sensor_predictor": (
                "P3/P4/P5 pooled directly to 4x4, concatenated, then "
                "spatial convolutional mean and uncertainty heads"
            ),
            "spatial_hidden_dim": 64,
            "mismatch_negative_generation": (
                "deterministic non-self batch donor selected from source ids"
            ),
            "mismatch_quality_target": 0.0,
            "mismatch_uses_annotations": False,
            "missing_sensor_fallback": "spatial RGB-predicted 4x4 soft sensor",
        },
        "sensor_source": {
            "mode": build.get("mode"),
            "model": build.get("model"),
            "source_image_count": build.get("source_image_count"),
            "full_image_only": True,
            "annotation_inputs": False,
            "hardware_measurement": False,
        },
        "software_sensor_degradation": degradation,
        "mismatch_probability": float(args.mismatch_probability),
        "train_args": train_args,
        "fusion_args": {
            "semantic_dim": int(args.semantic_dim),
            "sensor_hidden_dim": int(args.sensor_hidden_dim),
            "text_temperature": float(args.text_temperature),
            "sensor_reconstruction_weight": float(
                args.sensor_reconstruction_weight
            ),
            "sensor_calibration_weight": float(
                args.sensor_calibration_weight
            ),
            "reliability_calibration_weight": float(
                args.reliability_calibration_weight
            ),
            "reliability_hidden_dim": int(args.reliability_hidden_dim),
            "reliability_init": float(args.reliability_init),
        },
        "leakage_control": {
            "sensor_generation_reads_annotations": False,
            "metric_forward_reads_annotations": False,
            "spatial_predictor_reads_annotations": False,
            "mismatch_generation_reads_annotations": False,
            "mismatch_seed_inputs": [
                "run seed",
                "source image ids in current batch",
            ],
        },
        "claim_boundaries": [
            "This single-seed rule-sidecar run is mechanism development.",
            "Mismatches and degradations are software simulations.",
            "The observations are not physical sensor measurements.",
            "Formal claims require the preregistered Qwen three-seed protocol.",
        ],
        "source_bindings": {
            path.resolve().as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        },
    }


def run(args) -> dict[str, Any]:
    from mmfc.joint_seg_spatial import (
        SpatialReliableExternalSensorJointSegmentationModel,
    )

    original_protocol = v4._protocol
    original_model = (
        reliable_module.ReliableExternalSensorJointSegmentationModel
    )
    v4._protocol = _protocol
    reliable_module.ReliableExternalSensorJointSegmentationModel = (
        SpatialReliableExternalSensorJointSegmentationModel
    )
    try:
        result = v4.run(args)
    finally:
        v4._protocol = original_protocol
        reliable_module.ReliableExternalSensorJointSegmentationModel = (
            original_model
        )

    if result.get("state") == "complete":
        evaluation_path = Path(result["evaluation"])
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8-sig")
        )
        evaluation["evidence_role"] = (
            "rule_sidecar_spatial_contrastive_reliability_development"
        )
        temporary = evaluation_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(evaluation_path)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
