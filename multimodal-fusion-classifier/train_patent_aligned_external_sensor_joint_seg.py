"""Train the v6 patent-aligned reliable native Segment head."""

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
    "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
)
DEFAULT_BASE_WEIGHTS = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-spatial"
    / PARENT_NAME
    / "weights"
    / "best.pt"
)
DEFAULT_PROJECT = (
    PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg-patent"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent"
)
DEFAULT_NAME = (
    "yolo11s-seg-joint-patent-stack-rule-dev-seed42"
)
PROTOCOL_NAME = (
    "joint_text_external_scene_sensor_patent_stack_yolo_segment_v6"
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


def _patent_document() -> Path:
    matches = sorted(PROJECT_ROOT.glob("2026.4.3-1*.docx"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one 2026.4.3-1 patent DOCX"
        )
    return matches[0]


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
        _patent_document(),
        Path(__file__).resolve(),
        PROJECT_ROOT / "train_contrastive_reliable_external_sensor_joint_seg.py",
        PROJECT_ROOT / "mmfc" / "joint_seg_patent.py",
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
        "patent_document": {
            "path": _patent_document().resolve().as_posix(),
            "sha256": sha256_file(_patent_document()),
        },
        "architecture": {
            "base": "v5 spatial contrastive reliability head",
            "image_backbone": "YOLO11s-seg RGB P3/P4/P5",
            "image_backbone_input_channels": 3,
            "transport_channels": [
                "RGB",
                "4x4 sensor raster",
                "sensor availability",
            ],
            "visual_scale_projection": "P3/P4/P5 pooled and projected to 512",
            "sensor_projection": (
                "4x4 reliability-blended grid upsampled to 16x16, "
                "1x1 convolution to 512, LeakyReLU, global pooling"
            ),
            "text_projection": (
                "all 19 frozen CLIP ViT-B/16 class prototypes queried "
                "without ground-truth labels"
            ),
            "independent_adapters": {
                "count": 3,
                "dimension": "512->512",
                "activation": "LeakyReLU",
                "dropout": 0.3,
            },
            "film": {
                "gamma_initial": 1.0,
                "beta_initial": 0.1,
                "shape": [3, 512],
            },
            "cosine_modality_weighting": True,
            "deep_interaction": {
                "down_projection": "512->128",
                "attention_heads": 8,
                "tokens": ["visual", "text", "sensor"],
                "up_projection": "128->512",
                "residual_connection": True,
            },
            "dynamic_multiscale_fusion": {
                "scales": ["P3", "P4", "P5"],
                "initial_prior": [0.3, 0.5, 0.2],
            },
            "segment_head_conditioning": (
                "final 512-dimensional fusion feature generates "
                "identity-initialized FiLM parameters for P3/P4/P5 "
                "immediately before the native Segment head"
            ),
            "missing_sensor_fallback": "spatial RGB-predicted 4x4 soft sensor",
            "head_outputs": ["boxes", "classes", "instance masks"],
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
            "patent_feature_dim": 512,
            "patent_attention_dim": 128,
            "patent_attention_heads": 8,
            "patent_dropout": 0.3,
        },
        "claim_level_alignment": {
            "claim_1_s2_yolo_clip_sensor_features": True,
            "claim_1_s3_complete_fusion_stack": True,
            "claim_1_s4_native_detection_and_segmentation_head": True,
            "claim_2_rgb_d_depth_input": False,
            "claim_2_physical_sensor_array": False,
            "medical_waste_taxonomy": False,
        },
        "leakage_control": {
            "sensor_generation_reads_annotations": False,
            "metric_forward_reads_annotations": False,
            "text_query_reads_ground_truth_class": False,
            "mismatch_generation_reads_annotations": False,
        },
        "claim_boundaries": [
            "This is a single-seed rule-sidecar mechanism-development run.",
            "The input is RGB, not RGB-D or a depth image.",
            "The sensor observation is image-conditioned and not hardware.",
            "The taxonomy contains 19 general-waste classes, not medical waste.",
            "Software degradation does not establish physical robustness.",
            "Formal claims still require the preregistered Qwen three-seed protocol.",
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
    from mmfc.joint_seg_patent import (
        PatentAlignedReliableJointSegmentationModel,
    )

    original_protocol = v4._protocol
    original_model = (
        reliable_module.ReliableExternalSensorJointSegmentationModel
    )
    v4._protocol = _protocol
    reliable_module.ReliableExternalSensorJointSegmentationModel = (
        PatentAlignedReliableJointSegmentationModel
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
            "rule_sidecar_patent_stack_mechanism_development"
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
