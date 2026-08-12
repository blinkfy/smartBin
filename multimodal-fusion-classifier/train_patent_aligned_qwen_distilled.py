"""Adapt the v6 patent stack to RGB-only object-Qwen-distilled sidecars."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import train_patent_aligned_external_sensor_joint_seg as v6
from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from train_external_sensor_joint_seg import DEFAULT_OBSERVATIONS


RUN_PROTOCOL_NAME = (
    "joint_text_external_scene_sensor_patent_stack_"
    "object_qwen_distilled_v6"
)
REGISTERED_PROTOCOL_NAME = (
    "object_qwen_distilled_patent_stack_development_v6"
)
DEFAULT_SENSOR_OBSERVATIONS = (
    PROJECT_ROOT
    / "data"
    / "scene-qwen-distilled-sidecars"
    / "observations.jsonl"
)
DEFAULT_PROJECT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-patent-qwen-distilled"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent-qwen-distilled"
)
DEFAULT_NAME = (
    "yolo11s-seg-joint-patent-qwen-distilled-adapt-seed42"
)
REGISTERED_PROTOCOL = (
    DEFAULT_REPORT / "DEVELOPMENT_PROTOCOL.json"
)
BASE_PROTOCOL_BUILDER = v6._protocol


def _specified(option: str) -> bool:
    return any(
        token == option or token.startswith(option + "=")
        for token in sys.argv[1:]
    )


def parse_args():
    args = v6.parse_args()
    if Path(args.sensor_observations) == DEFAULT_OBSERVATIONS:
        args.sensor_observations = DEFAULT_SENSOR_OBSERVATIONS
    if Path(args.project) == v6.DEFAULT_PROJECT:
        args.project = DEFAULT_PROJECT
    if Path(args.report_dir) == v6.DEFAULT_REPORT:
        args.report_dir = DEFAULT_REPORT
    if args.name == v6.DEFAULT_NAME:
        args.name = DEFAULT_NAME
    if not _specified("--epochs"):
        args.epochs = 20
    return args


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _registered_protocol(args) -> dict[str, Any]:
    if not REGISTERED_PROTOCOL.is_file():
        raise FileNotFoundError(
            "freeze the distilled development protocol before training"
        )
    registered = json.loads(
        REGISTERED_PROTOCOL.read_text(encoding="utf-8-sig")
    )
    expected = registered.get("training", {})
    actual = {
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "workers": int(args.workers),
        "freeze_layers": int(args.freeze_layers),
        "amp": not args.no_amp,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
    }
    if (
        registered.get("protocol") != REGISTERED_PROTOCOL_NAME
        or registered.get("status")
        != "development_protocol_frozen_before_training"
        or registered.get("formal_paper_metric") is not False
        or expected != actual
        or registered.get("bindings", {})
        .get("scene_observations", {})
        .get("sha256")
        != sha256_file(Path(args.sensor_observations))
    ):
        raise ValueError("distilled development protocol or runtime drifted")
    return registered


def _protocol(args, paths, train_args, degradation):
    registered = _registered_protocol(args)
    protocol = BASE_PROTOCOL_BUILDER(
        args, paths, train_args, degradation
    )
    protocol["protocol"] = RUN_PROTOCOL_NAME
    protocol["status"] = "development_protocol_frozen_before_training"
    protocol["formal_paper_metric"] = False
    protocol["registered_protocol"] = _binding(REGISTERED_PROTOCOL)
    protocol["sensor_source"].update(
        {
            "mode": "object-qwen-scene-distilled",
            "model": "SceneQwenDistiller",
            "direct_full_image_qwen_api": False,
            "teacher_source": "object-crop Qwen sidecars",
            "teacher_uses_annotations": True,
            "joint_inference_uses_teacher": False,
            "joint_inference_uses_annotations": False,
            "calibration_fit_split": "val",
        }
    )
    protocol["leakage_control"].update(
        {
            "teacher_annotation_use_is_offline_only": True,
            "student_sidecar_generation_reads_annotations": False,
            "joint_train_and_metric_forward_read_teacher": False,
        }
    )
    protocol["claim_boundaries"] = registered["claim_boundaries"]
    for path in (
        Path(__file__),
        REGISTERED_PROTOCOL,
        PROJECT_ROOT / "mmfc" / "scene_distillation.py",
        PROJECT_ROOT / "predict_scene_qwen_distiller.py",
    ):
        protocol["source_bindings"][path.resolve().as_posix()] = _binding(path)
    return protocol


def run(args) -> dict[str, Any]:
    original_protocol = v6._protocol
    v6._protocol = _protocol
    try:
        result = v6.run(args)
    finally:
        v6._protocol = original_protocol

    if result.get("state") == "complete":
        evaluation_path = Path(result["evaluation"])
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8-sig")
        )
        evaluation["evidence_role"] = (
            "object_qwen_distilled_patent_stack_development"
        )
        evaluation["registered_protocol"] = _binding(REGISTERED_PROTOCOL)
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
