"""Train the reliability-aware external-sensor joint Segment head v3."""

from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path
from prepare_joint_multimodal_seg import sha256_file
from train_external_sensor_joint_seg import (
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_DATA,
    DEFAULT_OBSERVATIONS,
    DEFAULT_TEXT_PROTOTYPES,
    _atomic_write_text,
    _cache_value,
    _validate_args,
)
from train_yolo_seg import (
    _json_safe,
    _read_training_summary,
    _runtime_environment,
)


DEFAULT_PROJECT = (
    PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg-reliable"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-reliable"
)
DEFAULT_NAME = "yolo11s-seg-joint-reliable-rule-dev-seed42-retry4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train reliability-aware external-sensor YOLO segmentation"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument(
        "--sensor-observations", type=Path, default=DEFAULT_OBSERVATIONS
    )
    parser.add_argument(
        "--text-prototypes", type=Path, default=DEFAULT_TEXT_PROTOTYPES
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze-layers", type=int, default=24)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--sensor-hidden-dim", type=int, default=256)
    parser.add_argument("--text-temperature", type=float, default=0.10)
    parser.add_argument("--sensor-reconstruction-weight", type=float, default=0.25)
    parser.add_argument("--sensor-calibration-weight", type=float, default=0.05)
    parser.add_argument(
        "--reliability-calibration-weight", type=float, default=0.30
    )
    parser.add_argument("--reliability-hidden-dim", type=int, default=32)
    parser.add_argument("--reliability-init", type=float, default=0.85)
    parser.add_argument("--missing-probability", type=float, default=0.10)
    parser.add_argument("--noise-probability", type=float, default=0.50)
    parser.add_argument("--noise-std-low", type=float, default=0.02)
    parser.add_argument("--noise-std-high", type=float, default=0.15)
    parser.add_argument("--bias-probability", type=float, default=0.25)
    parser.add_argument("--bias-strength", type=float, default=0.15)
    parser.add_argument("--stuck-probability", type=float, default=0.15)
    parser.add_argument("--quality-scale", type=float, default=0.25)
    parser.add_argument(
        "--cache", choices=("false", "disk", "ram"), default="false"
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _degradation_config(args: argparse.Namespace) -> dict[str, Any]:
    probabilities = (
        args.missing_probability,
        args.noise_probability,
        args.bias_probability,
        args.stuck_probability,
    )
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("sensor degradation probabilities must be in [0,1]")
    if (
        args.noise_std_low < 0.0
        or args.noise_std_high < args.noise_std_low
        or args.bias_strength < 0.0
        or args.quality_scale <= 0.0
    ):
        raise ValueError("invalid sensor degradation strengths")
    return {
        "enabled": True,
        "missing_probability": float(args.missing_probability),
        "noise_probability": float(args.noise_probability),
        "noise_std_range": [
            float(args.noise_std_low),
            float(args.noise_std_high),
        ],
        "bias_probability": float(args.bias_probability),
        "bias_strength": float(args.bias_strength),
        "stuck_probability": float(args.stuck_probability),
        "quality_scale": float(args.quality_scale),
    }


def _protocol(
    args: argparse.Namespace,
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
        PROJECT_ROOT / "mmfc" / "joint_seg_reliable.py",
        PROJECT_ROOT / "mmfc" / "joint_seg_external.py",
        PROJECT_ROOT / "mmfc" / "joint_seg.py",
        PROJECT_ROOT / "mmfc" / "model.py",
        PROJECT_ROOT / "mmfc" / "training.py",
    )
    return {
        "schema_version": 1,
        "protocol": (
            "joint_text_external_scene_sensor_reliability_yolo_segment_v3"
        ),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_protocol_frozen_before_training",
        "formal_paper_metric": False,
        "run_name": args.name,
        "seed": int(args.seed),
        "architecture": {
            "image_backbone_inputs": ["RGB"],
            "transport_channels": [
                "RGB",
                "4x4 sensor raster",
                "sensor availability",
            ],
            "joint_head_inputs": [
                "P3",
                "P4",
                "P5",
                "CLIP class prototypes",
                "reliability-blended external 4x4 sensor",
            ],
            "reliability_features": [
                "uncertainty-weighted discrepancy",
                "cosine consistency",
                "predicted uncertainty",
                "availability",
                "raw and cross-modal grid-quality statistics",
            ],
            "missing_sensor_fallback": "RGB-predicted 4x4 soft sensor",
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
            "degradation_uses_labels": False,
            "degradation_seed_inputs": [
                "run seed",
                "source image id",
            ],
        },
        "claim_boundaries": [
            "This rule-sidecar run is mechanism development, not a formal Qwen metric.",
            "Software degradation does not establish hardware robustness.",
            "No RGB-D input or synchronized physical sensor is present.",
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.reliability_calibration_weight < 0.0
        or args.reliability_hidden_dim <= 0
        or not 0.0 < args.reliability_init < 1.0
    ):
        raise ValueError("invalid reliability gate parameters")
    paths = _validate_args(args)
    degradation = _degradation_config(args)
    train_args: dict[str, Any] = {
        "data": str(paths["data"]),
        "model": str(paths["base_weights"]),
        "project": str(paths["project"]),
        "name": args.name,
        "exist_ok": False,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch": int(args.batch_size),
        "imgsz": int(args.image_size),
        "workers": int(args.workers),
        "device": str(args.device),
        "seed": int(args.seed),
        "deterministic": True,
        "amp": not args.no_amp,
        "optimizer": "AdamW",
        "lr0": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "cos_lr": True,
        "nbs": 64,
        "cache": _cache_value(args.cache),
        "pretrained": True,
        "plots": True,
        "save": True,
        "verbose": True,
        "compile": False,
        "freeze": int(args.freeze_layers),
        "warmup_epochs": 1.0,
        "warmup_bias_lr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "erasing": 0.0,
        "close_mosaic": 0,
    }
    report_run = paths["report_dir"] / args.name
    protocol = _protocol(args, paths, train_args, degradation)
    protocol_path = report_run / "PROTOCOL.json"
    _atomic_write_text(
        protocol_path,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    if args.validate_only:
        return {
            "state": "validated",
            "valid": True,
            "protocol": protocol_path.as_posix(),
        }

    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils import RANK

    from mmfc.joint_seg import load_text_prototypes, parameter_summary
    from mmfc.joint_seg_external import load_scene_sensor_index
    from mmfc.joint_seg_reliable import (
        ReliableExternalSensorJointSegmentationModel,
        ReliableExternalSensorSegmentationValidator,
        build_reliable_external_sensor_dataset,
    )

    class ReliableExternalSensorTrainer(SegmentationTrainer):
        def __init__(self, *trainer_args: Any, **trainer_kwargs: Any) -> None:
            self.sensor_observation_index = load_scene_sensor_index(
                paths["sensor_observations"]
            )
            self.text_prototypes = load_text_prototypes(
                paths["text_prototypes"], num_classes=19
            )
            super().__init__(*trainer_args, **trainer_kwargs)

        def build_dataset(
            self,
            img_path: str,
            mode: str = "train",
            batch: int | None = None,
        ):
            return build_reliable_external_sensor_dataset(
                trainer=self,
                img_path=img_path,
                mode=mode,
                batch=batch,
                sensor_observations=self.sensor_observation_index,
                degradation=degradation,
                degradation_seed=int(args.seed),
            )

        def get_model(
            self,
            cfg: dict[str, Any] | str | None = None,
            weights=None,
            verbose: bool = True,
        ):
            model = ReliableExternalSensorJointSegmentationModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                text_prototypes=self.text_prototypes,
                semantic_dim=int(args.semantic_dim),
                sensor_hidden_dim=int(args.sensor_hidden_dim),
                text_temperature=float(args.text_temperature),
                sensor_reconstruction_weight=float(
                    args.sensor_reconstruction_weight
                ),
                sensor_calibration_weight=float(
                    args.sensor_calibration_weight
                ),
                reliability_calibration_weight=float(
                    args.reliability_calibration_weight
                ),
                reliability_hidden_dim=int(args.reliability_hidden_dim),
                reliability_init=float(args.reliability_init),
            )
            if weights:
                model.load(weights)
            return model

        def get_validator(self):
            self.loss_names = (
                "box_loss",
                "seg_loss",
                "cls_loss",
                "dfl_loss",
                "sem_loss",
                "sensor_loss",
            )
            return ReliableExternalSensorSegmentationValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
                sensor_observations=self.sensor_observation_index,
                degradation={"enabled": False},
                degradation_seed=int(args.seed),
            )

    trainer = ReliableExternalSensorTrainer(overrides=train_args)
    trainer.train()
    run_dir = paths["project"] / args.name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("reliable joint training checkpoints are missing")

    observation_index = load_scene_sensor_index(paths["sensor_observations"])
    validators = []

    def validator_factory(*factory_args: Any, **factory_kwargs: Any):
        validator = ReliableExternalSensorSegmentationValidator(
            *factory_args,
            **factory_kwargs,
            sensor_observations=observation_index,
            degradation={"enabled": False},
            degradation_seed=int(args.seed),
        )
        validators.append(validator)
        return validator

    best_model = YOLO(str(best_path))
    external_metrics = best_model.val(
        validator=validator_factory,
        data=str(paths["data"]),
        split="test",
        imgsz=int(args.image_size),
        batch=int(args.batch_size),
        device=str(args.device),
        workers=int(args.workers),
        project=str(report_run),
        name="test-external-clean",
        exist_ok=False,
        plots=True,
        save_json=False,
        verbose=True,
    )
    validator = validators[0]
    if validator.external_sensor_sample_count != 570:
        raise ValueError("reliable external test did not consume 570 images")
    external_aux = best_model.model.joint_fusion.last_aux
    if not bool((external_aux["sensor_available"] == 1).all()):
        raise ValueError("clean external sensor was not available")

    fallback_model = YOLO(str(best_path))
    fallback_metrics = fallback_model.val(
        data=str(paths["data"]),
        split="test",
        imgsz=int(args.image_size),
        batch=int(args.batch_size),
        device=str(args.device),
        workers=int(args.workers),
        project=str(report_run),
        name="test-rgb-fallback",
        exist_ok=False,
        plots=False,
        save_json=False,
        verbose=True,
    )
    fallback_aux = fallback_model.model.joint_fusion.last_aux
    if not bool((fallback_aux["sensor_available"] == 0).all()):
        raise ValueError("RGB fallback unexpectedly marked sensor available")

    evaluation = {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "valid": True,
        "formal_paper_metric": False,
        "evidence_role": "rule_sidecar_reliability_development",
        "external_clean": {
            "metrics": _json_safe(external_metrics.results_dict),
            "speed_ms_per_image": _json_safe(external_metrics.speed),
            "metric_source_image_count": validator.external_sensor_sample_count,
        },
        "rgb_fallback": {
            "metrics": _json_safe(fallback_metrics.results_dict),
            "speed_ms_per_image": _json_safe(fallback_metrics.speed),
        },
        "last_batch_gate": {
            "clean_reliability_mean": float(
                external_aux["sensor_reliability"].float().mean().item()
            ),
            "clean_consistency_mean": float(
                external_aux["sensor_consistency"].float().mean().item()
            ),
            "fallback_reliability_mean": float(
                fallback_aux["sensor_reliability"].float().mean().item()
            ),
        },
        "parameters": parameter_summary(best_model.model),
        "training": _read_training_summary(run_dir / "results.csv"),
        "checkpoint": {
            "path": best_path.resolve().as_posix(),
            "size_bytes": best_path.stat().st_size,
            "sha256": sha256_file(best_path),
        },
        "sensor_observations": {
            "path": paths["sensor_observations"].resolve().as_posix(),
            "size_bytes": paths["sensor_observations"].stat().st_size,
            "sha256": sha256_file(paths["sensor_observations"]),
            "mode": protocol["sensor_source"]["mode"],
        },
        "runtime": _runtime_environment(),
        "limitations": protocol["claim_boundaries"],
    }
    evaluation_path = report_run / "evaluation.json"
    _atomic_write_text(
        evaluation_path,
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "checkpoint": evaluation["checkpoint"],
        "external_clean_metrics": evaluation["external_clean"]["metrics"],
        "rgb_fallback_metrics": evaluation["rgb_fallback"]["metrics"],
        "evaluation": evaluation_path.as_posix(),
    }


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
