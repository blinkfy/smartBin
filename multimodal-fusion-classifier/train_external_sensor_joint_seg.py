"""Train joint Segment head v2 with an external full-image 4x4 observation."""

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
from train_yolo_seg import (
    _json_safe,
    _read_training_summary,
    _runtime_environment,
)
from validate_yolo_seg_export import validate_export


DEFAULT_DATA = PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "dataset.yaml"
DEFAULT_BASE_WEIGHTS = (
    PROJECT_ROOT
    / "checkpoints"
    / "yolo-seg"
    / "yolo11s-seg-qwen-split"
    / "weights"
    / "best.pt"
)
DEFAULT_OBSERVATIONS = (
    PROJECT_ROOT / "data" / "scene-sensor-sidecars" / "observations.jsonl"
)
DEFAULT_TEXT_PROTOTYPES = (
    PROJECT_ROOT / "data" / "joint-multimodal-seg" / "text_prototypes.npy"
)
DEFAULT_PROJECT = (
    PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg-external"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train external-scene-sensor joint YOLO segmentation"
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
    parser.add_argument(
        "--name", default="yolo11s-seg-joint-external-rule-smoke"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
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
        "--cache", choices=("false", "disk", "ram"), default="false"
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-rgb-fallback", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _cache_value(value: str) -> bool | str:
    return False if value == "false" else value


def _validate_args(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "data": resolve_project_path(args.data),
        "base_weights": resolve_project_path(args.base_weights),
        "sensor_observations": resolve_project_path(
            args.sensor_observations
        ),
        "text_prototypes": resolve_project_path(args.text_prototypes),
        "project": resolve_project_path(args.project),
        "report_dir": resolve_project_path(args.report_dir),
    }
    for key in (
        "data",
        "base_weights",
        "sensor_observations",
        "text_prototypes",
    ):
        if not paths[key].is_file():
            raise FileNotFoundError(f"missing {key}: {paths[key]}")
    validation_path = (
        paths["sensor_observations"].parent / "VALIDATION_REPORT.json"
    )
    build_path = paths["sensor_observations"].parent / "BUILD_REPORT.json"
    if not validation_path.is_file() or not build_path.is_file():
        raise FileNotFoundError("scene sensor build/validation reports are missing")
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    build = json.loads(build_path.read_text(encoding="utf-8-sig"))
    if (
        validation.get("valid") is not True
        or validation.get("source_image_count") != 5990
        or build.get("valid") is not True
        or build.get("source_contract", {}).get("full_image_only") is not True
        or build.get("source_contract", {}).get("annotation_files_read")
        is not False
        or build.get("output", {}).get("sha256")
        != sha256_file(paths["sensor_observations"])
    ):
        raise ValueError("scene sensor evidence is not complete and valid")
    paths["sensor_validation"] = validation_path
    paths["sensor_build"] = build_path
    if args.epochs <= 0 or args.patience < 0:
        raise ValueError("epochs must be positive and patience non-negative")
    if (
        args.batch_size <= 0
        or args.image_size <= 0
        or args.workers < 0
        or args.freeze_layers < 0
    ):
        raise ValueError("invalid batch, image-size, workers, or freeze value")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer parameters")
    if (
        args.semantic_dim <= 0
        or args.sensor_hidden_dim <= 0
        or args.text_temperature <= 0.0
        or args.sensor_reconstruction_weight < 0.0
        or args.sensor_calibration_weight < 0.0
    ):
        raise ValueError("invalid fusion parameters")
    if not args.name.strip() or Path(args.name).name != args.name:
        raise ValueError("name must be one non-empty path component")
    export = validate_export(paths["data"].parent)
    if not export["valid"] or export["link_mode"] != "symlink":
        raise ValueError("external joint training requires valid symlink export")
    run_dir = paths["project"] / args.name
    report_run = paths["report_dir"] / args.name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite training run: {run_dir}")
    if report_run.exists() and any(report_run.iterdir()):
        raise FileExistsError(f"refusing to overwrite report run: {report_run}")
    return paths


def _protocol(
    args: argparse.Namespace,
    paths: dict[str, Path],
    train_args: dict[str, Any],
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
        PROJECT_ROOT / "mmfc" / "joint_seg_external.py",
        PROJECT_ROOT / "mmfc" / "joint_seg.py",
    )
    return {
        "schema_version": 1,
        "protocol": "joint_text_external_scene_sensor_yolo_segment_v2",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_training",
        "run_name": args.name,
        "seed": int(args.seed),
        "architecture": {
            "base": "independently trained RGB YOLO11s-seg checkpoint",
            "image_backbone_inputs": ["RGB"],
            "joint_head_inputs": [
                "P3",
                "P4",
                "P5",
                "CLIP class prototypes",
                "external full-image 4x4 soft sensor",
            ],
            "sensor_transport": (
                "quantized fourth loader channel split before the RGB backbone"
            ),
            "injection_point": "P3/P4/P5 immediately before native Segment head",
            "missing_sensor_fallback": "RGB-predicted 4x4 soft sensor",
            "head_outputs": ["boxes", "classes", "instance_masks"],
        },
        "sensor_source": {
            "mode": build.get("mode"),
            "model": build.get("model"),
            "source_image_count": build.get("source_image_count"),
            "full_image_only": True,
            "annotation_files_read": False,
            "label_files_read": False,
            "bbox_inputs": False,
            "polygon_inputs": False,
            "class_inputs": False,
        },
        "leakage_control": {
            "metric_forward_inputs": [
                "RGB",
                "full_image_4x4_soft_sensor",
            ],
            "ground_truth_boxes_consumed_by_sensor": False,
            "ground_truth_masks_consumed_by_sensor": False,
            "ground_truth_classes_consumed_by_sensor": False,
            "annotation_conditioned_spatial_projection": False,
        },
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
        },
        "claim_boundaries": [
            "The current full-image sidecar is a software sensor observation.",
            "Qwen mode remains image-conditioned rather than physical measurement.",
            "Rule mode is an integration control and not the final Qwen experiment.",
            "No RGB-D input or synchronized hardware sensor is introduced.",
            "Metrics from this v2 task must be reported separately from v1 RGB-only deployment.",
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
    paths = _validate_args(args)
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
    protocol = _protocol(args, paths, train_args)
    protocol_path = report_run / "PROTOCOL.json"
    _atomic_write_text(
        protocol_path,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    if args.validate_only:
        return {
            "state": "validated",
            "protocol": protocol_path.as_posix(),
        }

    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils import RANK

    from mmfc.joint_seg import (
        load_text_prototypes,
        parameter_summary,
    )
    from mmfc.joint_seg_external import (
        ExternalSensorJointSegmentationModel,
        ExternalSensorSegmentationValidator,
        build_external_sensor_dataset,
        load_scene_sensor_index,
    )

    class ExternalSensorTrainer(SegmentationTrainer):
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
            return build_external_sensor_dataset(
                trainer=self,
                img_path=img_path,
                mode=mode,
                batch=batch,
                sensor_observations=self.sensor_observation_index,
            )

        def get_model(
            self,
            cfg: dict[str, Any] | str | None = None,
            weights=None,
            verbose: bool = True,
        ):
            model = ExternalSensorJointSegmentationModel(
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
                fusion_mode="full",
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
            return ExternalSensorSegmentationValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
                sensor_observations=paths["sensor_observations"],
            )

    trainer = ExternalSensorTrainer(overrides=train_args)
    trainer.train()
    run_dir = paths["project"] / args.name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("external joint training checkpoints are missing")

    validators = []

    def validator_factory(*factory_args: Any, **factory_kwargs: Any):
        validator = ExternalSensorSegmentationValidator(
            *factory_args,
            **factory_kwargs,
            sensor_observations=paths["sensor_observations"],
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
        name="test-external-sensor",
        exist_ok=False,
        plots=True,
        save_json=False,
        verbose=True,
    )
    validator = validators[0]
    if validator.external_sensor_sample_count != 570:
        raise ValueError("external sensor test did not consume all 570 images")
    trained_model = best_model.model
    external_used = trained_model.joint_fusion.last_aux[
        "external_sensor_used"
    ]
    if not bool((external_used == 1).all()):
        raise ValueError("external sensor was not used in the final metric batch")

    fallback = None
    if not args.skip_rgb_fallback:
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
        if not bool((fallback_aux["external_sensor_used"] == 0).all()):
            raise ValueError("RGB fallback unexpectedly used an external sensor")
        fallback = {
            "metrics": _json_safe(fallback_metrics.results_dict),
            "speed_ms_per_image": _json_safe(fallback_metrics.speed),
        }

    evaluation = {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "valid": True,
        "external_sensor": {
            "metrics": _json_safe(external_metrics.results_dict),
            "per_class": _json_safe(
                external_metrics.summary(decimals=8)
            ),
            "speed_ms_per_image": _json_safe(external_metrics.speed),
            "metric_source_image_count": (
                validator.external_sensor_sample_count
            ),
            "metric_batch_count": validator.external_sensor_batch_count,
            "forward_inputs": [
                "RGB",
                "full_image_4x4_soft_sensor",
            ],
        },
        "rgb_fallback": fallback,
        "parameters": parameter_summary(trained_model),
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
        "leakage_control": protocol["leakage_control"],
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
        "checkpoint": evaluation["checkpoint"],
        "external_sensor_metrics": evaluation["external_sensor"]["metrics"],
        "rgb_fallback_metrics": (
            fallback["metrics"] if fallback is not None else None
        ),
        "evaluation": evaluation_path.as_posix(),
    }


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
