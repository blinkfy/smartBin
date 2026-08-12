"""Train an independent YOLO11 Segment head with text and soft-sensor fusion."""

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
from train_yolo_seg import _json_safe, _runtime_environment, _read_training_summary
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
DEFAULT_PREPARED = PROJECT_ROOT / "data" / "joint-multimodal-seg"
DEFAULT_PROJECT = PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg"
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "joint-multimodal-seg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train joint text/soft-sensor YOLO11 segmentation."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument(
        "--sensor-targets",
        type=Path,
        default=DEFAULT_PREPARED / "image_sensor_targets.jsonl",
    )
    parser.add_argument(
        "--text-prototypes",
        type=Path,
        default=DEFAULT_PREPARED / "text_prototypes.npy",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--name", default="yolo11s-seg-joint-v1-seed42")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=30)
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
        "--fusion-mode",
        choices=("full", "text_only", "sensor_only"),
        default="full",
    )
    parser.add_argument(
        "--cache",
        choices=("false", "disk", "ram"),
        default="false",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_args(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "data": resolve_project_path(args.data),
        "base_weights": resolve_project_path(args.base_weights),
        "sensor_targets": resolve_project_path(args.sensor_targets),
        "text_prototypes": resolve_project_path(args.text_prototypes),
        "project": resolve_project_path(args.project),
        "report_dir": resolve_project_path(args.report_dir),
    }
    for key in ("data", "base_weights", "sensor_targets", "text_prototypes"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"missing {key}: {paths[key]}")
    if args.epochs <= 0 or args.patience < 0:
        raise ValueError("epochs must be positive and patience non-negative")
    if (
        args.batch_size <= 0
        or args.image_size <= 0
        or args.workers < 0
        or args.freeze_layers < 0
    ):
        raise ValueError("invalid batch, image-size, or workers value")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer parameters")
    if (
        args.semantic_dim <= 0
        or args.sensor_hidden_dim <= 0
        or args.text_temperature <= 0
    ):
        raise ValueError("invalid fusion dimensions or temperature")
    if (
        args.sensor_reconstruction_weight < 0
        or args.sensor_calibration_weight < 0
    ):
        raise ValueError("sensor loss weights must be non-negative")
    if not args.name.strip() or Path(args.name).name != args.name:
        raise ValueError("name must be one non-empty path component")

    export = validate_export(paths["data"].parent)
    if not export["valid"] or export["link_mode"] != "symlink":
        raise ValueError("joint training requires the valid symlink-only YOLO export")
    run_dir = paths["project"] / args.name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite training run: {run_dir}")
    report_run = paths["report_dir"] / args.name
    if report_run.exists() and any(report_run.iterdir()):
        raise FileExistsError(f"refusing to overwrite report run: {report_run}")
    return paths


def _protocol(
    args: argparse.Namespace,
    paths: dict[str, Path],
    train_args: dict[str, Any],
) -> dict[str, Any]:
    preparation_report = paths["sensor_targets"].parent / "PREPARATION_REPORT.json"
    bindings = {}
    for path in (
        paths["data"],
        paths["base_weights"],
        paths["sensor_targets"],
        paths["text_prototypes"],
        preparation_report,
        Path(__file__).resolve(),
        PROJECT_ROOT / "mmfc" / "joint_seg.py",
    ):
        if path.is_file():
            bindings[path.resolve().as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": 1,
        "protocol": "joint_text_predicted_soft_sensor_yolo_segment_v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_training",
        "run_name": args.name,
        "seed": int(args.seed),
        "architecture": {
            "base": "independently trained YOLO11s-seg checkpoint",
            "injection_point": "P3/P4/P5 immediately before native Segment head",
            "fusion_mode": args.fusion_mode,
            "text_enabled": args.fusion_mode != "sensor_only",
            "sensor_enabled": args.fusion_mode != "text_only",
            "text": (
                "frozen CLIP class prototypes with spatial semantic attention"
                if args.fusion_mode != "sensor_only"
                else "disabled for independently retrained ablation"
            ),
            "sensor": (
                "RGB-predicted 4x4 soft-sensor grid with uncertainty reliability"
                if args.fusion_mode != "text_only"
                else "disabled for independently retrained ablation"
            ),
            "head_outputs": ["boxes", "classes", "instance_masks"],
        },
        "leakage_control": {
            "target_use": "auxiliary sensor-distillation loss only",
            "metric_forward_inputs": ["RGB"],
            "ground_truth_sensor_target_consumed_by_head": False,
            "ground_truth_boxes_consumed_by_sensor_predictor": False,
            "external_sensor_default": False,
        },
        "train_args": train_args,
        "fusion_args": {
            "fusion_mode": args.fusion_mode,
            "semantic_dim": int(args.semantic_dim),
            "sensor_hidden_dim": int(args.sensor_hidden_dim),
            "text_temperature": float(args.text_temperature),
            "sensor_reconstruction_weight": float(
                args.sensor_reconstruction_weight
            ),
            "sensor_calibration_weight": float(args.sensor_calibration_weight),
        },
        "performance_policy": {
            "workers": int(args.workers),
            "batch_size": int(args.batch_size),
            "freeze_layers": int(args.freeze_layers),
            "amp": not args.no_amp,
            "cache": args.cache,
            "workers4_long_run_withdrawn_after_prior_stall": True,
            "optimizer_nominal_batch_size": 64,
        },
        "claim_boundaries": [
            "No RGB-D input is introduced.",
            "No synchronized hardware sensor measurement is introduced.",
            "Qwen/rule targets are image-conditioned soft-sensor distillation targets.",
            "The metric forward path is deployable without annotations or sidecars.",
        ],
        "source_bindings": bindings,
    }


def _cache_value(value: str) -> bool | str:
    return False if value == "false" else value


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
        # Spatial transforms would invalidate the fixed full-image 4x4 target.
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
            "run_dir": (paths["project"] / args.name).as_posix(),
        }

    from ultralytics import YOLO
    from ultralytics.models import yolo
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.utils import RANK, colorstr
    from ultralytics.utils.torch_utils import unwrap_model

    from mmfc.joint_seg import (
        JointMultimodalSegmentationModel,
        SoftSensorTargetDataset,
        load_sensor_target_index,
        load_text_prototypes,
        parameter_summary,
    )

    class JointTrainer(SegmentationTrainer):
        def __init__(self, *trainer_args: Any, **trainer_kwargs: Any) -> None:
            self.sensor_target_index = load_sensor_target_index(
                paths["sensor_targets"]
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
            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            pad = 0.0 if mode == "train" else 0.5
            fraction = self.args.fraction if mode == "train" else 1.0
            return SoftSensorTargetDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect or mode == "val",
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=stride,
                pad=pad,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=fraction,
                sensor_targets=self.sensor_target_index,
            )

        def get_model(
            self,
            cfg: dict[str, Any] | str | None = None,
            weights=None,
            verbose: bool = True,
        ):
            model = JointMultimodalSegmentationModel(
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
                sensor_calibration_weight=float(args.sensor_calibration_weight),
                fusion_mode=args.fusion_mode,
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
            return yolo.segment.SegmentationValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

    trainer = JointTrainer(overrides=train_args)
    trainer.train()
    run_dir = paths["project"] / args.name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("joint training did not create best.pt and last.pt")

    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(paths["data"]),
        split="test",
        imgsz=int(args.image_size),
        batch=int(args.batch_size),
        device=str(args.device),
        workers=int(args.workers),
        project=str(report_run),
        name="test",
        exist_ok=False,
        plots=True,
        save_json=False,
        verbose=True,
    )
    trained_model = best_model.model
    evaluation = {
        "schema_version": 1,
        "protocol": protocol["protocol"],
        "metrics": _json_safe(metrics.results_dict),
        "per_class": _json_safe(metrics.summary(decimals=8)),
        "speed_ms_per_image": _json_safe(metrics.speed),
        "parameters": parameter_summary(trained_model),
        "fusion_mode": args.fusion_mode,
        "training": _read_training_summary(run_dir / "results.csv"),
        "checkpoint": {
            "path": best_path.resolve().as_posix(),
            "size_bytes": best_path.stat().st_size,
            "sha256": sha256_file(best_path),
        },
        "runtime": _runtime_environment(),
        "forward_contract": protocol["leakage_control"],
        "limitations": protocol["claim_boundaries"],
    }
    evaluation_path = report_run / "evaluation.json"
    _atomic_write_text(
        evaluation_path,
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
    )
    return {
        "state": "complete",
        "checkpoint": evaluation["checkpoint"],
        "metrics": evaluation["metrics"],
        "evaluation": evaluation_path.as_posix(),
    }


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
