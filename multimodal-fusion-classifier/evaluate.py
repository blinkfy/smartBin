"""Independent train/validation/test split evaluation CLI."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.evaluation import (
    evaluate_model,
    grouped_bootstrap_classification_intervals,
    write_evaluation_artifacts,
)
from mmfc.model import (
    SENSOR_RELIABILITY_EVIDENCE,
    VALID_MODALITIES,
    build_production_model,
)
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用训练检查点独立评估智能垃圾分类数据集"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.json",
        help="UTF-8 JSON 配置文件",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "qwen-material" / "best.pt",
        help="默认使用按验证准确率选择的 Qwen best.pt",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="默认只评估独立测试集",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="评估产物目录；默认 reports/qwen-material/<split>",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument(
        "--disable-sensor-reliability",
        action="store_true",
        help=(
            "Post-hoc intervention: load the full checkpoint, then bypass "
            "sensor reliability gating during inference"
        ),
    )
    parser.add_argument(
        "--posthoc-modalities",
        choices=("visual", "visual_text", "visual_sensor", "trimodal"),
        default=None,
        help="Inference-only modality-path intervention on a loaded checkpoint",
    )
    parser.add_argument(
        "--disable-sensor-grid-quality",
        action="store_true",
        help="Inference-only removal of the raw-grid quality logit branch",
    )
    parser.add_argument(
        "--disable-sensor-cross-modal-quality",
        action="store_true",
        help="Inference-only removal of predicted-observed grid comparison",
    )
    parser.add_argument(
        "--disable-reliability-evidence",
        nargs="+",
        choices=SENSOR_RELIABILITY_EVIDENCE,
        default=(),
        help="Inference-only masking of selected reliability-head inputs",
    )
    parser.add_argument(
        "--disable-film",
        action="store_true",
        help="Inference-only removal of learned FiLM scaling and bias",
    )
    parser.add_argument(
        "--disable-attention",
        action="store_true",
        help="Inference-only replacement of attention by equal-mean fusion",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅用于冒烟测试；正式论文评估不要设置",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=0,
        help="按 source_image_id 聚类重采样次数；0 表示关闭",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260727,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "num_workers", "max_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0 and name != "num_workers":
            raise ValueError(f"--{name.replace('_', '-')} 必须大于 0")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0")
    if args.ece_bins < 2:
        raise ValueError("--ece-bins 必须大于等于 2")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches 不能小于 0")
    if args.bootstrap_repeats < 0 or args.bootstrap_repeats == 1:
        raise ValueError("--bootstrap-repeats 必须为 0 或至少 2")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence 必须位于 (0,1)")


def _checkpoint_metadata(
    checkpoint: dict[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    return {
        "path": str(checkpoint_path.resolve()),
        "size_bytes": checkpoint_path.stat().st_size,
        "epoch": checkpoint.get("epoch"),
        "selection_val_accuracy": checkpoint.get("val_accuracy"),
    }


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in (
        "torch",
        "torchvision",
        "ultralytics",
        "transformers",
        "scikit-learn",
        "matplotlib",
        "opencv-python",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
    }


def apply_evaluation_interventions(
    model: torch.nn.Module,
    *,
    disable_sensor_reliability: bool = False,
    posthoc_modalities: str | None = None,
    disable_sensor_grid_quality: bool = False,
    disable_sensor_cross_modal_quality: bool = False,
    disabled_reliability_evidence: tuple[str, ...] = (),
    disable_film: bool = False,
    disable_attention: bool = False,
) -> dict[str, Any]:
    """Apply explicit inference-only interventions after checkpoint loading."""

    modality_sets = {
        "visual": ("visual",),
        "visual_text": ("visual", "text"),
        "visual_sensor": ("visual", "sensor"),
        "trimodal": VALID_MODALITIES,
    }
    configured_modalities = tuple(getattr(model, "enabled_modalities", ()))
    configured_reliability = bool(
        getattr(model, "sensor_reliability_enabled", False)
    )
    configured_grid_quality = bool(
        getattr(model, "sensor_grid_quality_enabled", False)
    )
    configured_cross_modal_quality = bool(
        getattr(model, "sensor_cross_modal_quality_enabled", False)
    )
    configured_film = bool(getattr(model, "film_enabled", False))
    configured_fusion_mode = getattr(model, "fusion_mode", None)

    if posthoc_modalities is not None:
        if (
            not hasattr(model, "enabled_modalities")
            or not hasattr(model, "modality_mask")
        ):
            raise ValueError("model does not expose runtime modality controls")
        active_modalities = modality_sets[posthoc_modalities]
        setattr(model, "enabled_modalities", active_modalities)
        mask = torch.tensor(
            [name in active_modalities for name in VALID_MODALITIES],
            device=model.modality_mask.device,
            dtype=model.modality_mask.dtype,
        )
        model.modality_mask.copy_(mask)
        if not {"text", "sensor"}.issubset(active_modalities):
            if hasattr(model, "sensor_reliability_enabled"):
                setattr(model, "sensor_reliability_enabled", False)
            if hasattr(model, "sensor_grid_quality_enabled"):
                setattr(model, "sensor_grid_quality_enabled", False)
            if hasattr(model, "sensor_cross_modal_quality_enabled"):
                setattr(model, "sensor_cross_modal_quality_enabled", False)

    if not hasattr(model, "sensor_reliability_enabled"):
        if disable_sensor_reliability:
            raise ValueError(
                "model does not expose sensor_reliability_enabled"
            )
        if any(
            (
                disable_sensor_grid_quality,
                disable_sensor_cross_modal_quality,
                bool(disabled_reliability_evidence),
                disable_film,
                disable_attention,
                posthoc_modalities is not None,
            )
        ):
            raise ValueError("model does not expose requested intervention controls")
        return {
            "sensor_reliability_configured": False,
            "sensor_reliability_disabled": False,
            "sensor_reliability_active": False,
        }
    if disable_sensor_reliability:
        setattr(model, "sensor_reliability_enabled", False)

    if disable_sensor_grid_quality:
        if not hasattr(model, "sensor_grid_quality_enabled"):
            raise ValueError("model does not expose sensor_grid_quality_enabled")
        setattr(model, "sensor_grid_quality_enabled", False)
    if disable_sensor_cross_modal_quality:
        if not hasattr(model, "sensor_cross_modal_quality_enabled"):
            raise ValueError(
                "model does not expose sensor_cross_modal_quality_enabled"
            )
        setattr(model, "sensor_cross_modal_quality_enabled", False)
    evidence = tuple(dict.fromkeys(disabled_reliability_evidence))
    if evidence:
        setter = getattr(
            model,
            "set_disabled_sensor_reliability_evidence",
            None,
        )
        if setter is None:
            raise ValueError("model does not expose reliability evidence controls")
        setter(evidence)
    if disable_film:
        if not hasattr(model, "film_enabled"):
            raise ValueError("model does not expose film_enabled")
        setattr(model, "film_enabled", False)
    if disable_attention:
        if not hasattr(model, "fusion_mode"):
            raise ValueError("model does not expose fusion_mode")
        setattr(model, "fusion_mode", "mean")

    return {
        "sensor_reliability_configured": configured_reliability,
        "sensor_reliability_disabled": bool(disable_sensor_reliability),
        "sensor_reliability_active": bool(
            getattr(model, "sensor_reliability_enabled")
        ),
        "configured_modalities": list(configured_modalities),
        "posthoc_modalities": posthoc_modalities,
        "active_modalities": list(getattr(model, "enabled_modalities", ())),
        "sensor_grid_quality_configured": configured_grid_quality,
        "sensor_grid_quality_disabled": bool(disable_sensor_grid_quality),
        "sensor_grid_quality_active": bool(
            getattr(model, "sensor_grid_quality_enabled", False)
        ),
        "sensor_cross_modal_quality_configured": configured_cross_modal_quality,
        "sensor_cross_modal_quality_disabled": bool(
            disable_sensor_cross_modal_quality
        ),
        "sensor_cross_modal_quality_active": bool(
            getattr(model, "sensor_cross_modal_quality_enabled", False)
        ),
        "reliability_evidence_disabled": list(evidence),
        "film_configured": configured_film,
        "film_disabled": bool(disable_film),
        "film_active": bool(getattr(model, "film_enabled", False)),
        "fusion_mode_configured": configured_fusion_mode,
        "attention_disabled": bool(disable_attention),
        "fusion_mode_active": getattr(model, "fusion_mode", None),
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    config = load_config(args.config)
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    device = torch.device(
        args.device
        or training_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    checkpoint_path = resolve_project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到模型检查点：{checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = validated_checkpoint_class_names(checkpoint, config)
    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])
    intervention = apply_evaluation_interventions(
        model,
        disable_sensor_reliability=args.disable_sensor_reliability,
        posthoc_modalities=args.posthoc_modalities,
        disable_sensor_grid_quality=args.disable_sensor_grid_quality,
        disable_sensor_cross_modal_quality=(
            args.disable_sensor_cross_modal_quality
        ),
        disabled_reliability_evidence=tuple(
            args.disable_reliability_evidence
        ),
        disable_film=args.disable_film,
        disable_attention=args.disable_attention,
    )

    manifest_key = f"{args.split}_manifest"
    if manifest_key not in data_config:
        raise ValueError(f"配置 data 缺少 {manifest_key}")
    dataset = ManifestDataset(
        data_config[manifest_key],
        class_names=class_names,
        image_size=int(data_config.get("image_size", 640)),
        sensor_manifest=data_config.get("sensor_manifest"),
        preprocess=config.get("preprocess", {}),
        load_images=(
            "visual"
            in set(
                config.get("model", {}).get(
                    "enabled_modalities", ("visual", "text", "sensor")
                )
            )
        ),
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(training_config.get("batch_size", 16))
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training_config.get("num_workers", 0))
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    result = evaluate_model(
        model,
        loader,
        device=device,
        class_names=class_names,
        ece_bins=args.ece_bins,
        warmup_batches=args.warmup_batches,
    )
    if args.bootstrap_repeats:
        prediction_records = result["prediction_records"]
        result["confidence_intervals"] = (
            grouped_bootstrap_classification_intervals(
                np.asarray(
                    [record["target_id"] for record in prediction_records],
                    dtype=np.int64,
                ),
                np.asarray(
                    [record["probabilities"] for record in prediction_records],
                    dtype=np.float64,
                ),
                [
                    record["source_image_id"]
                    for record in prediction_records
                ],
                class_names,
                repeats=args.bootstrap_repeats,
                confidence=args.bootstrap_confidence,
                seed=args.bootstrap_seed,
            )
        )
    result["dataset"] = {
        "split": args.split,
        "manifest": str(resolve_project_path(data_config[manifest_key])),
        "evaluated_samples": len(dataset),
        "full_split_samples": (
            len(dataset.dataset) if isinstance(dataset, Subset) else len(dataset)
        ),
        "sensor_manifest": str(
            resolve_project_path(data_config["sensor_manifest"])
        ),
        "sensor_degradation_enabled": False,
    }
    result["checkpoint"] = _checkpoint_metadata(checkpoint, checkpoint_path)
    result["class_names"] = list(class_names)
    result["runtime"] = _runtime_metadata(device)
    result["posthoc_intervention"] = intervention
    result["evaluation_config"] = {
        "config_path": config["_config_path"],
        "image_size": int(data_config.get("image_size", 640)),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "ece_bins": args.ece_bins,
        "warmup_batches": args.warmup_batches,
        "max_samples": args.max_samples,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_confidence": args.bootstrap_confidence,
        "bootstrap_seed": args.bootstrap_seed,
    }
    output_dir = resolve_project_path(
        args.output
        if args.output is not None
        else PROJECT_ROOT / "reports" / "qwen-material" / args.split
    )
    artifacts = write_evaluation_artifacts(
        result,
        output_dir,
        class_names=class_names,
        split=args.split,
    )
    console_result = {
        "metrics": result["metrics"],
        "performance": result["performance"],
        "checkpoint": result["checkpoint"],
        "dataset": result["dataset"],
        "artifacts": artifacts,
    }
    if "confidence_intervals" in result:
        console_result["confidence_intervals"] = result[
            "confidence_intervals"
        ]
    print(json.dumps(console_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
