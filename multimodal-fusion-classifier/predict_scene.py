"""Whole-image YOLO detection/segmentation followed by MMFC classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch

from mmfc.scene_inference import (
    SceneMultimodalClassifier,
    annotate_scene,
    crop_scene_objects,
    extract_ultralytics_detections,
    prepare_scene_sensors,
)
from mmfc.training import PROJECT_ROOT, load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="整图 YOLO 检测/分割后执行多模态垃圾复分类"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "full_cross_modal_quality"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "ablations"
            / "full_cross_modal_quality.json"
        ),
    )
    parser.add_argument(
        "--sensor-source",
        choices=("missing", "rule-v1", "sidecar"),
        default="missing",
    )
    parser.add_argument(
        "--sensor-json",
        type=Path,
        help="sensor-source=sidecar 时，按 detection_index 提供每个目标的 4x4 阵列",
    )
    parser.add_argument("--rule-seed", type=int, default=20260724)
    parser.add_argument("--detector-confidence", type=float, default=0.8)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--detector-image-size", type=int, default=480)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--include-masks", action="store_true")
    parser.add_argument("--device", type=str)
    parser.add_argument("--detector-device", type=str)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-image", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.detector_confidence <= 1.0:
        raise ValueError("--detector-confidence must be in [0,1]")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in [0,1]")
    if args.detector_image_size <= 0:
        raise ValueError("--detector-image-size must be greater than zero")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be greater than zero")
    if args.sensor_source == "sidecar" and args.sensor_json is None:
        raise ValueError("sensor-source=sidecar requires --sensor-json")
    if args.sensor_source != "sidecar" and args.sensor_json is not None:
        raise ValueError("--sensor-json requires sensor-source=sidecar")


def _load_sidecar(path: Path | None) -> Any:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    image_path = args.image.expanduser().resolve()
    detector_checkpoint = args.detector_checkpoint.expanduser().resolve()
    classifier_checkpoint = resolve_project_path(args.checkpoint)
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到输入图像：{image_path}")
    if not detector_checkpoint.is_file():
        raise FileNotFoundError(f"找不到检测器检查点：{detector_checkpoint}")
    if not classifier_checkpoint.is_file():
        raise FileNotFoundError(
            f"找不到多模态分类器检查点：{classifier_checkpoint}"
        )

    config = load_config(args.config)
    device = torch.device(
        args.device
        or config.get("training", {}).get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    with Image.open(image_path) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")

    from ultralytics import YOLO

    detector = YOLO(str(detector_checkpoint))
    detector_start = time.perf_counter()
    detector_results = detector.predict(
        source=np.asarray(image),
        conf=args.detector_confidence,
        iou=args.nms_iou,
        imgsz=args.detector_image_size,
        max_det=args.max_detections,
        device=args.detector_device or str(device),
        verbose=False,
    )
    detector_ms = 1000.0 * (time.perf_counter() - detector_start)
    if len(detector_results) != 1:
        raise RuntimeError("single-image detector call returned an invalid batch")
    detections = extract_ultralytics_detections(
        detector_results[0],
        include_masks=args.include_masks,
    )

    sidecar = _load_sidecar(args.sensor_json)
    sensor, availability, provenance = prepare_scene_sensors(
        image,
        detections,
        source=args.sensor_source,
        image_identity=str(image_path),
        sidecar=sidecar,
        rule_seed=args.rule_seed,
    )
    image_size = int(config.get("data", {}).get("image_size", 480))
    crops = crop_scene_objects(
        image,
        detections,
        image_size=image_size,
        preprocess=config.get("preprocess", {}),
    )
    if detections:
        classifier = SceneMultimodalClassifier(
            config,
            classifier_checkpoint,
            device,
        )
        classifications, classifier_ms = classifier.predict(
            crops,
            sensor,
            availability,
        )
    else:
        classifications, classifier_ms = [], 0.0
    for detection, classification, sensor_source in zip(
        detections,
        classifications,
        provenance,
    ):
        detection["classification"] = classification
        detection["sensor_source"] = sensor_source

    result = {
        "schema_version": 1,
        "image": str(image_path),
        "image_size": {"width": image.width, "height": image.height},
        "detector": {
            "checkpoint": str(detector_checkpoint),
            "confidence_threshold": args.detector_confidence,
            "nms_iou_threshold": args.nms_iou,
            "image_size": args.detector_image_size,
        },
        "classifier": {
            "checkpoint": str(classifier_checkpoint),
            "config": str(resolve_project_path(args.config)),
            "sensor_source": args.sensor_source,
        },
        "timing_ms": {
            "detector": detector_ms,
            "classifier_batch": classifier_ms,
            "total_model": detector_ms + classifier_ms,
        },
        "detection_count": len(detections),
        "detections": detections,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_json.with_suffix(output_json.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output_json)
    if args.output_image is not None:
        output_image = args.output_image.expanduser().resolve()
        output_image.parent.mkdir(parents=True, exist_ok=True)
        annotate_scene(image, detections).save(output_image)
    print(rendered)


if __name__ == "__main__":
    main()
