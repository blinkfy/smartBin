"""Independent whole-image detector-to-multimodal-classifier adapter."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import Tensor

from .checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from .model import build_production_model
from .sensors import contains_leakage_key, simulate_rule_sensor
from .training import patent_preprocess


def detection_sample_id(
    image_identity: str,
    detection_index: int,
    bbox: Sequence[float],
) -> str:
    payload = json.dumps(
        {
            "image": image_identity,
            "detection_index": int(detection_index),
            "bbox": [round(float(value), 3) for value in bbox],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _tensor_rows(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_ultralytics_detections(
    result: Any,
    *,
    include_masks: bool = False,
) -> list[dict[str, Any]]:
    """Convert one Ultralytics Results object into stable JSON records."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _tensor_rows(boxes.xyxy)
    confidence = _tensor_rows(boxes.conf).reshape(-1)
    class_ids = _tensor_rows(boxes.cls).reshape(-1)
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError("detector boxes.xyxy must have shape [N,4]")
    if not (len(xyxy) == len(confidence) == len(class_ids)):
        raise ValueError("detector box fields have inconsistent lengths")

    names = getattr(result, "names", {})
    mask_polygons: list[Any] = []
    masks = getattr(result, "masks", None)
    if include_masks and masks is not None:
        mask_polygons = list(getattr(masks, "xy", []) or [])

    detections: list[dict[str, Any]] = []
    for index, (bbox, score, class_id_value) in enumerate(
        zip(xyxy, confidence, class_ids)
    ):
        class_id = int(class_id_value)
        label = (
            str(names.get(class_id, class_id))
            if isinstance(names, Mapping)
            else str(names[class_id])
        )
        record: dict[str, Any] = {
            "detection_index": index,
            "bbox": [round(float(value), 3) for value in bbox],
            "detector_label_id": class_id,
            "detector_label": label,
            "detector_confidence": float(score),
        }
        if index < len(mask_polygons):
            polygon = np.asarray(mask_polygons[index], dtype=np.float64)
            if polygon.ndim == 2 and polygon.shape[1] == 2:
                record["mask_polygon"] = [
                    [round(float(x), 2), round(float(y), 2)]
                    for x, y in polygon
                ]
        detections.append(record)
    return detections


def _validated_grid(value: Any) -> list[list[float]]:
    if isinstance(value, Mapping):
        value = value.get("normalized_grid", value.get("grid"))
    array = np.asarray(value, dtype=np.float32)
    if array.size == 16 and array.shape != (4, 4):
        array = array.reshape(4, 4)
    if array.shape != (4, 4):
        raise ValueError("each sensor grid must be a 4x4 array")
    if not np.isfinite(array).all():
        raise ValueError("sensor grid values must be finite")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("sensor grid values must be in [0,1]")
    return array.tolist()


def load_detection_sidecar(
    document: Any,
    detection_count: int,
) -> list[list[list[float]]]:
    """Load one grid per detector output without accepting class fields."""

    records = document.get("detections") if isinstance(document, Mapping) else document
    if not isinstance(records, list):
        raise ValueError("sensor sidecar must be a list or contain detections")
    if contains_leakage_key(records):
        raise ValueError("sensor sidecar must not contain class or target fields")
    if len(records) != detection_count:
        raise ValueError(
            "sensor sidecar count must equal detector output count: "
            f"{len(records)} != {detection_count}"
        )
    grids: list[list[list[float]] | None] = [None] * detection_count
    for default_index, record in enumerate(records):
        index = (
            int(record.get("detection_index", default_index))
            if isinstance(record, Mapping)
            else default_index
        )
        if not 0 <= index < detection_count:
            raise ValueError(f"invalid detection_index in sensor sidecar: {index}")
        if grids[index] is not None:
            raise ValueError(f"duplicate detection_index in sensor sidecar: {index}")
        grids[index] = _validated_grid(record)
    if any(grid is None for grid in grids):
        raise ValueError("sensor sidecar does not cover every detection")
    return [grid for grid in grids if grid is not None]


def prepare_scene_sensors(
    image: Image.Image,
    detections: Sequence[Mapping[str, Any]],
    *,
    source: str,
    image_identity: str,
    sidecar: Any = None,
    rule_seed: int = 20260724,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    count = len(detections)
    if source not in {"missing", "rule-v1", "sidecar"}:
        raise ValueError("sensor source must be missing, rule-v1, or sidecar")
    if source == "sidecar":
        grids = load_detection_sidecar(sidecar, count)
        provenance = [{"generator": "provided-sidecar"} for _ in detections]
        availability = torch.ones(count, 1, dtype=torch.float32)
    elif source == "rule-v1":
        grids = []
        provenance = []
        for index, detection in enumerate(detections):
            bbox = [float(value) for value in detection["bbox"]]
            sample_id = detection_sample_id(image_identity, index, bbox)
            record = simulate_rule_sensor(
                image,
                bbox,
                sample_id,
                seed=rule_seed,
                source_image_id=image_identity,
            )
            grids.append(record["normalized_grid"])
            provenance.append(record["source"])
        availability = torch.ones(count, 1, dtype=torch.float32)
    else:
        grids = [[[0.0] * 4 for _ in range(4)] for _ in detections]
        provenance = [{"generator": "missing"} for _ in detections]
        availability = torch.zeros(count, 1, dtype=torch.float32)
    sensor = torch.tensor(grids, dtype=torch.float32).unsqueeze(1)
    return sensor, availability, provenance


def crop_scene_objects(
    image: Image.Image,
    detections: Sequence[Mapping[str, Any]],
    *,
    image_size: int,
    preprocess: Mapping[str, Any] | None,
) -> Tensor:
    """Crop detector boxes in memory using the production preprocessing path."""

    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    tensors: list[Tensor] = []
    for detection in detections:
        bbox = detection["bbox"]
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError("detection bbox must contain four coordinates")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        left = max(0, min(width - 1, math.floor(min(x1, x2))))
        top = max(0, min(height - 1, math.floor(min(y1, y2))))
        right = max(left + 1, min(width, math.ceil(max(x1, x2))))
        bottom = max(top + 1, min(height, math.ceil(max(y1, y2))))
        crop = image.crop((left, top, right, bottom))
        array_uint8 = np.asarray(crop, dtype=np.uint8).copy()
        if preprocess and bool(preprocess.get("enabled", False)):
            array_uint8 = patent_preprocess(
                array_uint8,
                dict(preprocess),
            )
        resized = Image.fromarray(array_uint8).resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    if not tensors:
        return torch.empty((0, 3, image_size, image_size), dtype=torch.float32)
    return torch.stack(tensors)


class SceneMultimodalClassifier:
    """Load one checkpoint and classify all detector boxes in one batch."""

    def __init__(
        self,
        config: Mapping[str, Any],
        checkpoint_path: Path,
        device: torch.device,
    ) -> None:
        checkpoint = load_checkpoint(checkpoint_path, device)
        self.class_names = validated_checkpoint_class_names(
            checkpoint,
            dict(config),
        )
        self.model = build_production_model(dict(config)).to(device)
        load_model_state_compatible(self.model, checkpoint["model_state"])
        self.model.eval()
        self.config = dict(config)
        self.device = device

    def predict(
        self,
        images: Tensor,
        sensors: Tensor,
        availability: Tensor,
    ) -> tuple[list[dict[str, Any]], float]:
        if not (len(images) == len(sensors) == len(availability)):
            raise ValueError("scene image, sensor, and availability counts differ")
        if len(images) == 0:
            return [], 0.0
        start = time.perf_counter()
        with torch.no_grad():
            logits, aux = self.model(
                images.to(self.device),
                sensors.to(self.device),
                sensor_available=availability.to(self.device),
                return_aux=True,
            )
            probabilities = torch.softmax(logits, dim=-1).cpu()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        outputs: list[dict[str, Any]] = []
        modality_weights = aux["modality_weights"].detach().cpu()
        scale_weights = aux["scale_weights"].detach().cpu()
        reliability = aux["sensor_reliability"].detach().cpu()
        available = aux["sensor_available"].detach().cpu()
        for index, row in enumerate(probabilities):
            predicted_id = int(row.argmax())
            outputs.append(
                {
                    "label_id": predicted_id,
                    "label": self.class_names[predicted_id],
                    "confidence": float(row[predicted_id]),
                    "probabilities": {
                        name: float(probability)
                        for name, probability in zip(self.class_names, row)
                    },
                    "modality_weights": modality_weights[index].tolist(),
                    "scale_weights": scale_weights[index].tolist(),
                    "sensor_reliability": float(reliability[index, 0]),
                    "sensor_available": float(available[index, 0]),
                }
            )
        return outputs, elapsed_ms


def annotate_scene(
    image: Image.Image,
    detections: Sequence[Mapping[str, Any]],
) -> Image.Image:
    annotated = ImageOps.exif_transpose(image).convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        bbox = tuple(float(value) for value in detection["bbox"])
        classification = detection.get("classification", {})
        label = str(classification.get("label", "unknown"))
        confidence = float(classification.get("confidence", 0.0))
        polygon = detection.get("mask_polygon")
        if isinstance(polygon, list) and len(polygon) >= 2:
            draw.line(
                [tuple(float(value) for value in point) for point in polygon],
                fill=(0, 170, 150),
                width=2,
                joint="curve",
            )
        draw.rectangle(bbox, outline=(220, 55, 45), width=3)
        text_origin = (bbox[0] + 3, max(0.0, bbox[1] - 13))
        draw.text(
            text_origin,
            f"{label} {confidence:.2f}",
            fill=(220, 55, 45),
        )
    return annotated
