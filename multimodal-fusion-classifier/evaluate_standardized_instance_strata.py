"""Evaluate target-centric box and mask recall by fixed instance strata."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from mmfc.model import PROJECT_ROOT


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "standardized-instance-strata"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_RUNTIME_ADDENDUM = (
    REPORT_ROOT / "FINAL_VALIDATION_ADDENDUM.json"
)
DEFAULT_OUTPUT = REPORT_ROOT / "evaluation-final-chunk64"
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
MODEL_ORDER = ("rgb_yolo", "v5_rgb_fallback", "v6_rgb_fallback")
DIMENSION_ORDER = ("coco_visible_mask_area", "bbox_overlap_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standardized instance-scale and overlap strata"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--runtime-addendum",
        type=Path,
        default=DEFAULT_RUNTIME_ADDENDUM,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            records.append(value)
    return records


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


def tree_binding(root: Path, pattern: str) -> dict[str, Any]:
    root = _resolve(root)
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for path in sorted(
        (item for item in root.glob(pattern) if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        size_bytes += path.stat().st_size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_count += 1
    return {
        "path": root.relative_to(PROJECT_ROOT).as_posix(),
        "pattern": pattern,
        "file_count": file_count,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def assign_coco_visible_scale(area_pixels: int) -> str:
    if area_pixels < 0:
        raise ValueError("visible mask area must be non-negative")
    if area_pixels < 32**2:
        return "small"
    if area_pixels < 96**2:
        return "medium"
    return "large"


def assign_overlap_proxy(max_intersection_over_target: float) -> str:
    value = float(max_intersection_over_target)
    if not 0.0 <= value <= 1.0:
        raise ValueError("overlap proxy must be in [0,1]")
    if value == 0.0:
        return "isolated"
    if value < 0.25:
        return "overlap_low"
    return "overlap_high"


def _bbox_intersection(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    return max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )


def _parse_targets(
    label_path: Path,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) < 7 or len(tokens) % 2 == 0:
            raise ValueError(
                f"invalid YOLO polygon at {label_path}:{line_number}"
            )
        points = np.asarray(
            [float(value) for value in tokens[1:]], dtype=np.float64
        ).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        integer_points = np.rint(points).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [integer_points], 1)
        bbox = [
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ]
        visible_area = int(mask.sum())
        targets.append(
            {
                "class_id": int(tokens[0]),
                "bbox": bbox,
                "mask": mask.astype(bool),
                "visible_mask_area_pixels": visible_area,
                "scale_stratum": assign_coco_visible_scale(visible_area),
            }
        )
    for target_index, target in enumerate(targets):
        x1, y1, x2, y2 = target["bbox"]
        target_bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        maximum = 0.0
        for other_index, other in enumerate(targets):
            if target_index == other_index:
                continue
            intersection = _bbox_intersection(
                target["bbox"], other["bbox"]
            )
            if target_bbox_area > 0.0:
                maximum = max(maximum, intersection / target_bbox_area)
        target["maximum_bbox_intersection_over_target"] = maximum
        target["overlap_stratum"] = assign_overlap_proxy(maximum)
    return targets


def load_target_contract(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = _resolve(manifest_path)
    images: list[dict[str, Any]] = []
    target_identity: list[dict[str, Any]] = []
    scale_counts: Counter[str] = Counter()
    overlap_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for record in _read_jsonl(manifest_path):
        source_id = str(record["source_image_id"])
        if source_id in seen:
            raise ValueError(f"duplicate source image id: {source_id}")
        seen.add(source_id)
        image_path = _resolve(Path(str(record["linked_image_path"])))
        label_path = _resolve(Path(str(record["label_path"])))
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"test pair is missing for {source_id}")
        with Image.open(image_path) as image:
            width, height = image.size
        targets = _parse_targets(
            label_path,
            width=int(width),
            height=int(height),
        )
        images.append(
            {
                "source_image_id": source_id,
                "image_path": image_path,
                "label_path": label_path,
                "width": int(width),
                "height": int(height),
                "targets": targets,
            }
        )
        for target_index, target in enumerate(targets):
            scale_counts[target["scale_stratum"]] += 1
            overlap_counts[target["overlap_stratum"]] += 1
            target_identity.append(
                {
                    "source_image_id": source_id,
                    "target_index": target_index,
                    "class_id": target["class_id"],
                    "visible_mask_area_pixels": target[
                        "visible_mask_area_pixels"
                    ],
                    "scale_stratum": target["scale_stratum"],
                    "maximum_bbox_intersection_over_target": round(
                        target[
                            "maximum_bbox_intersection_over_target"
                        ],
                        12,
                    ),
                    "overlap_stratum": target["overlap_stratum"],
                }
            )
    canonical = json.dumps(
        target_identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    contract = {
        "source_image_count": len(images),
        "target_count": len(target_identity),
        "image_shapes": dict(
            Counter(
                f"{record['width']}x{record['height']}" for record in images
            )
        ),
        "scale_support": {
            key: scale_counts[key] for key in ("small", "medium", "large")
        },
        "overlap_support": {
            key: overlap_counts[key]
            for key in ("isolated", "overlap_low", "overlap_high")
        },
        "target_contract_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    return images, contract


def box_iou_matrix(
    targets: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
) -> np.ndarray:
    result = np.zeros((len(targets), len(predictions)), dtype=np.float64)
    for target_index, target in enumerate(targets):
        tx1, ty1, tx2, ty2 = map(float, target)
        target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
        for prediction_index, prediction in enumerate(predictions):
            px1, py1, px2, py2 = map(float, prediction)
            prediction_area = max(0.0, px2 - px1) * max(
                0.0, py2 - py1
            )
            intersection = _bbox_intersection(target, prediction)
            union = target_area + prediction_area - intersection
            result[target_index, prediction_index] = (
                intersection / union if union > 0.0 else 0.0
            )
    return result


def mask_iou_matrix(
    targets: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
) -> np.ndarray:
    if not targets or not predictions:
        return np.zeros((len(targets), len(predictions)), dtype=np.float64)
    left = np.stack(targets).reshape(len(targets), -1).astype(np.float32)
    right = (
        np.stack(predictions)
        .reshape(len(predictions), -1)
        .astype(np.float32)
    )
    intersection = left @ right.T
    union = left.sum(axis=1, keepdims=True) + right.sum(
        axis=1, keepdims=True
    ).T - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    ).astype(np.float64)


def greedy_class_aware_matches(
    iou: np.ndarray,
    target_classes: Sequence[int],
    prediction_classes: Sequence[int],
    threshold: float,
) -> np.ndarray:
    if iou.shape != (len(target_classes), len(prediction_classes)):
        raise ValueError("IoU matrix shape is inconsistent with classes")
    candidates = [
        (float(iou[target_index, prediction_index]), target_index, prediction_index)
        for target_index, target_class in enumerate(target_classes)
        for prediction_index, prediction_class in enumerate(
            prediction_classes
        )
        if int(target_class) == int(prediction_class)
        and float(iou[target_index, prediction_index]) >= threshold
    ]
    candidates.sort(reverse=True)
    matched_targets: set[int] = set()
    matched_predictions: set[int] = set()
    for _, target_index, prediction_index in candidates:
        if (
            target_index in matched_targets
            or prediction_index in matched_predictions
        ):
            continue
        matched_targets.add(target_index)
        matched_predictions.add(prediction_index)
    result = np.zeros(len(target_classes), dtype=bool)
    if matched_targets:
        result[list(matched_targets)] = True
    return result


def _prediction_masks(result: Any) -> list[np.ndarray]:
    count = int(result.boxes.cls.shape[0])
    if count == 0:
        return []
    if result.masks is None:
        raise ValueError("segmentation model returned boxes without masks")
    height, width = map(int, result.orig_shape)
    masks = result.masks.data.detach().float().cpu().numpy()
    if masks.shape[0] != count:
        raise ValueError("prediction mask and box counts differ")
    output = []
    for mask in masks:
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        output.append(mask > 0.5)
    return output


def evaluate_model(
    model_spec: Mapping[str, Any],
    images: Sequence[Mapping[str, Any]],
    *,
    inference: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    checkpoint = _resolve(Path(str(model_spec["checkpoint"])))
    model = YOLO(str(checkpoint))
    availability: list[torch.Tensor] = []
    hook = None
    joint_fusion = getattr(model.model, "joint_fusion", None)
    if joint_fusion is not None:

        def collect_availability(module: Any, inputs: Any, output: Any) -> None:
            value = module.last_aux.get("sensor_available")
            if isinstance(value, torch.Tensor):
                availability.append(value.detach().float().cpu())

        hook = joint_fusion.register_forward_hook(collect_availability)

    started = perf_counter()
    per_image: list[dict[str, Any]] = []
    path_identity_modes: Counter[str] = Counter()
    chunk_size = int(inference.get("prediction_chunk_size", len(images)))
    if chunk_size <= 0:
        raise ValueError("prediction chunk size must be positive")
    try:
        for chunk_start in range(0, len(images), chunk_size):
            chunk = images[chunk_start : chunk_start + chunk_size]
            predictions = model.predict(
                source=[str(record["image_path"]) for record in chunk],
                imgsz=int(inference["image_size"]),
                conf=float(inference["confidence"]),
                iou=float(inference["nms_iou"]),
                max_det=int(inference["max_detections"]),
                batch=int(inference["batch_size"]),
                device=str(device),
                half=bool(inference["half"]),
                retina_masks=True,
                save=False,
                save_txt=False,
                verbose=False,
                stream=True,
            )
            for local_index, (expected, result) in enumerate(
                zip(chunk, predictions)
            ):
                source_id = str(expected["source_image_id"])
                result_stem = Path(str(result.path)).stem
                if result_stem == source_id:
                    path_identity_modes["source_image_id"] += 1
                elif result_stem == f"image{local_index}":
                    path_identity_modes["synthetic_list_index"] += 1
                else:
                    raise ValueError(
                        "prediction order mismatch: "
                        f"{result.path} != {source_id}"
                    )
                target_classes = [
                    int(target["class_id"])
                    for target in expected["targets"]
                ]
                prediction_classes = (
                    result.boxes.cls.detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )
                prediction_boxes = (
                    result.boxes.xyxy.detach()
                    .float()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                prediction_masks = _prediction_masks(result)
                box_iou = box_iou_matrix(
                    [target["bbox"] for target in expected["targets"]],
                    prediction_boxes,
                )
                mask_iou = mask_iou_matrix(
                    [target["mask"] for target in expected["targets"]],
                    prediction_masks,
                )
                box_matches = np.stack(
                    [
                        greedy_class_aware_matches(
                            box_iou,
                            target_classes,
                            prediction_classes,
                            threshold,
                        )
                        for threshold in IOU_THRESHOLDS
                    ],
                    axis=1,
                )
                mask_matches = np.stack(
                    [
                        greedy_class_aware_matches(
                            mask_iou,
                            target_classes,
                            prediction_classes,
                            threshold,
                        )
                        for threshold in IOU_THRESHOLDS
                    ],
                    axis=1,
                )
                per_image.append(
                    {
                        "source_image_id": source_id,
                        "prediction_count": len(prediction_classes),
                        "targets": [
                            {
                                "class_id": target["class_id"],
                                "scale_stratum": target["scale_stratum"],
                                "overlap_stratum": target[
                                    "overlap_stratum"
                                ],
                                "box_matches": box_matches[index]
                                .astype(int)
                                .tolist(),
                                "mask_matches": mask_matches[index]
                                .astype(int)
                                .tolist(),
                            }
                            for index, target in enumerate(
                                expected["targets"]
                            )
                        ],
                    }
                )
            del predictions
            if "result" in locals():
                del result
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if hook is not None:
            hook.remove()
    if len(per_image) != len(images):
        raise ValueError("prediction stream did not cover all test images")
    elapsed = perf_counter() - started
    availability_values = (
        torch.cat([value.reshape(-1) for value in availability])
        if availability
        else None
    )
    if model_spec.get("mode") == "rgb_fallback":
        if (
            availability_values is None
            or not bool((availability_values == 0.0).all())
        ):
            raise ValueError("RGB fallback did not keep sensor availability zero")
    return {
        "model": str(model_spec["name"]),
        "mode": str(model_spec["mode"]),
        "checkpoint": _binding(checkpoint),
        "source_image_count": len(per_image),
        "target_count": sum(len(row["targets"]) for row in per_image),
        "prediction_count": sum(
            int(row["prediction_count"]) for row in per_image
        ),
        "prediction_path_identity_modes": dict(path_identity_modes),
        "prediction_chunk_size": chunk_size,
        "elapsed_seconds": elapsed,
        "images_per_second": len(per_image) / elapsed,
        "sensor_availability": (
            {
                "sample_count": int(availability_values.numel()),
                "minimum": float(availability_values.min().item()),
                "maximum": float(availability_values.max().item()),
                "mean": float(availability_values.mean().item()),
            }
            if availability_values is not None
            else None
        ),
        "per_image": per_image,
    }


def _ratio_interval(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    valid = denominators > 0
    ratios = numerators[valid] / denominators[valid]
    alpha = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(ratios, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def build_contributions(
    model_result: Mapping[str, Any],
    *,
    image_count: int,
) -> dict[tuple[str, str, str], np.ndarray]:
    contributions: dict[tuple[str, str, str], np.ndarray] = {}
    for image_index, record in enumerate(model_result["per_image"]):
        for target in record["targets"]:
            memberships = (
                (
                    "coco_visible_mask_area",
                    str(target["scale_stratum"]),
                ),
                ("bbox_overlap_proxy", str(target["overlap_stratum"])),
            )
            for dimension, stratum in memberships:
                for metric in ("box", "mask"):
                    key = (dimension, stratum, metric)
                    if key not in contributions:
                        contributions[key] = np.zeros(
                            (image_count, 1 + len(IOU_THRESHOLDS)),
                            dtype=np.int64,
                        )
                    contributions[key][image_index, 0] += 1
                    contributions[key][image_index, 1:] += np.asarray(
                        target[f"{metric}_matches"], dtype=np.int64
                    )
    return contributions


def summarize_contributions(
    model_name: str,
    contributions: Mapping[tuple[str, str, str], np.ndarray],
    *,
    bootstrap_weights: np.ndarray,
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, np.ndarray]]]:
    rows = []
    samples: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for key in sorted(
        contributions,
        key=lambda item: (
            DIMENSION_ORDER.index(item[0]),
            item[1],
            item[2],
        ),
    ):
        dimension, stratum, metric = key
        values = contributions[key]
        point = values.sum(axis=0)
        sampled = bootstrap_weights @ values
        support = int(point[0])
        if support <= 0:
            raise ValueError(f"empty stratum: {key}")
        recall50 = float(point[1] / support)
        recall75 = float(point[6] / support)
        mean_recall = float(point[1:].sum() / (support * 10))
        sample_support = sampled[:, 0]
        sample_recall50 = np.divide(
            sampled[:, 1],
            sample_support,
            out=np.full(sample_support.shape, np.nan, dtype=np.float64),
            where=sample_support > 0,
        )
        sample_recall75 = np.divide(
            sampled[:, 6],
            sample_support,
            out=np.full(sample_support.shape, np.nan, dtype=np.float64),
            where=sample_support > 0,
        )
        sample_mean_recall = np.divide(
            sampled[:, 1:].sum(axis=1),
            sample_support * 10,
            out=np.full(sample_support.shape, np.nan, dtype=np.float64),
            where=sample_support > 0,
        )
        intervals = {
            "recall50": _ratio_interval(
                sampled[:, 1], sample_support, confidence=confidence
            ),
            "recall75": _ratio_interval(
                sampled[:, 6], sample_support, confidence=confidence
            ),
            "mean_recall50_95": _ratio_interval(
                sampled[:, 1:].sum(axis=1),
                sample_support * 10,
                confidence=confidence,
            ),
        }
        rows.append(
            {
                "model": model_name,
                "dimension": dimension,
                "stratum": stratum,
                "metric": metric,
                "support": support,
                "recall50": recall50,
                "recall50_ci_lower": intervals["recall50"][0],
                "recall50_ci_upper": intervals["recall50"][1],
                "recall75": recall75,
                "recall75_ci_lower": intervals["recall75"][0],
                "recall75_ci_upper": intervals["recall75"][1],
                "mean_recall50_95": mean_recall,
                "mean_recall50_95_ci_lower": intervals[
                    "mean_recall50_95"
                ][0],
                "mean_recall50_95_ci_upper": intervals[
                    "mean_recall50_95"
                ][1],
            }
        )
        samples[key] = {
            "recall50": sample_recall50,
            "recall75": sample_recall75,
            "mean_recall50_95": sample_mean_recall,
        }
    return rows, samples


def summarize_paired_differences(
    sample_metrics: Mapping[
        str, Mapping[tuple[str, str, str], Mapping[str, np.ndarray]]
    ],
    rows: Sequence[Mapping[str, Any]],
    *,
    confidence: float,
) -> list[dict[str, Any]]:
    point_lookup = {
        (
            str(row["model"]),
            str(row["dimension"]),
            str(row["stratum"]),
            str(row["metric"]),
        ): row
        for row in rows
    }
    alpha = 0.5 * (1.0 - confidence)
    output = []
    for candidate in ("v5_rgb_fallback", "v6_rgb_fallback"):
        reference = "rgb_yolo"
        for key in sample_metrics[candidate]:
            dimension, stratum, metric = key
            for score in ("recall50", "recall75", "mean_recall50_95"):
                samples = (
                    sample_metrics[candidate][key][score]
                    - sample_metrics[reference][key][score]
                )
                samples = samples[np.isfinite(samples)]
                if samples.size == 0:
                    raise ValueError(
                        f"paired bootstrap has no valid samples: {key}"
                    )
                lower, upper = np.quantile(
                    samples, [alpha, 1.0 - alpha]
                )
                candidate_point = point_lookup[
                    (candidate, dimension, stratum, metric)
                ][score]
                reference_point = point_lookup[
                    (reference, dimension, stratum, metric)
                ][score]
                output.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "dimension": dimension,
                        "stratum": stratum,
                        "metric": metric,
                        "score": score,
                        "difference": float(
                            candidate_point - reference_point
                        ),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                    }
                )
    return output


def _validate_protocol(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    runtime_addendum_path: Path,
) -> dict[str, Any]:
    if (
        protocol.get("protocol")
        != "standardized_instance_scale_overlap_diagnostic_v1"
        or protocol.get("status")
        != "frozen_before_new_prediction_run"
        or protocol.get("formal_paper_metric") is not False
        or protocol.get("iou_thresholds") != list(IOU_THRESHOLDS)
        or [model.get("name") for model in protocol.get("models", [])]
        != list(MODEL_ORDER)
    ):
        raise ValueError("standardized-strata protocol is inconsistent")
    runtime_addendum_path = _resolve(runtime_addendum_path)
    if not runtime_addendum_path.is_file():
        raise FileNotFoundError(
            f"runtime recovery addendum is missing: {runtime_addendum_path}"
        )
    final_addendum = _read_json(runtime_addendum_path)
    adjudication_binding = final_addendum.get("bindings", {}).get(
        "numerical_adjudication", {}
    )
    adjudication_path = _resolve(
        Path(str(adjudication_binding.get("path", "")))
    )
    if not adjudication_path.is_file():
        raise FileNotFoundError("numerical adjudication is missing")
    addendum = _read_json(adjudication_path)
    evaluator_binding = protocol.get("source_bindings", {}).get(
        "evaluator", {}
    )
    addendum2_binding = addendum.get("bindings", {}).get(
        "runtime_addendum_2", {}
    )
    addendum1_binding = addendum.get("bindings", {}).get(
        "runtime_addendum_1", {}
    )
    addendum2_path = _resolve(Path(str(addendum2_binding.get("path", ""))))
    addendum1_path = _resolve(Path(str(addendum1_binding.get("path", ""))))
    if not addendum2_path.is_file() or not addendum1_path.is_file():
        raise FileNotFoundError("runtime addendum chain is missing")
    addendum2 = _read_json(addendum2_path)
    addendum1 = _read_json(addendum1_path)
    if (
        addendum.get("status")
        != "frozen_after_batch_sensitivity_before_final_rerun"
        or addendum.get("valid") is not True
        or addendum.get("formal_paper_metric") is not False
        or addendum.get("bindings", {})
        .get("protocol", {})
        .get("sha256")
        != _sha256(protocol_path)
        or final_addendum.get("status")
        != "frozen_after_validator_binding_key_failure_before_final_rerun"
        or final_addendum.get("valid") is not True
        or final_addendum.get("formal_paper_metric") is not False
        or adjudication_binding.get("sha256")
        != _sha256(adjudication_path)
        or addendum2_binding.get("sha256") != _sha256(addendum2_path)
        or addendum1_binding.get("sha256") != _sha256(addendum1_path)
        or addendum1.get("source_revision", {}).get(
            "original_evaluator_sha256"
        )
        != evaluator_binding.get("sha256")
        or addendum2.get("source_revision", {}).get(
            "previous_evaluator_sha256"
        )
        != addendum1.get("source_revision", {}).get(
            "revised_evaluator_sha256"
        )
        or addendum.get("source_revision", {}).get(
            "previous_evaluator_sha256"
        )
        != addendum2.get("source_revision", {}).get(
            "revised_evaluator_sha256"
        )
        or addendum.get("source_revision", {}).get(
            "revised_evaluator_sha256"
        )
        != final_addendum.get("source_revision", {}).get(
            "previous_evaluator_sha256"
        )
        or final_addendum.get("source_revision", {}).get(
            "revised_evaluator_sha256"
        )
        != _sha256(Path(__file__))
        or addendum.get("adjudication", {})
        .get("selected_runtime", {})
        .get("batch_size")
        != 8
        or addendum.get("adjudication", {})
        .get("selected_runtime", {})
        .get("prediction_chunk_size")
        != 64
    ):
        raise ValueError("runtime recovery addendum is inconsistent")
    for name, binding in protocol.get("source_bindings", {}).items():
        if not isinstance(binding, Mapping):
            raise ValueError("malformed protocol source binding")
        path = _resolve(Path(str(binding["path"])))
        if name == "evaluator":
            if (
                path != Path(__file__).resolve()
                or binding.get("sha256")
                != addendum1["source_revision"][
                    "original_evaluator_sha256"
                ]
            ):
                raise ValueError("original evaluator binding is inconsistent")
            continue
        if (
            not path.is_file()
            or path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            raise ValueError(f"protocol source binding drift: {path}")
    if not protocol.get("source_bindings"):
        raise ValueError("protocol source bindings are missing")
    for binding in protocol.get("tree_bindings", {}).values():
        if not isinstance(binding, Mapping):
            raise ValueError("malformed protocol tree binding")
        actual = tree_binding(
            Path(str(binding["path"])),
            str(binding["pattern"]),
        )
        if actual != binding:
            raise ValueError(
                f"protocol tree binding drift: {binding['path']}"
            )
    if not protocol.get("tree_bindings"):
        raise ValueError("protocol tree bindings are missing")
    return addendum


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    if not rows:
        raise ValueError("cannot write empty metrics CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _format_interval(row: Mapping[str, Any], score: str) -> str:
    return (
        f"{float(row[score]) * 100:.2f}% "
        f"[{float(row[f'{score}_ci_lower']) * 100:.2f}%, "
        f"{float(row[f'{score}_ci_upper']) * 100:.2f}%]"
    )


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    rows = list(report["rows"])
    lines = [
        "# 标准化实例尺度与重叠代理诊断",
        "",
        (
            "该分析在新推理前冻结阈值，但模型和内部测试集此前已被检查，"
            "因此属于描述性、非正式性能证据。"
        ),
        "",
        "| 模型 | 维度 | 分层 | 指标 | N | R@0.50 [95% CI] | "
        "R@0.75 [95% CI] | mR@0.50:0.95 [95% CI] |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['dimension']} | {row['stratum']} | "
            f"{row['metric']} | {row['support']} | "
            f"{_format_interval(row, 'recall50')} | "
            f"{_format_interval(row, 'recall75')} | "
            f"{_format_interval(row, 'mean_recall50_95')} |"
        )
    lines.extend(
        [
            "",
            "## 样本量与解释边界",
            "",
            (
                f"- COCO可见掩膜面积 small/medium/large 支持数："
                f"{report['target_contract']['scale_support']}。"
            ),
            (
                f"- 框重叠 isolated/low/high 支持数："
                f"{report['target_contract']['overlap_support']}。"
            ),
            "- small 与 overlap_high 样本很少，区间较宽，不支持强结论。",
            (
                "- COCO 阈值只用于可见实例掩膜面积分层；这里报告的是固定"
                "conf/NMS下的目标召回，不是 COCOeval AP。"
            ),
            (
                "- 框交叠是拥挤或潜在遮挡代理，不是人工标注的遮挡比例，"
                "也不是不可见区域的amodal标注。"
            ),
            "- v5/v6条件均为同检查点RGB fallback，不包含外部sidecar。",
            "",
        ]
    )
    _atomic_text(output / "REPORT.md", "\n".join(lines))


def evaluate(
    protocol_path: Path = DEFAULT_PROTOCOL,
    runtime_addendum_path: Path = DEFAULT_RUNTIME_ADDENDUM,
    output: Path = DEFAULT_OUTPUT,
    *,
    device: str = "0",
) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    output = _resolve(output)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"protocol is missing: {protocol_path}")
    protocol = _read_json(protocol_path)
    runtime_addendum = _validate_protocol(
        protocol,
        protocol_path,
        runtime_addendum_path,
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = _resolve(Path(str(protocol["dataset"]["manifest"])))
    images, target_contract = load_target_contract(manifest_path)
    if target_contract != protocol["target_contract"]:
        raise ValueError("target contract differs from frozen protocol")
    bootstrap = protocol["bootstrap"]
    effective_inference = dict(protocol["inference"])
    effective_inference["batch_size"] = int(
        runtime_addendum["adjudication"]["selected_runtime"]["batch_size"]
    )
    effective_inference["prediction_chunk_size"] = int(
        runtime_addendum["adjudication"]["selected_runtime"][
            "prediction_chunk_size"
        ]
    )
    rng = np.random.default_rng(int(bootstrap["seed"]))
    image_count = len(images)
    bootstrap_weights = rng.multinomial(
        image_count,
        np.full(image_count, 1.0 / image_count),
        size=int(bootstrap["repeats"]),
    )

    model_results = []
    all_rows: list[dict[str, Any]] = []
    all_samples: dict[
        str, Mapping[tuple[str, str, str], Mapping[str, np.ndarray]]
    ] = {}
    for model_spec in protocol["models"]:
        result = evaluate_model(
            model_spec,
            images,
            inference=effective_inference,
            device=device,
        )
        contributions = build_contributions(
            result, image_count=image_count
        )
        rows, samples = summarize_contributions(
            str(model_spec["name"]),
            contributions,
            bootstrap_weights=bootstrap_weights,
            confidence=float(bootstrap["confidence"]),
        )
        model_results.append(
            {
                key: value
                for key, value in result.items()
                if key != "per_image"
            }
        )
        all_rows.extend(rows)
        all_samples[str(model_spec["name"])] = samples
        _atomic_text(
            output / f"{model_spec['name']}.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "valid": True,
                    "protocol": _binding(protocol_path),
                    "runtime_addendum": _binding(runtime_addendum_path),
                    "model": model_results[-1],
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    paired = summarize_paired_differences(
        all_samples,
        all_rows,
        confidence=float(bootstrap["confidence"]),
    )
    batch8_rgb = _read_json(output / "rgb_yolo.json")
    reference_paths = [
        _resolve(
            Path(str(runtime_addendum["bindings"][name]["path"]))
        )
        for name in ("batch32_rgb", "batch8_rgb")
    ]
    references = [_read_json(path) for path in reference_paths]
    previous_chunk_path = _resolve(
        Path(str(runtime_addendum["bindings"]["chunk64_rgb"]["path"]))
    )
    previous_chunk = _read_json(previous_chunk_path)
    point_keys = (
        "recall50",
        "recall75",
        "mean_recall50_95",
    )
    reference_rows = {
        (
            row["dimension"],
            row["stratum"],
            row["metric"],
        ): row
        for row in references[0]["rows"]
    }
    candidate_rows = {
        (
            row["dimension"],
            row["stratum"],
            row["metric"],
        ): row
        for row in batch8_rgb["rows"]
    }
    batch_sensitivity = {
        "exact_batch_invariance": False,
        "reference_batch_sizes": [32, 8],
        "candidate_batch_size": 8,
        "candidate_prediction_chunk_size": 64,
        "reference_artifacts": [
            _binding(path) for path in reference_paths
        ],
        "previous_chunked_artifact": _binding(previous_chunk_path),
        "candidate_artifact": _binding(output / "rgb_yolo.json"),
        "all_metric_rows_exact": all(
            reference.get("rows") == batch8_rgb.get("rows")
            for reference in references
        ),
        "chunked_metric_rows_reproduced_exactly": (
            previous_chunk.get("rows") == batch8_rgb.get("rows")
        ),
        "recall50_rows_exact": all(
            reference_rows[key]["recall50"]
            == candidate_rows[key]["recall50"]
            for key in reference_rows
        ),
        "recall75_rows_exact": all(
            reference_rows[key]["recall75"]
            == candidate_rows[key]["recall75"]
            for key in reference_rows
        ),
        "point_differences": [
            {
                "dimension": key[0],
                "stratum": key[1],
                "metric": key[2],
                "score": score,
                "difference": float(
                    candidate_rows[key][score]
                    - reference_rows[key][score]
                ),
            }
            for key in reference_rows
            for score in point_keys
            if candidate_rows[key][score] != reference_rows[key][score]
        ],
        "checkpoint_equal": all(
            reference.get("model", {})
            .get("checkpoint", {})
            .get("sha256")
            == batch8_rgb.get("model", {})
            .get("checkpoint", {})
            .get("sha256")
            for reference in references
        ),
        "prediction_count_equal": all(
            reference.get("model", {}).get("prediction_count")
            == batch8_rgb.get("model", {}).get("prediction_count")
            for reference in references
        ),
    }
    if not all(
        (
            batch_sensitivity["chunked_metric_rows_reproduced_exactly"],
            batch_sensitivity["recall50_rows_exact"],
            batch_sensitivity["recall75_rows_exact"],
            batch_sensitivity["checkpoint_equal"],
            batch_sensitivity["prediction_count_equal"],
        )
    ):
        raise ValueError("final chunked RGB result violates adjudication")
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "evidence_role": "posthoc_standardized_strata_diagnostic",
        "protocol": protocol["protocol"],
        "frozen_protocol": _binding(protocol_path),
        "runtime_recovery_addendum": _binding(runtime_addendum_path),
        "target_contract": target_contract,
        "frozen_inference": protocol["inference"],
        "effective_inference": effective_inference,
        "batch_sensitivity": batch_sensitivity,
        "numerical_adjudication": runtime_addendum["adjudication"],
        "iou_thresholds": list(IOU_THRESHOLDS),
        "bootstrap": bootstrap,
        "models": model_results,
        "rows": all_rows,
        "paired_differences": paired,
        "limitations": protocol["limitations"],
        "bindings": {
            "protocol": _binding(protocol_path),
            "runtime_addendum": _binding(runtime_addendum_path),
            "evaluator_source": _binding(Path(__file__)),
            "test_manifest": _binding(manifest_path),
        },
    }
    _atomic_text(
        output / "evaluation.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _write_csv(output / "metrics.csv", all_rows)
    _write_csv(output / "paired_differences.csv", paired)
    _write_report(output, report)
    return report


def main() -> None:
    args = parse_args()
    result = evaluate(
        args.protocol,
        args.runtime_addendum,
        args.output,
        device=str(args.device),
    )
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "model_count": len(result["models"]),
                "source_image_count": result["target_contract"][
                    "source_image_count"
                ],
                "target_count": result["target_contract"]["target_count"],
                "scale_support": result["target_contract"][
                    "scale_support"
                ],
                "overlap_support": result["target_contract"][
                    "overlap_support"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
