"""Evaluate target-domain YOLO localization followed by MMFC reclassification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch

from mmfc.scene_inference import (
    SceneMultimodalClassifier,
    crop_scene_objects,
    extract_ultralytics_detections,
    prepare_scene_sensors,
)
from mmfc.data import read_jsonl
from mmfc.training import PROJECT_ROOT, load_config, resolve_project_path


DEFAULT_DATASET = PROJECT_ROOT / "data" / "yolo-seg-unique-linked"
DEFAULT_DETECTOR = (
    PROJECT_ROOT
    / "checkpoints"
    / "yolo-seg"
    / "yolo11s-seg-qwen-split"
    / "weights"
    / "best.pt"
)
DEFAULT_CLASSIFIER_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "ablations"
    / "full_cross_modal_quality.json"
)
DEFAULT_CLASSIFIER = (
    PROJECT_ROOT
    / "checkpoints"
    / "ablations"
    / "full_cross_modal_quality"
    / "best.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "yolo-seg" / "integrated"
)
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "yolo-seg"
    / "INTEGRATED_EVALUATION_PROTOCOL.json"
)
DEFAULT_PROTOCOL_ADDENDUM = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "yolo-seg"
    / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 YOLO-seg 定位后 v3 多模态复分类的固定测试性能"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=DEFAULT_DETECTOR,
    )
    parser.add_argument(
        "--classifier-config",
        type=Path,
        default=DEFAULT_CLASSIFIER_CONFIG,
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=DEFAULT_CLASSIFIER,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--protocol-addendum",
        type=Path,
        default=DEFAULT_PROTOCOL_ADDENDUM,
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.8],
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--detector-image-size", type=int, default=640)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--classifier-device", default="cuda:0")
    parser.add_argument("--rule-seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol_file_hashes(
    protocol: Mapping[str, Any],
    *,
    project_root: Path,
    expected_hash_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("protocol inputs must be an object")
    for name, entry in inputs.items():
        if not isinstance(entry, Mapping) or "sha256" not in entry:
            continue
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"protocol input {name} has no path")
        path = Path(path_value)
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (project_root / path).resolve()
        )
        if not resolved.is_file():
            raise FileNotFoundError(
                f"protocol input {name} is missing: {resolved}"
            )
        expected = str(
            (expected_hash_overrides or {}).get(name, entry["sha256"])
        ).lower()
        actual = _sha256(resolved)
        if actual != expected:
            raise ValueError(
                f"protocol input hash mismatch for {name}: "
                f"{actual} != {expected}"
            )
        checks.append(
            {
                "name": str(name),
                "path": resolved.as_posix(),
                "sha256": actual,
            }
        )
    return checks


def validate_protocol_addendum(
    addendum: Mapping[str, Any],
    *,
    addendum_path: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    evaluation_source: Path,
) -> dict[str, str]:
    if (
        addendum.get("schema_version") != 1
        or addendum.get("status")
        != "frozen_after_path_identity_failure_before_recovery_rerun"
    ):
        raise ValueError("integrated evaluation protocol addendum is not frozen")
    protocol_binding = addendum.get("original_protocol", {})
    implementation = addendum.get("implementation_fix", {})
    original_source = implementation.get("original_evaluation_source", {})
    revised_source = implementation.get("revised_evaluation_source", {})
    protocol_source = protocol.get("inputs", {}).get(
        "evaluation_source", {}
    )
    if (
        protocol_binding.get("sha256") != _sha256(protocol_path)
        or original_source.get("sha256") != protocol_source.get("sha256")
        or revised_source.get("sha256") != _sha256(evaluation_source)
        or implementation.get("allowed_change")
        != "accept_ultralytics_synthetic_list_index_for_order_validation"
    ):
        raise ValueError("integrated evaluation protocol addendum mismatch")
    return {
        "path": addendum_path.resolve().as_posix(),
        "sha256": _sha256(addendum_path),
        "evaluation_source_sha256": str(revised_source["sha256"]),
    }


def _validate_protocol_arguments(
    protocol: Mapping[str, Any],
    args: argparse.Namespace,
    thresholds: Sequence[float],
) -> None:
    detector = protocol["detector_inference"]
    matching = protocol["matching"]
    classification = protocol["classification"]
    statistics = protocol["paired_statistics"]
    protocol_inputs = protocol["inputs"]
    expected_thresholds = sorted(
        float(value) for value in detector["reported_confidence_thresholds"]
    )
    checks = {
        "confidence_thresholds": (
            list(thresholds),
            expected_thresholds,
        ),
        "nms_iou": (float(args.nms_iou), float(detector["nms_iou"])),
        "match_iou": (
            float(args.match_iou),
            float(matching["box_iou_threshold"]),
        ),
        "detector_image_size": (
            int(args.detector_image_size),
            int(detector["image_size"]),
        ),
        "detector_batch_size": (
            int(args.detector_batch_size),
            int(detector["batch_size"]),
        ),
        "classifier_batch_size": (
            int(args.classifier_batch_size),
            int(classification["classifier_batch_size"]),
        ),
        "max_detections": (
            int(args.max_detections),
            int(detector["max_detections"]),
        ),
        "rule_seed": (
            int(args.rule_seed),
            int(classification["rule_seed"]),
        ),
        "bootstrap_repeats": (
            int(args.bootstrap_repeats),
            int(statistics["repeats"]),
        ),
        "bootstrap_seed": (
            int(args.bootstrap_seed),
            int(statistics["seed"]),
        ),
    }
    mismatches = [
        f"{name}: {actual!r} != {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    expected_paths = {
        "dataset": Path(
            str(protocol_inputs["dataset_export"]["path"])
        ).parent,
        "detector_checkpoint": Path(
            str(protocol_inputs["detector_checkpoint"]["path"])
        ),
        "classifier_config": Path(
            str(protocol_inputs["classifier_config"]["path"])
        ),
        "classifier_checkpoint": Path(
            str(protocol_inputs["classifier_checkpoint"]["path"])
        ),
    }
    actual_paths = {
        "dataset": args.dataset,
        "detector_checkpoint": args.detector_checkpoint,
        "classifier_config": args.classifier_config,
        "classifier_checkpoint": args.classifier_checkpoint,
    }
    for name, expected_path in expected_paths.items():
        actual_resolved = resolve_project_path(actual_paths[name])
        expected_resolved = resolve_project_path(expected_path)
        if actual_resolved != expected_resolved:
            mismatches.append(
                f"{name}: {actual_resolved} != {expected_resolved}"
            )
    if mismatches:
        raise ValueError(
            "evaluation arguments differ from frozen protocol: "
            + "; ".join(mismatches)
        )


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("boxes must contain four coordinates")
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_iou_match(
    ground_truth: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in [0,1]")
    candidates = []
    for ground_truth_index, target in enumerate(ground_truth):
        for detection_index, detection in enumerate(detections):
            iou = box_iou(target["bbox"], detection["bbox"])
            if iou >= iou_threshold:
                candidates.append(
                    (-iou, ground_truth_index, detection_index)
                )
    candidates.sort()
    used_ground_truth: set[int] = set()
    used_detections: set[int] = set()
    matches: list[dict[str, Any]] = []
    for negative_iou, ground_truth_index, detection_index in candidates:
        if (
            ground_truth_index in used_ground_truth
            or detection_index in used_detections
        ):
            continue
        used_ground_truth.add(ground_truth_index)
        used_detections.add(detection_index)
        matches.append(
            {
                "ground_truth_index": ground_truth_index,
                "detection_index": detection_index,
                "iou": -negative_iou,
            }
        )
    matches.sort(key=lambda item: item["ground_truth_index"])
    return {
        "matches": matches,
        "unmatched_ground_truth_indices": [
            index
            for index in range(len(ground_truth))
            if index not in used_ground_truth
        ],
        "unmatched_detection_indices": [
            index
            for index in range(len(detections))
            if index not in used_detections
        ],
    }


def _prediction_id(
    detection: Mapping[str, Any],
    prediction_source: str,
) -> int:
    if prediction_source == "detector":
        return int(detection["detector_label_id"])
    classifications = detection.get("classifications")
    if not isinstance(classifications, Mapping):
        raise ValueError("detection is missing MMFC classifications")
    classification = classifications.get(prediction_source)
    if not isinstance(classification, Mapping):
        raise ValueError(
            f"detection is missing classification source {prediction_source}"
        )
    return int(classification["label_id"])


def summarize_end_to_end(
    images: Sequence[Mapping[str, Any]],
    *,
    class_names: Sequence[str],
    confidence_threshold: float,
    match_iou: float,
    prediction_source: str,
    include_per_source: bool = True,
) -> dict[str, Any]:
    class_names = tuple(class_names)
    class_count = len(class_names)
    true_positive = np.zeros(class_count, dtype=np.int64)
    false_positive = np.zeros(class_count, dtype=np.int64)
    false_negative = np.zeros(class_count, dtype=np.int64)
    support = np.zeros(class_count, dtype=np.int64)
    detection_count = 0
    matched_count = 0
    matched_correct = 0
    source_rows: list[dict[str, Any]] = []

    for image in images:
        ground_truth = list(image["ground_truth"])
        detections = [
            detection
            for detection in image["detections"]
            if float(detection["detector_confidence"])
            >= confidence_threshold
        ]
        detection_count += len(detections)
        for target in ground_truth:
            support[int(target["class_id"])] += 1
        matching = greedy_iou_match(
            ground_truth,
            detections,
            iou_threshold=match_iou,
        )
        image_correct = 0
        for match in matching["matches"]:
            target = ground_truth[match["ground_truth_index"]]
            detection = detections[match["detection_index"]]
            target_id = int(target["class_id"])
            predicted_id = _prediction_id(detection, prediction_source)
            if not 0 <= predicted_id < class_count:
                raise ValueError("predicted class id is outside class_names")
            matched_count += 1
            if predicted_id == target_id:
                true_positive[target_id] += 1
                matched_correct += 1
                image_correct += 1
            else:
                false_negative[target_id] += 1
                false_positive[predicted_id] += 1
        for index in matching["unmatched_ground_truth_indices"]:
            false_negative[int(ground_truth[index]["class_id"])] += 1
        for index in matching["unmatched_detection_indices"]:
            predicted_id = _prediction_id(
                detections[index], prediction_source
            )
            if not 0 <= predicted_id < class_count:
                raise ValueError("predicted class id is outside class_names")
            false_positive[predicted_id] += 1
        if include_per_source:
            source_rows.append(
                {
                    "source_image_id": str(image["source_image_id"]),
                    "ground_truth_count": len(ground_truth),
                    "detection_count": len(detections),
                    "localized_count": len(matching["matches"]),
                    "localized_and_classified_count": image_correct,
                }
            )

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros(class_count, dtype=np.float64),
        where=(true_positive + false_positive) > 0,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros(class_count, dtype=np.float64),
        where=(true_positive + false_negative) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    ground_truth_count = int(support.sum())
    correct_count = int(true_positive.sum())
    return {
        "prediction_source": prediction_source,
        "confidence_threshold": float(confidence_threshold),
        "match_iou_threshold": float(match_iou),
        "source_image_count": len(images),
        "ground_truth_count": ground_truth_count,
        "detection_count": detection_count,
        "localized_count": matched_count,
        "localized_and_classified_count": correct_count,
        "localization_precision": (
            matched_count / detection_count if detection_count else 0.0
        ),
        "localization_recall": (
            matched_count / ground_truth_count
            if ground_truth_count
            else 0.0
        ),
        "classification_accuracy_on_localized": (
            matched_correct / matched_count if matched_count else 0.0
        ),
        "end_to_end_precision": (
            correct_count / detection_count if detection_count else 0.0
        ),
        "end_to_end_recall": (
            correct_count / ground_truth_count
            if ground_truth_count
            else 0.0
        ),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "class_id": index,
                "class_name": name,
                "support": int(support[index]),
                "true_positive": int(true_positive[index]),
                "false_positive": int(false_positive[index]),
                "false_negative": int(false_negative[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(class_names)
        ],
        "per_source": source_rows,
    }


def paired_source_bootstrap(
    images: Sequence[Mapping[str, Any]],
    *,
    class_names: Sequence[str],
    confidence_threshold: float,
    match_iou: float,
    reference_source: str,
    candidate_source: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("bootstrap repeats must be at least 2")
    if not images:
        raise ValueError("bootstrap images must not be empty")
    metric_names = (
        "classification_accuracy_on_localized",
        "end_to_end_precision",
        "end_to_end_recall",
        "macro_f1",
    )
    reference = summarize_end_to_end(
        images,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        match_iou=match_iou,
        prediction_source=reference_source,
        include_per_source=False,
    )
    candidate = summarize_end_to_end(
        images,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        match_iou=match_iou,
        prediction_source=candidate_source,
        include_per_source=False,
    )
    rng = np.random.default_rng(seed)
    differences = {
        metric: np.empty(repeats, dtype=np.float64)
        for metric in metric_names
    }
    for repeat in range(repeats):
        indices = rng.integers(0, len(images), size=len(images))
        sampled = [images[int(index)] for index in indices]
        sampled_reference = summarize_end_to_end(
            sampled,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            match_iou=match_iou,
            prediction_source=reference_source,
            include_per_source=False,
        )
        sampled_candidate = summarize_end_to_end(
            sampled,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            match_iou=match_iou,
            prediction_source=candidate_source,
            include_per_source=False,
        )
        for metric in metric_names:
            differences[metric][repeat] = (
                float(sampled_candidate[metric])
                - float(sampled_reference[metric])
            )
    return {
        "reference_source": reference_source,
        "candidate_source": candidate_source,
        "cluster_unit": "source_image_id",
        "repeats": repeats,
        "confidence": 0.95,
        "seed": seed,
        "differences": {
            metric: {
                "point": (
                    float(candidate[metric]) - float(reference[metric])
                ),
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for metric, values in differences.items()
        },
    }


def _load_ground_truth(
    label_path: Path,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    ground_truth: list[dict[str, Any]] = []
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
        class_id = int(tokens[0])
        coordinates = np.asarray(
            [float(value) for value in tokens[1:]],
            dtype=np.float64,
        ).reshape(-1, 2)
        ground_truth.append(
            {
                "class_id": class_id,
                "bbox": [
                    float(coordinates[:, 0].min() * width),
                    float(coordinates[:, 1].min() * height),
                    float(coordinates[:, 0].max() * width),
                    float(coordinates[:, 1].max() * height),
                ],
            }
        )
    return ground_truth


def _load_test_images(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    export_report = json.loads(
        (dataset_dir / "export_report.json").read_text(encoding="utf-8")
    )
    class_names = [str(name) for name in export_report["class_names"]]
    images: list[dict[str, Any]] = []
    for record in read_jsonl(dataset_dir / "manifests" / "test.jsonl"):
        original_path = Path(str(record["original_image_path"]))
        linked_path = Path(str(record["linked_image_path"]))
        label_path = Path(str(record["label_path"]))
        if not original_path.is_file() or not linked_path.is_file():
            raise FileNotFoundError(
                f"missing original/link for {record['source_image_id']}"
            )
        with Image.open(original_path) as source:
            width, height = ImageOps.exif_transpose(source).size
        images.append(
            {
                "source_image_id": str(record["source_image_id"]),
                "original_image_path": original_path,
                "linked_image_path": linked_path,
                "ground_truth": _load_ground_truth(
                    label_path,
                    width=width,
                    height=height,
                ),
                "detections": [],
            }
        )
    return images, class_names


def _result_path_matches(
    result_path: Path,
    *,
    original_path: Path,
    result_index: int,
) -> bool:
    synthetic_index_path = (
        not result_path.is_absolute()
        and result_path.parent == Path(".")
        and result_path.name == f"image{result_index}.jpg"
    )
    return synthetic_index_path or (
        result_path.resolve() == original_path.resolve()
    )


def _flush_classifier_batch(
    *,
    classifier: SceneMultimodalClassifier,
    pending: dict[str, list[Any]],
) -> dict[str, float]:
    if not pending["record"]:
        return {"missing": 0.0, "rule-v1": 0.0}
    images = torch.stack(pending["crop"])
    timings: dict[str, float] = {}
    for source in ("missing", "rule-v1"):
        sensors = torch.stack(pending[f"{source}_sensor"])
        availability = torch.stack(pending[f"{source}_availability"])
        outputs, elapsed_ms = classifier.predict(
            images,
            sensors,
            availability,
        )
        timings[source] = elapsed_ms
        for detection, output in zip(pending["record"], outputs):
            detection.setdefault("classifications", {})[source] = output
    for values in pending.values():
        values.clear()
    return timings


def _runtime() -> dict[str, Any]:
    import ultralytics

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def _write_artifacts(
    output_dir: Path,
    report: Mapping[str, Any],
) -> None:
    _atomic_write_text(
        output_dir / "evaluation.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    metrics_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for threshold_key, sources in report["scenarios"].items():
        for source, metrics in sources.items():
            metrics_rows.append(
                {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"per_class", "per_source"}
                }
            )
            for row in metrics["per_class"]:
                per_class_rows.append(
                    {
                        "confidence_threshold": threshold_key,
                        "prediction_source": source,
                        **row,
                    }
                )
    for path, rows in (
        (output_dir / "metrics.csv", metrics_rows),
        (output_dir / "per_class.csv", per_class_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    with (output_dir / "per_image.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for image in report["images"]:
            handle.write(
                json.dumps(image, ensure_ascii=False, sort_keys=True) + "\n"
            )

    lines = [
        "# YOLO-seg + MMFC 整图固定测试评估",
        "",
        "## 主表",
        "",
        "| Conf | 类别来源 | 定位 P | 定位 R | 匹配分类 Acc | 端到端 P | "
        "端到端 R | Macro F1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "detector": "YOLO 类别",
        "missing": "v3 缺失传感器",
        "rule-v1": "v3 rule-v1",
    }
    for threshold_key, sources in report["scenarios"].items():
        for source, metrics in sources.items():
            percent = lambda key: f"{100.0 * float(metrics[key]):.2f}%"
            lines.append(
                f"| {threshold_key} | {labels[source]} | "
                f"{percent('localization_precision')} | "
                f"{percent('localization_recall')} | "
                f"{percent('classification_accuracy_on_localized')} | "
                f"{percent('end_to_end_precision')} | "
                f"{percent('end_to_end_recall')} | "
                f"{percent('macro_f1')} |"
            )
    lines.extend(
        [
            "",
            "## 协议边界",
            "",
            "- 一对一贪心匹配按 box IoU 从高到低执行，阈值为 0.5；",
            "- 同一轮 detector 输出同时复算 confidence 0.25 与专利阈值 0.8；",
            "- `missing` 不使用软传感器，`rule-v1` 只读取预测框裁剪和几何；",
            "- MMFC 使用锁定 v3 检查点，不根据整图 test 结果调参；",
            "- 指标是固定内部 test 的系统联调证据，不是外部盲测或真实硬件部署精度。",
            "",
            "## 配对统计",
            "",
            "所有类别来源差异均按 `source_image_id` 聚类配对 bootstrap；区间、重复次数"
            "与种子见 `evaluation.json`。该统计只比较同一批 detector 输出上的类别来源，"
            "不把它解释为检测器训练因果效应。",
            "",
        ]
    )
    _atomic_write_text(output_dir / "REPORT.md", "\n".join(lines))


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = sorted(set(float(value) for value in args.confidence_thresholds))
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("confidence thresholds must be in (0,1]")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("NMS IoU must be in [0,1]")
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("match IoU must be in [0,1]")
    if args.detector_image_size <= 0:
        raise ValueError("detector image size must be greater than zero")
    if args.detector_batch_size <= 0 or args.classifier_batch_size <= 0:
        raise ValueError("batch sizes must be greater than zero")
    if args.max_detections <= 0:
        raise ValueError("max detections must be greater than zero")
    if args.bootstrap_repeats < 2:
        raise ValueError("bootstrap repeats must be at least 2")
    protocol_path = resolve_project_path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_detector_training":
        raise ValueError("integrated evaluation protocol is not frozen")
    _validate_protocol_arguments(protocol, args, thresholds)
    addendum_path = resolve_project_path(args.protocol_addendum)
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    addendum_binding = validate_protocol_addendum(
        addendum,
        addendum_path=addendum_path,
        protocol_path=protocol_path,
        protocol=protocol,
        evaluation_source=Path(__file__).resolve(),
    )
    protocol_hash_checks = verify_protocol_file_hashes(
        protocol,
        project_root=PROJECT_ROOT,
        expected_hash_overrides={
            "evaluation_source": addendum_binding[
                "evaluation_source_sha256"
            ]
        },
    )

    dataset_dir = resolve_project_path(args.dataset)
    detector_checkpoint = resolve_project_path(args.detector_checkpoint)
    classifier_config_path = resolve_project_path(args.classifier_config)
    classifier_checkpoint = resolve_project_path(args.classifier_checkpoint)
    output_dir = resolve_project_path(args.output)
    for path in (
        dataset_dir / "export_report.json",
        detector_checkpoint,
        classifier_config_path,
        classifier_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )

    images, class_names = _load_test_images(dataset_dir)
    config = load_config(classifier_config_path)
    classifier_device = torch.device(args.classifier_device)
    classifier = SceneMultimodalClassifier(
        config,
        classifier_checkpoint,
        classifier_device,
    )
    if list(classifier.class_names) != class_names:
        raise ValueError("detector dataset and MMFC class order differ")

    from ultralytics import YOLO

    detector = YOLO(str(detector_checkpoint))
    pending: dict[str, list[Any]] = {
        "record": [],
        "crop": [],
        "missing_sensor": [],
        "missing_availability": [],
        "rule-v1_sensor": [],
        "rule-v1_availability": [],
    }
    classifier_ms = {"missing": 0.0, "rule-v1": 0.0}
    linked_paths = [str(image["linked_image_path"]) for image in images]
    started = time.perf_counter()
    results = detector.predict(
        source=linked_paths,
        stream=True,
        batch=args.detector_batch_size,
        conf=min(thresholds),
        iou=args.nms_iou,
        imgsz=args.detector_image_size,
        max_det=args.max_detections,
        device=args.detector_device,
        verbose=False,
        save=False,
    )
    processed_images = 0
    detector_names_validated = False
    for image_record, result in zip(images, results):
        result_index = processed_images
        processed_images += 1
        result_path = Path(str(result.path))
        if not _result_path_matches(
            result_path,
            original_path=image_record["original_image_path"],
            result_index=result_index,
        ):
            raise ValueError(
                "detector result order/path differs from test manifest: "
                f"{result_path.resolve()}"
            )
        if not detector_names_validated:
            names = getattr(result, "names", {})
            detector_class_names = (
                [str(names[index]) for index in range(len(names))]
                if isinstance(names, Mapping)
                else [str(name) for name in names]
            )
            if detector_class_names != class_names:
                raise ValueError("detector and dataset class order differ")
            detector_names_validated = True
        detections = extract_ultralytics_detections(result)
        image_record["detections"] = detections
        if not detections:
            continue
        with Image.open(image_record["original_image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        crops = crop_scene_objects(
            image,
            detections,
            image_size=int(config["data"]["image_size"]),
            preprocess=config.get("preprocess", {}),
        )
        sensor_modes = {}
        for sensor_source in ("missing", "rule-v1"):
            sensor_modes[sensor_source] = prepare_scene_sensors(
                image,
                detections,
                source=sensor_source,
                image_identity=str(image_record["source_image_id"]),
                rule_seed=args.rule_seed,
            )
        for index, detection in enumerate(detections):
            pending["record"].append(detection)
            pending["crop"].append(crops[index])
            for sensor_source in ("missing", "rule-v1"):
                sensors, availability, _ = sensor_modes[sensor_source]
                pending[f"{sensor_source}_sensor"].append(sensors[index])
                pending[f"{sensor_source}_availability"].append(
                    availability[index]
                )
            if len(pending["record"]) >= args.classifier_batch_size:
                elapsed = _flush_classifier_batch(
                    classifier=classifier,
                    pending=pending,
                )
                for source, milliseconds in elapsed.items():
                    classifier_ms[source] += milliseconds
    if processed_images != len(images):
        raise RuntimeError(
            "detector returned a different number of results than test images: "
            f"{processed_images} != {len(images)}"
        )
    elapsed = _flush_classifier_batch(classifier=classifier, pending=pending)
    for source, milliseconds in elapsed.items():
        classifier_ms[source] += milliseconds
    total_seconds = time.perf_counter() - started

    scenarios: dict[str, Any] = {}
    paired_comparisons: dict[str, Any] = {}
    for threshold_index, threshold in enumerate(thresholds):
        threshold_key = f"{threshold:.2f}"
        scenarios[threshold_key] = {
            source: summarize_end_to_end(
                images,
                class_names=class_names,
                confidence_threshold=threshold,
                match_iou=args.match_iou,
                prediction_source=source,
            )
            for source in ("detector", "missing", "rule-v1")
        }
        comparison_specs = (
            ("missing_vs_detector", "detector", "missing"),
            ("rule_v1_vs_detector", "detector", "rule-v1"),
            ("rule_v1_vs_missing", "missing", "rule-v1"),
        )
        paired_comparisons[threshold_key] = {
            name: paired_source_bootstrap(
                images,
                class_names=class_names,
                confidence_threshold=threshold,
                match_iou=args.match_iou,
                reference_source=reference,
                candidate_source=candidate,
                repeats=args.bootstrap_repeats,
                seed=args.bootstrap_seed + threshold_index * 10 + index,
            )
            for index, (name, reference, candidate) in enumerate(
                comparison_specs
            )
        }
    serializable_images = [
        {
            "source_image_id": image["source_image_id"],
            "ground_truth": image["ground_truth"],
            "detections": image["detections"],
        }
        for image in images
    ]
    report = {
        "schema_version": 1,
        "protocol": "fixed_source_grouped_scene_test",
        "frozen_protocol": {
            "path": protocol_path.resolve().as_posix(),
            "sha256": _sha256(protocol_path),
            "input_hash_checks": protocol_hash_checks,
            "addendum": addendum_binding,
        },
        "scenarios": scenarios,
        "paired_comparisons": paired_comparisons,
        "class_names": class_names,
        "dataset": {
            "path": dataset_dir.resolve().as_posix(),
            "source_image_count": len(images),
            "ground_truth_object_count": sum(
                len(image["ground_truth"]) for image in images
            ),
            "export_report_sha256": _sha256(
                dataset_dir / "export_report.json"
            ),
            "prediction_order_validation": (
                "ultralytics synthetic image{index}.jpg list index or "
                "resolved original path"
            ),
        },
        "detector": {
            "checkpoint": detector_checkpoint.resolve().as_posix(),
            "sha256": _sha256(detector_checkpoint),
            "minimum_confidence": min(thresholds),
            "reported_confidence_thresholds": thresholds,
            "nms_iou": float(args.nms_iou),
            "image_size": int(args.detector_image_size),
            "max_detections": int(args.max_detections),
        },
        "classifier": {
            "config": classifier_config_path.resolve().as_posix(),
            "checkpoint": classifier_checkpoint.resolve().as_posix(),
            "checkpoint_sha256": _sha256(classifier_checkpoint),
            "sensor_sources": ["missing", "rule-v1"],
            "rule_seed": int(args.rule_seed),
        },
        "timing": {
            "scene_iteration_seconds": total_seconds,
            "classifier_batch_ms": classifier_ms,
            "total_seconds": total_seconds,
            "scope_note": (
                "Offline batch evaluation including detector iteration, crop "
                "preprocessing, rule sensors, and classifiers; classifier model "
                "forward time is also reported separately. Not single-sample "
                "deployment latency."
            ),
        },
        "runtime": _runtime(),
        "limitations": [
            "RGB detector only; no depth or hardware sensor input.",
            "MMFC was trained on ground-truth crops and is evaluated here on predicted boxes.",
            "The fixed internal test split is not an external blind deployment set.",
        ],
        "images": serializable_images,
    }
    _write_artifacts(output_dir, report)
    return {
        "state": "complete",
        "output": output_dir.as_posix(),
        "source_images": len(images),
        "ground_truth_objects": report["dataset"]["ground_truth_object_count"],
        "scenarios": {
            threshold: {
                source: {
                    "end_to_end_recall": values["end_to_end_recall"],
                    "macro_f1": values["macro_f1"],
                }
                for source, values in sources.items()
            }
            for threshold, sources in scenarios.items()
        },
    }


def main() -> None:
    result = run_evaluation(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
