"""Evaluate scene-pipeline recall across preregistered target strata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_scene_pipeline import greedy_iou_match


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INTEGRATED_PROTOCOL_ADDENDUM = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "yolo-seg"
    / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
)
DEFAULT_STRATIFIED_PROTOCOL_ADDENDUM = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "yolo-seg"
    / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
)
SOURCES = ("detector", "missing", "rule-v1")
DIMENSION_ORDER = (
    "relative_bbox_area",
    "scene_density",
    "bbox_overlap_proxy",
)


def parse_args() -> argparse.Namespace:
    reports = PROJECT_ROOT / "reports" / "qwen-material"
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate preregistered relative-size and crowding strata from "
            "the frozen integrated scene predictions"
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "yolo-seg-unique-linked",
    )
    parser.add_argument(
        "--integrated-report",
        type=Path,
        default=reports / "yolo-seg" / "integrated" / "evaluation.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=reports / "yolo-seg" / "STRATIFIED_EVALUATION_PROTOCOL.json",
    )
    parser.add_argument(
        "--protocol-addendum",
        type=Path,
        default=DEFAULT_STRATIFIED_PROTOCOL_ADDENDUM,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=reports / "yolo-seg" / "integrated" / "stratified",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    resolved_root = root.resolve()
    for path in sorted(
        (item.resolve() for item in root.rglob("*.txt")),
        key=str,
    ):
        relative = path.relative_to(resolved_root).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        file_size = path.stat().st_size
        size_bytes += file_size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_count += 1
    return {
        "file_count": file_count,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def assign_size_stratum(
    area_ratio: float,
    *,
    lower_quartile: float,
    upper_quartile: float,
) -> str:
    if not 0.0 <= area_ratio <= 1.0:
        raise ValueError("normalized bbox area ratio must be in [0,1]")
    if not 0.0 < lower_quartile < upper_quartile < 1.0:
        raise ValueError("size cut points must be ordered inside (0,1)")
    if area_ratio < lower_quartile:
        return "relative_small"
    if area_ratio < upper_quartile:
        return "relative_medium"
    return "relative_large"


def bbox_overlap_flags(
    boxes: Sequence[Sequence[float]],
) -> list[bool]:
    flags = [False] * len(boxes)
    for left_index, left in enumerate(boxes):
        if len(left) != 4:
            raise ValueError("each box must contain four coordinates")
        lx1, ly1, lx2, ly2 = (float(value) for value in left)
        if lx2 < lx1 or ly2 < ly1:
            raise ValueError("box coordinates are not ordered")
        for right_index in range(left_index + 1, len(boxes)):
            right = boxes[right_index]
            if len(right) != 4:
                raise ValueError("each box must contain four coordinates")
            rx1, ry1, rx2, ry2 = (float(value) for value in right)
            if rx2 < rx1 or ry2 < ry1:
                raise ValueError("box coordinates are not ordered")
            intersection_width = max(
                0.0,
                min(lx2, rx2) - max(lx1, rx1),
            )
            intersection_height = max(
                0.0,
                min(ly2, ry2) - max(ly1, ry1),
            )
            if intersection_width * intersection_height > 0.0:
                flags[left_index] = True
                flags[right_index] = True
    return flags


def _normalized_targets(
    label_path: Path,
    *,
    lower_quartile: float,
    upper_quartile: float,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    boxes: list[list[float]] = []
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
        coordinates = np.asarray(
            [float(value) for value in tokens[1:]],
            dtype=np.float64,
        ).reshape(-1, 2)
        box = [
            float(coordinates[:, 0].min()),
            float(coordinates[:, 1].min()),
            float(coordinates[:, 0].max()),
            float(coordinates[:, 1].max()),
        ]
        boxes.append(box)
        area_ratio = (box[2] - box[0]) * (box[3] - box[1])
        targets.append(
            {
                "class_id": int(tokens[0]),
                "area_ratio": area_ratio,
                "size_stratum": assign_size_stratum(
                    area_ratio,
                    lower_quartile=lower_quartile,
                    upper_quartile=upper_quartile,
                ),
            }
        )
    overlap_flags = bbox_overlap_flags(boxes)
    density = "single_object" if len(targets) == 1 else "multi_object"
    for target, overlaps in zip(targets, overlap_flags):
        target["density_stratum"] = density
        target["overlap_stratum"] = (
            "bbox_overlap" if overlaps else "bbox_isolated"
        )
    return targets


def load_target_strata(
    dataset_dir: Path,
    *,
    lower_quartile: float,
    upper_quartile: float,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    manifest_path = dataset_dir / "manifests" / "test.jsonl"
    for record in _read_jsonl(manifest_path):
        source_image_id = str(record["source_image_id"])
        if source_image_id in result:
            raise ValueError(f"duplicate test source image: {source_image_id}")
        label_path = Path(str(record["label_path"]))
        if not label_path.is_absolute():
            label_path = dataset_dir / label_path
        result[source_image_id] = _normalized_targets(
            label_path,
            lower_quartile=lower_quartile,
            upper_quartile=upper_quartile,
        )
    return result


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


def _target_memberships(target: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return (
        ("relative_bbox_area", str(target["size_stratum"])),
        ("scene_density", str(target["density_stratum"])),
        ("bbox_overlap_proxy", str(target["overlap_stratum"])),
    )


def build_contributions(
    images: Sequence[Mapping[str, Any]],
    target_strata: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    confidence_thresholds: Sequence[float],
    match_iou: float,
    sources: Sequence[str] = SOURCES,
) -> tuple[
    list[str],
    dict[tuple[float, str, str, str], np.ndarray],
]:
    source_image_ids = [str(image["source_image_id"]) for image in images]
    if len(set(source_image_ids)) != len(source_image_ids):
        raise ValueError("integrated report contains duplicate source images")
    contributions: dict[
        tuple[float, str, str, str],
        np.ndarray,
    ] = {}
    for image_index, image in enumerate(images):
        source_image_id = source_image_ids[image_index]
        strata = target_strata.get(source_image_id)
        if strata is None:
            raise ValueError(
                f"strata metadata is missing for {source_image_id}"
            )
        ground_truth = list(image["ground_truth"])
        if len(ground_truth) != len(strata):
            raise ValueError(
                f"ground-truth count mismatch for {source_image_id}"
            )
        for target, metadata in zip(ground_truth, strata):
            if int(target["class_id"]) != int(metadata["class_id"]):
                raise ValueError(
                    f"ground-truth order mismatch for {source_image_id}"
                )

        for threshold in confidence_thresholds:
            detections = [
                detection
                for detection in image["detections"]
                if float(detection["detector_confidence"]) >= threshold
            ]
            matching = greedy_iou_match(
                ground_truth,
                detections,
                iou_threshold=match_iou,
            )
            detection_by_target = {
                int(match["ground_truth_index"]): detections[
                    int(match["detection_index"])
                ]
                for match in matching["matches"]
            }
            for target_index, (target, metadata) in enumerate(
                zip(ground_truth, strata)
            ):
                detection = detection_by_target.get(target_index)
                for source in sources:
                    localized = int(detection is not None)
                    correct = int(
                        detection is not None
                        and _prediction_id(detection, source)
                        == int(target["class_id"])
                    )
                    for dimension, stratum in _target_memberships(metadata):
                        key = (float(threshold), source, dimension, stratum)
                        if key not in contributions:
                            contributions[key] = np.zeros(
                                (len(images), 3),
                                dtype=np.int64,
                            )
                        contributions[key][image_index] += (
                            1,
                            localized,
                            correct,
                        )
    return source_image_ids, contributions


def _ratio_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    confidence: float,
) -> tuple[float | None, float | None]:
    valid = denominator > 0
    if not valid.any():
        return None, None
    ratios = numerator[valid] / denominator[valid]
    alpha = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(ratios, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def summarize_contributions(
    source_image_ids: Sequence[str],
    contributions: Mapping[
        tuple[float, str, str, str],
        np.ndarray,
    ],
    *,
    repeats: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be in (0,1)")
    image_count = len(source_image_ids)
    if image_count == 0:
        raise ValueError("source image list must not be empty")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        image_count,
        np.full(image_count, 1.0 / image_count),
        size=repeats,
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(
        contributions,
        key=lambda item: (
            item[0],
            SOURCES.index(item[1]),
            DIMENSION_ORDER.index(item[2]),
            item[3],
        ),
    ):
        threshold, source, dimension, stratum = key
        values = np.asarray(contributions[key], dtype=np.int64)
        if values.shape != (image_count, 3):
            raise ValueError(f"invalid contribution shape for {key}")
        point = values.sum(axis=0)
        sampled = weights @ values
        support = int(point[0])
        localized = int(point[1])
        correct = int(point[2])
        localization_interval = _ratio_interval(
            sampled[:, 1],
            sampled[:, 0],
            confidence=confidence,
        )
        classification_interval = _ratio_interval(
            sampled[:, 2],
            sampled[:, 1],
            confidence=confidence,
        )
        end_to_end_interval = _ratio_interval(
            sampled[:, 2],
            sampled[:, 0],
            confidence=confidence,
        )
        rows.append(
            {
                "confidence_threshold": threshold,
                "prediction_source": source,
                "dimension": dimension,
                "stratum": stratum,
                "source_image_count_with_support": int(
                    np.count_nonzero(values[:, 0])
                ),
                "support": support,
                "localized_count": localized,
                "correct_count": correct,
                "localization_recall": (
                    localized / support if support else None
                ),
                "localization_recall_ci_lower": localization_interval[0],
                "localization_recall_ci_upper": localization_interval[1],
                "classification_accuracy_on_localized": (
                    correct / localized if localized else None
                ),
                "classification_accuracy_ci_lower": (
                    classification_interval[0]
                ),
                "classification_accuracy_ci_upper": (
                    classification_interval[1]
                ),
                "end_to_end_recall": correct / support if support else None,
                "end_to_end_recall_ci_lower": end_to_end_interval[0],
                "end_to_end_recall_ci_upper": end_to_end_interval[1],
            }
        )
    return rows


def _validate_protocol_hashes(
    protocol: Mapping[str, Any],
    *,
    expected_hash_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for entry in protocol.get("input_hashes", []):
        path = _resolve(Path(str(entry["path"])))
        key = Path(str(entry["path"])).as_posix()
        expected = str(
            (expected_hash_overrides or {}).get(key, entry["sha256"])
        ).lower()
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"protocol input hash mismatch for {path}: "
                f"{actual} != {expected}"
            )
        checks.append(
            {
                "path": path.as_posix(),
                "sha256": actual,
            }
        )
    if not checks:
        raise ValueError("protocol does not contain input hashes")
    expected_tree = protocol.get("label_tree")
    if not isinstance(expected_tree, Mapping):
        raise ValueError("protocol does not contain a label-tree identity")
    label_root = _resolve(Path(str(expected_tree["path"])))
    actual_tree = _sha256_tree(label_root)
    for key in ("file_count", "size_bytes", "sha256"):
        if actual_tree[key] != expected_tree[key]:
            raise ValueError(
                f"protocol label-tree {key} mismatch: "
                f"{actual_tree[key]} != {expected_tree[key]}"
            )
    checks.append(
        {
            "path": label_root.as_posix(),
            "sha256": str(actual_tree["sha256"]),
        }
    )
    return checks


def _input_hash(protocol: Mapping[str, Any], relative_path: str) -> str:
    for entry in protocol.get("input_hashes", []):
        if Path(str(entry["path"])).as_posix() == relative_path:
            return str(entry["sha256"]).lower()
    raise ValueError(f"protocol input hash is missing: {relative_path}")


def _validate_protocol_addenda(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_addendum_path: Path,
    integrated: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    integrated_addendum_path = DEFAULT_INTEGRATED_PROTOCOL_ADDENDUM.resolve()
    integrated_addendum = _read_json(integrated_addendum_path)
    protocol_addendum = _read_json(protocol_addendum_path)
    integrated_addendum_hash = _sha256(integrated_addendum_path)
    protocol_addendum_hash = _sha256(protocol_addendum_path)

    if (
        integrated_addendum.get("status")
        != "frozen_after_path_identity_failure_before_recovery_rerun"
        or integrated_addendum.get("valid") is not True
    ):
        raise ValueError("integrated protocol addendum is not valid")
    if (
        protocol_addendum.get("status")
        != "frozen_after_integrated_path_addendum_before_stratified_run"
        or protocol_addendum.get("valid") is not True
    ):
        raise ValueError("stratified protocol addendum is not valid")
    if (
        protocol_addendum.get("original_protocol", {}).get("sha256")
        != _sha256(protocol_path)
        or protocol_addendum.get("dependency_protocol_addendum", {}).get(
            "sha256"
        )
        != integrated_addendum_hash
    ):
        raise ValueError("stratified protocol addendum bindings mismatch")

    fix = protocol_addendum.get("implementation_fix", {})
    original_strata_hash = str(
        fix.get("original_strata_source", {}).get("sha256", "")
    ).lower()
    revised_strata_hash = str(
        fix.get("revised_strata_source", {}).get("sha256", "")
    ).lower()
    original_integrated_hash = str(
        fix.get("original_integrated_source", {}).get("sha256", "")
    ).lower()
    revised_integrated_hash = str(
        fix.get("revised_integrated_source", {}).get("sha256", "")
    ).lower()
    integrated_fix = integrated_addendum.get("implementation_fix", {})
    if (
        original_strata_hash
        != _input_hash(protocol, "evaluate_scene_strata.py")
        or original_integrated_hash
        != _input_hash(protocol, "evaluate_scene_pipeline.py")
        or original_integrated_hash
        != str(
            integrated_fix.get("original_evaluation_source", {}).get(
                "sha256", ""
            )
        ).lower()
        or revised_integrated_hash
        != str(
            integrated_fix.get("revised_evaluation_source", {}).get(
                "sha256", ""
            )
        ).lower()
        or revised_strata_hash != _sha256(Path(__file__).resolve())
        or revised_integrated_hash
        != _sha256((PROJECT_ROOT / "evaluate_scene_pipeline.py").resolve())
    ):
        raise ValueError("protocol addendum source hashes mismatch")

    integrated_binding = (
        integrated.get("frozen_protocol", {}).get("addendum", {})
    )
    if (
        integrated_binding.get("sha256") != integrated_addendum_hash
        or integrated_binding.get("evaluation_source_sha256")
        != revised_integrated_hash
    ):
        raise ValueError("integrated report does not bind its protocol addendum")

    return (
        integrated_addendum_path,
        integrated_addendum,
        {
            "path": protocol_addendum_path.as_posix(),
            "sha256": protocol_addendum_hash,
            "strata_source_sha256": revised_strata_hash,
        },
    )


def _write_artifacts(
    output_dir: Path,
    report: Mapping[str, Any],
) -> None:
    _atomic_write(
        output_dir / "evaluation.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    rows = list(report["rows"])
    csv_path = output_dir / "metrics.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)

    lines = [
        "# 整图分层辅助评估",
        "",
        "该分析在 detector 训练前冻结分层定义，只复用整图评估的同一批预测。",
        "",
        "| Conf | 类别来源 | 维度 | 分层 | N | 定位 Recall [95% CI] | "
        "匹配分类 Acc [95% CI] | 端到端 Recall [95% CI] |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]

    def interval(row: Mapping[str, Any], metric: str) -> str:
        value = row[metric]
        lower = row[f"{metric}_ci_lower"]
        upper = row[f"{metric}_ci_upper"]
        if value is None or lower is None or upper is None:
            return "NA"
        return (
            f"{100.0 * float(value):.2f}% "
            f"[{100.0 * float(lower):.2f}%,"
            f"{100.0 * float(upper):.2f}%]"
        )

    for row in rows:
        lines.append(
            f"| {float(row['confidence_threshold']):.2f} | "
            f"{row['prediction_source']} | {row['dimension']} | "
            f"{row['stratum']} | {int(row['support'])} | "
            f"{interval(row, 'localization_recall')} | "
            f"{interval(row, 'classification_accuracy_on_localized')} | "
            f"{interval(row, 'end_to_end_recall')} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `relative_small/medium/large` 是固定 test 标注框面积的四分位分层，"
            "不是 COCO 小目标定义；",
            "- `bbox_overlap` 仅表示标注框有正面积交叠，是拥挤/潜在遮挡代理，"
            "不是真实遮挡标签；",
            "- 单次 detector 检查点和固定内部 test 不支持外部部署外推；",
            "- 三种类别来源共享定位结果，定位 Recall 重复列出仅便于同表阅读。",
            "",
        ]
    )
    _atomic_write(output_dir / "REPORT.md", "\n".join(lines))


def run_evaluation(
    *,
    dataset_dir: Path,
    integrated_report_path: Path,
    protocol_path: Path,
    protocol_addendum_path: Path = DEFAULT_STRATIFIED_PROTOCOL_ADDENDUM,
    output_dir: Path,
) -> dict[str, Any]:
    dataset_dir = _resolve(dataset_dir)
    integrated_report_path = _resolve(integrated_report_path)
    protocol_path = _resolve(protocol_path)
    protocol_addendum_path = _resolve(protocol_addendum_path)
    output_dir = _resolve(output_dir)
    protocol = _read_json(protocol_path)
    if protocol.get("status") != "frozen_before_detector_training":
        raise ValueError("stratified evaluation protocol is not frozen")
    integrated = _read_json(integrated_report_path)
    if integrated.get("protocol") != "fixed_source_grouped_scene_test":
        raise ValueError("input is not the fixed integrated scene evaluation")
    (
        integrated_addendum_path,
        integrated_addendum,
        stratified_addendum_binding,
    ) = _validate_protocol_addenda(
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_addendum_path=protocol_addendum_path,
        integrated=integrated,
    )
    integrated_source_hash = str(
        integrated_addendum.get("implementation_fix", {})
        .get("revised_evaluation_source", {})
        .get("sha256", "")
    )
    stratified_source_hash = str(
        stratified_addendum_binding["strata_source_sha256"]
    )
    hash_checks = _validate_protocol_hashes(
        protocol,
        expected_hash_overrides={
            "evaluate_scene_pipeline.py": integrated_source_hash,
            "evaluate_scene_strata.py": stratified_source_hash,
        },
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(integrated["images"])
    dataset = integrated["dataset"]
    if (
        int(dataset["source_image_count"]) != 570
        or int(dataset["ground_truth_object_count"]) != 825
    ):
        raise ValueError("integrated test-set counts do not match protocol")

    size_definition = protocol["strata"]["relative_bbox_area"]
    lower_quartile = float(size_definition["lower_quartile"])
    upper_quartile = float(size_definition["upper_quartile"])
    target_strata = load_target_strata(
        dataset_dir,
        lower_quartile=lower_quartile,
        upper_quartile=upper_quartile,
    )
    confidence_thresholds = [
        float(value)
        for value in protocol["evaluation"]["confidence_thresholds"]
    ]
    match_iou = float(protocol["evaluation"]["match_iou"])
    source_image_ids, contributions = build_contributions(
        images,
        target_strata,
        confidence_thresholds=confidence_thresholds,
        match_iou=match_iou,
    )
    bootstrap = protocol["bootstrap"]
    rows = summarize_contributions(
        source_image_ids,
        contributions,
        repeats=int(bootstrap["repeats"]),
        confidence=float(bootstrap["confidence"]),
        seed=int(bootstrap["seed"]),
    )
    report = {
        "schema_version": 1,
        "protocol": "preregistered_scene_stratified_analysis",
        "frozen_protocol": {
            "path": protocol_path.as_posix(),
            "sha256": _sha256(protocol_path),
            "input_hash_checks": hash_checks,
            "integrated_protocol_addendum": {
                "path": integrated_addendum_path.as_posix(),
                "sha256": _sha256(integrated_addendum_path),
            },
            "stratified_protocol_addendum": stratified_addendum_binding,
        },
        "source_integrated_report": {
            "path": integrated_report_path.as_posix(),
            "sha256": _sha256(integrated_report_path),
            "detector_checkpoint": integrated["detector"]["checkpoint"],
            "detector_checkpoint_sha256": integrated["detector"]["sha256"],
        },
        "dataset": {
            "source_image_count": len(source_image_ids),
            "ground_truth_object_count": sum(
                len(values) for values in target_strata.values()
            ),
        },
        "strata": protocol["strata"],
        "evaluation": protocol["evaluation"],
        "bootstrap": protocol["bootstrap"],
        "rows": rows,
        "limitations": protocol["limitations"],
    }
    _write_artifacts(output_dir, report)
    return {
        "state": "complete",
        "output": output_dir.as_posix(),
        "row_count": len(rows),
    }


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        dataset_dir=args.dataset,
        integrated_report_path=args.integrated_report,
        protocol_path=args.protocol,
        protocol_addendum_path=args.protocol_addendum,
        output_dir=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
