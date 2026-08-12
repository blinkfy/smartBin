"""Audit exact and high-confidence near-duplicate images across data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from mmfc.data import write_json
from mmfc.evaluation import compute_classification_metrics
from mmfc.training import PROJECT_ROOT, load_config, read_jsonl, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact and aligned near-duplicate source images across splits"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qwen-material/DATA_LEAKAGE_AUDIT.json"),
    )
    parser.add_argument("--phash-distance", type=int, default=10)
    parser.add_argument("--mae-threshold", type=float, default=0.012)
    parser.add_argument("--correlation-threshold", type=float, default=0.99)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional prediction JSONL for exclusion sensitivity; repeatable",
    )
    return parser.parse_args()


def _decode_grayscale(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def perceptual_hash(image: np.ndarray) -> int:
    """Return a deterministic 64-bit DCT perceptual hash."""

    if image.ndim != 2 or image.size == 0:
        raise ValueError("perceptual_hash expects a non-empty grayscale image")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    low_frequency = cv2.dct(resized)[:8, :8].reshape(-1)
    threshold = float(np.median(low_frequency[1:]))
    result = 0
    for value in low_frequency:
        result = (result << 1) | int(value > threshold)
    return result


def aligned_image_similarity(
    left: np.ndarray,
    right: np.ndarray,
    *,
    size: int = 128,
) -> tuple[float, float]:
    """Return normalized grayscale MAE and aligned Pearson correlation."""

    if left.ndim != 2 or right.ndim != 2 or left.size == 0 or right.size == 0:
        raise ValueError("aligned similarity expects non-empty grayscale images")
    left_compact = cv2.resize(
        left, (size, size), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    right_compact = cv2.resize(
        right, (size, size), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    left_compact /= 255.0
    right_compact /= 255.0
    mae = float(np.abs(left_compact - right_compact).mean())
    left_centered = left_compact - left_compact.mean()
    right_centered = right_compact - right_compact.mean()
    denominator = math.sqrt(
        float(np.square(left_centered).sum())
        * float(np.square(right_centered).sum())
    )
    correlation = (
        float((left_centered * right_centered).sum() / denominator)
        if denominator > 1e-12
        else 0.0
    )
    return mae, correlation


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def audit_manifests(
    manifests: Mapping[str, Path],
    *,
    phash_distance: int = 10,
    mae_threshold: float = 0.012,
    correlation_threshold: float = 0.99,
) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    if phash_distance < 0 or phash_distance > 64:
        raise ValueError("phash_distance must be in [0,64]")
    if mae_threshold < 0.0:
        raise ValueError("mae_threshold must be non-negative")
    if not -1.0 <= correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in [-1,1]")

    sources: dict[str, dict[str, Any]] = {}
    sample_to_source: dict[str, str] = {}
    object_identity_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    split_sample_counts: Counter[str] = Counter()
    for split, manifest in manifests.items():
        for record in read_jsonl(manifest):
            source_id = str(record["source_image_id"])
            sample_id = str(record["sample_id"])
            image_path = resolve_project_path(record["image_path"])
            split_sample_counts[str(split)] += 1
            existing = sources.get(source_id)
            if existing is None:
                sources[source_id] = {
                    "split": str(split),
                    "image_path": image_path,
                    "labels": {str(record["label"])},
                }
            else:
                if existing["split"] != str(split):
                    raise ValueError(
                        f"source_image_id crosses splits: {source_id}"
                    )
                if existing["image_path"] != image_path:
                    raise ValueError(
                        f"source_image_id maps to multiple image paths: {source_id}"
                    )
                existing["labels"].add(str(record["label"]))
            previous_source = sample_to_source.setdefault(sample_id, source_id)
            if previous_source != source_id:
                raise ValueError(f"sample_id maps to multiple sources: {sample_id}")
            bbox = record.get("bbox")
            if isinstance(bbox, Sequence) and len(bbox) == 4:
                object_key = (
                    "crop",
                    source_id,
                    str(record["label"]),
                    *(round(float(value), 8) for value in bbox),
                )
            elif "annotation_file" in record and "shape_index" in record:
                object_key = (
                    "annotation",
                    source_id,
                    str(record["annotation_file"]),
                    int(record["shape_index"]),
                    str(record["label"]),
                )
            else:
                object_key = ()
            if object_key:
                object_identity_groups[object_key].append(
                    {
                        "sample_id": sample_id,
                        "source_image_id": source_id,
                        "split": str(split),
                        "annotation_file": str(record["annotation_file"]),
                        "shape_index": int(record["shape_index"]),
                        "label": str(record["label"]),
                        "bbox": record.get("bbox"),
                    }
                )

    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    features: dict[str, dict[str, Any]] = {}
    for source_id, source in sources.items():
        path = Path(source["image_path"])
        payload = path.read_bytes()
        exact_groups[hashlib.sha256(payload).hexdigest()].append(
            {
                "source_image_id": source_id,
                "split": source["split"],
                "image_path": _relative_path(path),
                "labels": sorted(source["labels"]),
            }
        )
        grayscale = _decode_grayscale(path)
        features[source_id] = {
            **source,
            "phash": perceptual_hash(grayscale),
            "compact": cv2.resize(
                grayscale, (128, 128), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0,
        }

    cross_split_exact = [
        {"sha256": digest, "records": records}
        for digest, records in sorted(exact_groups.items())
        if len({record["split"] for record in records}) > 1
    ]
    split_names = list(manifests)
    split_features = {
        split: [
            (source_id, feature)
            for source_id, feature in features.items()
            if feature["split"] == split
        ]
        for split in split_names
    }
    distance_counts: Counter[int] = Counter()
    high_confidence_pairs: list[dict[str, Any]] = []
    for left_index, left_split in enumerate(split_names):
        for right_split in split_names[left_index + 1 :]:
            for left_id, left in split_features[left_split]:
                left_compact = left["compact"]
                left_centered = left_compact - left_compact.mean()
                left_energy = float(np.square(left_centered).sum())
                for right_id, right in split_features[right_split]:
                    distance = bin(left["phash"] ^ right["phash"]).count("1")
                    if distance > phash_distance:
                        continue
                    distance_counts[distance] += 1
                    right_compact = right["compact"]
                    mae = float(np.abs(left_compact - right_compact).mean())
                    right_centered = right_compact - right_compact.mean()
                    denominator = math.sqrt(
                        left_energy * float(np.square(right_centered).sum())
                    )
                    correlation = (
                        float(
                            (left_centered * right_centered).sum() / denominator
                        )
                        if denominator > 1e-12
                        else 0.0
                    )
                    if (
                        mae > mae_threshold
                        or correlation < correlation_threshold
                    ):
                        continue
                    high_confidence_pairs.append(
                        {
                            "phash_distance": distance,
                            "normalized_mae": mae,
                            "grayscale_correlation": correlation,
                            "left": {
                                "source_image_id": left_id,
                                "split": left_split,
                                "image_path": _relative_path(left["image_path"]),
                                "labels": sorted(left["labels"]),
                            },
                            "right": {
                                "source_image_id": right_id,
                                "split": right_split,
                                "image_path": _relative_path(right["image_path"]),
                                "labels": sorted(right["labels"]),
                            },
                        }
                    )

    high_confidence_pairs.sort(
        key=lambda pair: (
            -pair["grayscale_correlation"],
            pair["normalized_mae"],
            pair["left"]["source_image_id"],
            pair["right"]["source_image_id"],
        )
    )
    flagged_sources = {
        record["source_image_id"]
        for pair in high_confidence_pairs
        for record in (pair["left"], pair["right"])
    }
    duplicate_object_groups: list[dict[str, Any]] = []
    duplicate_object_counts_by_split: Counter[str] = Counter()
    for records in object_identity_groups.values():
        if len(records) < 2:
            continue
        records.sort(key=lambda record: record["sample_id"])
        duplicate_count = len(records) - 1
        split = records[0]["split"]
        duplicate_object_counts_by_split[split] += duplicate_count
        duplicate_object_groups.append(
            {
                "canonical_sample_id": records[0]["sample_id"],
                "duplicate_sample_ids": [
                    record["sample_id"] for record in records[1:]
                ],
                "record_count": len(records),
                "removed_record_count": duplicate_count,
                "source_image_id": records[0]["source_image_id"],
                "split": split,
                "annotation_file": records[0]["annotation_file"],
                "annotation_files": [
                    record["annotation_file"] for record in records
                ],
                "shape_index": records[0]["shape_index"],
                "shape_indices": [
                    record["shape_index"] for record in records
                ],
                "label": records[0]["label"],
                "bbox": records[0]["bbox"],
            }
        )
    duplicate_object_groups.sort(
        key=lambda group: (
            group["split"],
            group["source_image_id"],
            group["shape_index"],
            group["canonical_sample_id"],
        )
    )
    duplicate_object_record_count = sum(
        group["removed_record_count"] for group in duplicate_object_groups
    )
    report = {
        "method": {
            "exact": "SHA-256 file digest",
            "candidate_filter": "64-bit DCT perceptual hash",
            "confirmation": "aligned 128x128 grayscale MAE and Pearson correlation",
            "phash_distance_max": phash_distance,
            "normalized_mae_max": mae_threshold,
            "grayscale_correlation_min": correlation_threshold,
            "limitations": (
                "The aligned detector is conservative: it can miss moved or "
                "reframed captures and can flag a shared static background."
            ),
        },
        "source_image_count": len(sources),
        "sample_count": int(sum(split_sample_counts.values())),
        "unique_annotated_object_count": (
            int(sum(split_sample_counts.values()))
            - duplicate_object_record_count
        ),
        "split_source_counts": {
            split: len(split_features[split]) for split in split_names
        },
        "split_sample_counts": {
            split: int(split_sample_counts[split]) for split in split_names
        },
        "exact_object_duplicate_group_count": len(
            duplicate_object_groups
        ),
        "exact_object_duplicate_record_count": duplicate_object_record_count,
        "exact_object_duplicate_record_counts_by_split": {
            split: int(duplicate_object_counts_by_split[split])
            for split in split_names
        },
        "exact_object_duplicate_groups": duplicate_object_groups,
        "cross_split_exact_duplicate_group_count": len(cross_split_exact),
        "cross_split_exact_duplicate_groups": cross_split_exact,
        "phash_candidate_pair_count": int(sum(distance_counts.values())),
        "phash_candidate_distance_counts": {
            str(distance): int(distance_counts[distance])
            for distance in sorted(distance_counts)
        },
        "high_confidence_pair_count": len(high_confidence_pairs),
        "high_confidence_source_count": len(flagged_sources),
        "high_confidence_source_counts_by_split": {
            split: sum(
                source_id in flagged_sources
                for source_id, _ in split_features[split]
            )
            for split in split_names
        },
        "high_confidence_pairs": high_confidence_pairs,
    }
    return report, sample_to_source, flagged_sources


def _prediction_metrics(
    rows: Sequence[dict[str, Any]],
    class_names: Sequence[str],
) -> dict[str, Any]:
    targets = np.asarray([row["target_id"] for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [row["probabilities"] for row in rows], dtype=np.float64
    )
    result = compute_classification_metrics(
        targets, probabilities, class_names
    )["metrics"]
    return {
        name: result[name]
        for name in (
            "sample_count",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "negative_log_likelihood",
            "multiclass_brier_score",
            "ece",
        )
    }


def prediction_exclusion_sensitivity(
    records: Sequence[dict[str, Any]],
    *,
    sample_to_source: Mapping[str, str],
    excluded_sources: set[str],
    class_names: Sequence[str],
) -> dict[str, Any]:
    retained = []
    excluded = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in sample_to_source:
            raise ValueError(f"prediction sample is absent from manifests: {sample_id}")
        target = (
            excluded
            if sample_to_source[sample_id] in excluded_sources
            else retained
        )
        target.append(record)
    if not retained:
        raise ValueError("near-duplicate exclusion removed all prediction records")
    return {
        "all": _prediction_metrics(records, class_names),
        "after_high_confidence_exclusion": _prediction_metrics(
            retained, class_names
        ),
        "excluded_sample_count": len(excluded),
        "excluded_source_count": len(
            {
                sample_to_source[str(record["sample_id"])]
                for record in excluded
            }
        ),
    }


def prediction_duplicate_exclusion_sensitivity(
    records: Sequence[dict[str, Any]],
    *,
    excluded_sample_ids: set[str],
    class_names: Sequence[str],
) -> dict[str, Any]:
    retained = [
        record
        for record in records
        if str(record["sample_id"]) not in excluded_sample_ids
    ]
    excluded = [
        record
        for record in records
        if str(record["sample_id"]) in excluded_sample_ids
    ]
    if not retained:
        raise ValueError("exact-object duplicate exclusion removed all records")
    missing = excluded_sample_ids - {
        str(record["sample_id"]) for record in excluded
    }
    if missing:
        raise ValueError(
            "prediction file is missing duplicate sample ids: "
            + ", ".join(sorted(missing))
        )
    return {
        "all": _prediction_metrics(records, class_names),
        "after_exact_object_duplicate_exclusion": _prediction_metrics(
            retained, class_names
        ),
        "excluded_sample_count": len(excluded),
        "canonicalization": (
            "Within each identical annotated-object group, retain the "
            "lexicographically smallest sample_id."
        ),
    }


def _parse_prediction_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError("--prediction must use NAME=PATH")
        key = name.strip()
        if key in result:
            raise ValueError(f"duplicate prediction name: {key}")
        result[key] = resolve_project_path(path.strip())
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config.get("data", {})
    manifests = {
        split: resolve_project_path(data_config[f"{split}_manifest"])
        for split in ("train", "val", "test")
    }
    report, sample_to_source, flagged_sources = audit_manifests(
        manifests,
        phash_distance=args.phash_distance,
        mae_threshold=args.mae_threshold,
        correlation_threshold=args.correlation_threshold,
    )
    class_names = [
        item["name"] if isinstance(item, dict) else str(item)
        for item in config["classes"]
    ]
    prediction_records = {
        name: read_jsonl(path)
        for name, path in _parse_prediction_specs(args.prediction).items()
    }
    report["prediction_sensitivity"] = {
        name: prediction_exclusion_sensitivity(
            records,
            sample_to_source=sample_to_source,
            excluded_sources=flagged_sources,
            class_names=class_names,
        )
        for name, records in prediction_records.items()
    }
    duplicate_sample_ids = {
        sample_id
        for group in report["exact_object_duplicate_groups"]
        if group["split"] == "test"
        for sample_id in group["duplicate_sample_ids"]
    }
    report["exact_object_duplicate_prediction_sensitivity"] = {
        name: prediction_duplicate_exclusion_sensitivity(
            records,
            excluded_sample_ids=duplicate_sample_ids,
            class_names=class_names,
        )
        for name, records in prediction_records.items()
    }
    output = resolve_project_path(args.output)
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_image_count": report["source_image_count"],
                "cross_split_exact_duplicate_group_count": report[
                    "cross_split_exact_duplicate_group_count"
                ],
                "high_confidence_pair_count": report[
                    "high_confidence_pair_count"
                ],
                "high_confidence_source_count": report[
                    "high_confidence_source_count"
                ],
                "exact_object_duplicate_record_count": report[
                    "exact_object_duplicate_record_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
