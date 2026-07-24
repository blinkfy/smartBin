#!/usr/bin/env python
"""Validate generated manifests, sensor schemas, paths, ranges and leakage."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from build_dataset import DEFAULT_CONFIG_PATH, load_canonical_labels
from mmfc.data import (
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    SPLITS,
    read_jsonl,
    resolve_manifest_path,
)
from mmfc.sensors import PROPERTY_RANGES, contains_leakage_key


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_grid(
    value: object, *, normalized: bool, location: str, errors: list[str]
) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        errors.append(f"{location}: 必须是 4x4 数组")
        return
    for row_index, row in enumerate(value):
        for column_index, item in enumerate(row):
            if not _is_number(item):
                errors.append(
                    f"{location}[{row_index}][{column_index}]: 必须为有限数字"
                )
            elif normalized and not 0.0 <= float(item) <= 1.0:
                errors.append(
                    f"{location}[{row_index}][{column_index}]: 必须在 [0,1]"
                )


def validate_generated(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    project_root: Path | str = PROJECT_ROOT,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    project_root = Path(project_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    samples: dict[str, dict[str, Any]] = {}
    source_splits: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    try:
        canonical_labels = load_canonical_labels(
            config_path, project_root=project_root
        )
        label_to_index = {
            label: index for index, label in enumerate(canonical_labels)
        }
    except ValueError as exc:
        errors.append(str(exc))
        canonical_labels = []
        label_to_index = {}

    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"缺少文件：{path.name}")
            continue
        try:
            records = read_jsonl(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        for line_number, sample in enumerate(records, 1):
            location = f"{path.name}:{line_number}"
            required = {
                "schema_version",
                "sample_id",
                "source_image_id",
                "image_path",
                "image_width",
                "image_height",
                "bbox",
                "label",
                "label_id",
                "split",
            }
            missing = required - sample.keys()
            if missing:
                errors.append(f"{location}: 缺少字段 {sorted(missing)}")
                continue
            if sample["schema_version"] != 1:
                errors.append(f"{location}: 不支持的 schema_version")
            sample_id = sample["sample_id"]
            if not isinstance(sample_id, str) or not sample_id:
                errors.append(f"{location}: sample_id 非法")
                continue
            if sample_id in samples:
                errors.append(f"{location}: 重复 sample_id {sample_id}")
            samples[sample_id] = sample
            if sample["split"] != split:
                errors.append(f"{location}: split 字段与文件名不符")
            source_id = sample["source_image_id"]
            if not isinstance(source_id, str) or not source_id:
                errors.append(f"{location}: source_image_id 非法")
            else:
                previous = source_splits.setdefault(source_id, split)
                if previous != split:
                    errors.append(
                        f"{location}: source_image_id {source_id} 跨集合泄漏 "
                        f"({previous}/{split})"
                    )
            if not isinstance(sample["image_path"], str) or not sample["image_path"]:
                errors.append(f"{location}: image_path 非法")
            else:
                image_path = resolve_manifest_path(sample["image_path"], project_root)
                if not image_path.is_file():
                    errors.append(f"{location}: 原图不存在 {image_path}")
            label = sample["label"]
            if not isinstance(label, str) or not label.strip():
                errors.append(f"{location}: label 非法")
            elif label not in label_to_index:
                errors.append(f"{location}: 未知 label {label!r}")
            else:
                label_id = sample["label_id"]
                expected_label_id = label_to_index[label]
                if (
                    not isinstance(label_id, int)
                    or isinstance(label_id, bool)
                    or label_id != expected_label_id
                ):
                    errors.append(
                        f"{location}: label_id 应为 {expected_label_id}，"
                        f"实际为 {label_id!r}"
                    )
            bbox = sample["bbox"]
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(not _is_number(value) for value in bbox)
            ):
                errors.append(f"{location}: bbox 必须是 4 个有限数字")
            else:
                x1, y1, x2, y2 = (float(value) for value in bbox)
                width, height = sample["image_width"], sample["image_height"]
                if not (
                    _is_number(width)
                    and _is_number(height)
                    and float(width) > 0
                    and float(height) > 0
                ):
                    errors.append(f"{location}: 图像尺寸非法")
                elif not (
                    0 <= x1 < x2 <= float(width)
                    and 0 <= y1 < y2 <= float(height)
                ):
                    errors.append(f"{location}: bbox 超出图像范围")
            split_counts[split] += 1

    sensors_path = output_dir / "sensors.jsonl"
    sensors: dict[str, dict[str, Any]] = {}
    sensor_positions: dict[str, int] = {}
    if not sensors_path.is_file():
        errors.append("缺少文件：sensors.jsonl")
    else:
        try:
            sensor_records = read_jsonl(sensors_path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            sensor_records = []
        for line_number, sensor in enumerate(sensor_records, 1):
            location = f"sensors.jsonl:{line_number}"
            sample_id = sensor.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                errors.append(f"{location}: sample_id 非法")
                continue
            if sample_id in sensors:
                errors.append(f"{location}: 重复 sample_id {sample_id}")
            sensors[sample_id] = sensor
            sensor_positions.setdefault(sample_id, line_number - 1)
            if sensor.get("schema_version") != 1:
                errors.append(f"{location}: 不支持的 schema_version")
            if contains_leakage_key(sensor):
                errors.append(f"{location}: 传感器记录包含目标标签/类别字段")
            if sensor.get("grid_shape") != [4, 4]:
                errors.append(f"{location}: grid_shape 必须为 [4,4]")
            _validate_grid(
                sensor.get("grid"),
                normalized=True,
                location=f"{location}.grid",
                errors=errors,
            )
            _validate_grid(
                sensor.get("normalized_grid"),
                normalized=True,
                location=f"{location}.normalized_grid",
                errors=errors,
            )
            properties = sensor.get("properties")
            if not isinstance(properties, Mapping):
                errors.append(f"{location}: properties 必须是对象")
            else:
                for name, (lower, upper) in PROPERTY_RANGES.items():
                    value = properties.get(name)
                    if not _is_number(value):
                        errors.append(f"{location}: {name} 必须为有限数字")
                    elif not lower <= float(value) <= upper:
                        errors.append(
                            f"{location}: {name} 超出 [{lower},{upper}]"
                        )
            source = sensor.get("source")
            if (
                not isinstance(source, Mapping)
                or not isinstance(source.get("generator"), str)
                or not source.get("generator")
            ):
                errors.append(f"{location}: source.generator 非法")

    missing_sensors = sorted(set(samples) - set(sensors))
    orphan_sensors = sorted(set(sensors) - set(samples))
    if missing_sensors:
        errors.append(f"有 {len(missing_sensors)} 个样本缺少传感器记录")
    if orphan_sensors:
        errors.append(f"有 {len(orphan_sensors)} 条孤立传感器记录")

    index_path = output_dir / "index.json"
    if not index_path.is_file():
        errors.append("缺少文件：index.json")
    else:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(index, dict):
                raise AttributeError("根节点必须是对象")
            if index.get("schema_version") != 1:
                errors.append("index.json 的 schema_version 非法")
            indexed_labels = index.get("labels")
            if indexed_labels != canonical_labels:
                errors.append(
                    "index.json.labels 与配置类别顺序不一致"
                )
            indexed_label_map = index.get("label_to_index")
            if not isinstance(indexed_label_map, dict):
                errors.append("index.json.label_to_index 必须是对象")
            else:
                if indexed_label_map != label_to_index:
                    errors.append(
                        "index.json.label_to_index 与配置类别映射不一致"
                    )
                if list(indexed_label_map) != canonical_labels:
                    errors.append(
                        "index.json.label_to_index 的键顺序与配置不一致"
                    )
            indexed = index.get("samples", {})
            if not isinstance(indexed, dict) or set(indexed) != set(samples):
                errors.append("index.json 的样本集合与清单不一致")
            else:
                for sample_id, sample in samples.items():
                    location = f"index.json.samples[{sample_id!r}]"
                    entry = indexed[sample_id]
                    if not isinstance(entry, Mapping):
                        errors.append(f"{location}: 必须是对象")
                        continue
                    label = sample.get("label")
                    expected_label_id = label_to_index.get(label)
                    if (
                        expected_label_id is not None
                        and entry.get("label_index") != expected_label_id
                    ):
                        errors.append(
                            f"{location}.label_index 应为 "
                            f"{expected_label_id}，实际为 "
                            f"{entry.get('label_index')!r}"
                        )
                    if entry.get("split") != sample.get("split"):
                        errors.append(f"{location}.split 与清单不一致")
                    if entry.get("sensor_index") != sensor_positions.get(sample_id):
                        errors.append(
                            f"{location}.sensor_index 与 sensors.jsonl 不一致"
                        )
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"index.json 非法：{exc}")
    stats_path = output_dir / "stats.json"
    if not stats_path.is_file():
        warnings.append("缺少 stats.json")

    return {
        "valid": not errors,
        "sample_count": len(samples),
        "source_image_count": len(source_splits),
        "sensor_count": len(sensors),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="校验数据清单和传感器产物。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="定义 canonical 类别顺序的配置文件（相对 project-root 解析）。",
    )
    args = parser.parse_args()
    report = validate_generated(
        args.output,
        project_root=args.project_root,
        config_path=args.config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
