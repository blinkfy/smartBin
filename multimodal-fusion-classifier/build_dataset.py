#!/usr/bin/env python
"""Build object manifests and simulated 4x4 sensor records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from mmfc.data import (
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_IMAGES_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    SPLITS,
    parse_labelme_annotations,
    read_jsonl,
    resolve_manifest_path,
    split_samples,
    write_json,
    write_jsonl,
)
from mmfc.sensors import (
    PROPERTY_RANGES,
    QwenSensorEnhancer,
    contains_leakage_key,
    qwen_api_key,
    simulate_rule_sensor_from_path,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 LabelMe 标注构建对象清单与可复现 4x4 传感器数据。"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "定义 canonical 类别顺序的配置文件。相对路径固定从项目根目录解析，"
            "不依赖当前工作目录。"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument(
        "--sensor-mode", choices=("rule", "qwen"), default="rule"
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Qwen API Key 环境变量名（不会写入产物）。",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--model", default="qwen3-vl-flash")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用 output/sensors.jsonl 中已有且属于当前清单的记录。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="仅构建确定性排序后的前 N 个对象，便于冒烟测试。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析并打印摘要，不写文件、不开启网络请求。",
    )
    return parser


def _resolve_project_path(
    path: Path | str, *, project_root: Path | str = PROJECT_ROOT
) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root).resolve() / path).resolve()


def load_canonical_labels(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    project_root: Path | str = PROJECT_ROOT,
) -> list[str]:
    """Load the unique class names whose order defines every numeric label."""
    path = _resolve_project_path(config_path, project_root=project_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"无法读取类别配置 {path}：{exc}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"类别配置不是有效 UTF-8 JSON {path}：{exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"类别配置根节点必须是对象：{path}")
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"类别配置 classes 必须是非空数组：{path}")

    labels: list[str] = []
    for index, item in enumerate(classes):
        if not isinstance(item, Mapping):
            raise ValueError(f"类别配置 classes[{index}] 必须是对象")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"类别配置 classes[{index}].name 必须是非空字符串")
        labels.append(name.strip())
    duplicates = sorted(
        label for label, count in Counter(labels).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"类别配置包含重复名称：{duplicates}")
    return labels


def _write_ordered_json(path: Path, value: object) -> None:
    """Atomically write JSON while preserving canonical mapping insertion order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2))
        handle.write("\n")
    temporary.replace(path)


def _load_resume(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    try:
        for record in read_jsonl(path):
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                records[sample_id] = record
    except (OSError, ValueError) as exc:
        print(f"警告：无法复用旧传感器文件：{exc}", file=sys.stderr)
        return {}
    return records


def _valid_4x4(value: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        return False
    return all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        and 0.0 <= float(item) <= 1.0
        for row in value
        for item in row
    )


def resume_record_usable(
    record: Mapping[str, Any],
    *,
    sensor_mode: str,
    qwen_model: str,
) -> bool:
    """Return whether a prior sensor record satisfies the requested mode.

    Qwen resume intentionally rejects rule and fallback records so a later run
    with credentials can upgrade an existing complete rule dataset.
    """
    if contains_leakage_key(record):
        return False
    if record.get("schema_version") != 1:
        return False
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        return False
    if record.get("grid_shape") != [4, 4]:
        return False
    if not _valid_4x4(record.get("grid")) or not _valid_4x4(
        record.get("normalized_grid")
    ):
        return False
    properties = record.get("properties")
    if not isinstance(properties, Mapping):
        return False
    for name, (lower, upper) in PROPERTY_RANGES.items():
        value = properties.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not lower <= float(value) <= upper
        ):
            return False
    source = record.get("source")
    if not isinstance(source, Mapping):
        return False
    generator = source.get("generator")
    if sensor_mode == "rule":
        return generator == "rule-v1"
    if sensor_mode == "qwen":
        return (
            generator == qwen_model
            and source.get("enhancement_status") == "success"
        )
    return False


def build(args: argparse.Namespace) -> dict[str, Any]:
    canonical_labels = load_canonical_labels(
        getattr(args, "config", DEFAULT_CONFIG_PATH)
    )
    label_to_index = {
        label: index for index, label in enumerate(canonical_labels)
    }
    samples, report = parse_labelme_annotations(
        args.annotations, args.images, project_root=PROJECT_ROOT
    )
    unknown_labels = sorted(
        {sample["label"] for sample in samples} - set(label_to_index)
    )
    if unknown_labels:
        raise ValueError(
            "标注包含类别配置中不存在的标签："
            f"{unknown_labels}；请先更新配置或修正标注"
        )
    for sample in samples:
        sample["label_id"] = label_to_index[sample["label"]]
    samples.sort(key=lambda item: item["sample_id"])
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit 不能为负数")
        samples = samples[: args.limit]
    split_records = split_samples(samples, seed=args.seed, ratios=args.ratios)
    all_samples = [
        sample for split in SPLITS for sample in split_records[split]
    ]
    summary = {
        "samples": len(all_samples),
        "source_images": len({item["source_image_id"] for item in all_samples}),
        "split_counts": {split: len(split_records[split]) for split in SPLITS},
        "class_counts": dict(
            sorted(Counter(item["label"] for item in all_samples).items())
        ),
        "labels": canonical_labels,
        "issue_counts": report["issue_counts"],
        "sensor_mode": args.sensor_mode,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return summary

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(output / f"{split}.jsonl", split_records[split])

    resume_records = (
        _load_resume(output / "sensors.jsonl") if args.resume else {}
    )
    api_key = qwen_api_key(args.api_key_env) if args.sensor_mode == "qwen" else None
    enhancer = QwenSensorEnhancer(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        retries=args.retries,
        timeout=args.timeout,
    )
    sensors: list[dict[str, Any]] = []
    sensor_issues: list[dict[str, Any]] = []
    reused = 0
    rejected_resume_records = 0
    for number, sample in enumerate(
        sorted(all_samples, key=lambda item: item["sample_id"]), 1
    ):
        old_record = resume_records.get(sample["sample_id"])
        if old_record is not None and resume_record_usable(
            old_record,
            sensor_mode=args.sensor_mode,
            qwen_model=args.model,
        ):
            sensors.append(old_record)
            reused += 1
            continue
        if old_record is not None:
            rejected_resume_records += 1
        image_path = resolve_manifest_path(sample["image_path"], PROJECT_ROOT)
        try:
            rule_record = simulate_rule_sensor_from_path(
                image_path,
                sample["bbox"],
                sample["sample_id"],
                seed=args.seed,
                source_image_id=sample["source_image_id"],
            )
            if args.sensor_mode == "qwen":
                with Image.open(image_path) as image:
                    sensor_record = enhancer.enhance(
                        image, sample["bbox"], rule_record
                    )
            else:
                sensor_record = rule_record
            sensors.append(sensor_record)
        except Exception as exc:
            sensor_issues.append(
                {
                    "code": "sensor_generation_error",
                    "sample_id": sample["sample_id"],
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
        if number % 500 == 0:
            print(f"传感器进度：{number}/{len(all_samples)}", file=sys.stderr)

    sensors.sort(key=lambda item: item["sample_id"])
    write_jsonl(output / "sensors.jsonl", sensors)
    sensor_position = {
        record["sample_id"]: index for index, record in enumerate(sensors)
    }
    index = {
        "schema_version": 1,
        "seed": args.seed,
        "split_ratios": list(args.ratios),
        "labels": canonical_labels,
        "label_to_index": label_to_index,
        "samples": {
            sample["sample_id"]: {
                "split": sample["split"],
                "label_index": label_to_index[sample["label"]],
                "sensor_index": sensor_position.get(sample["sample_id"]),
            }
            for sample in sorted(all_samples, key=lambda item: item["sample_id"])
        },
    }
    _write_ordered_json(output / "index.json", index)

    generator_counts = Counter(
        str(record.get("source", {}).get("generator", "unknown"))
        for record in sensors
    )
    fallback_counts = Counter(
        str(record.get("source", {}).get("enhancement_status", "not_requested"))
        for record in sensors
    )
    stats = {
        "schema_version": 1,
        **summary,
        "sensor_records": len(sensors),
        "sensor_generator_counts": dict(sorted(generator_counts.items())),
        "sensor_enhancement_status_counts": dict(sorted(fallback_counts.items())),
        "resumed_sensor_records": reused,
        "rejected_resume_records": rejected_resume_records,
        "sensor_errors": len(sensor_issues),
        "qwen_api_key_available": bool(api_key) if args.sensor_mode == "qwen" else None,
    }
    write_json(output / "stats.json", stats)
    report["selected_samples"] = len(all_samples)
    report["sensor_issues"] = sensor_issues
    report["sensor_issue_counts"] = dict(
        sorted(Counter(item["code"] for item in sensor_issues).items())
    )
    write_json(output / "build_report.json", report)
    return stats


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = build(args)
    except (OSError, ValueError) as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.dry_run and summary.get("sensor_errors"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
