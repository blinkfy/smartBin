"""Audit whether object-level Qwen observations can support scene distillation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from mmfc.sensors import contains_leakage_key


DATA = PROJECT_ROOT / "data" / "qwen-material"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "OBJECT_QWEN_SCENE_DISTILLATION_FEASIBILITY.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit object-Qwen to full-image scene distillation"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: root is not an object")
            records.append(value)
    return records


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def audit(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    paths = {
        "train_manifest": DATA / "train.jsonl",
        "val_manifest": DATA / "val.jsonl",
        "test_manifest": DATA / "test.jsonl",
        "sensor_manifest": DATA / "sensors.jsonl",
        "sensor_source": PROJECT_ROOT / "mmfc" / "sensors.py",
        "auditor_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "object-Qwen feasibility inputs missing: " + ", ".join(missing)
        )
    manifests = {
        split: _read_jsonl(paths[f"{split}_manifest"])
        for split in ("train", "val", "test")
    }
    sensors = _read_jsonl(paths["sensor_manifest"])
    errors: list[str] = []
    sample_split: dict[str, str] = {}
    source_splits: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for split, records in manifests.items():
        split_counts[split] = len(records)
        sources: set[str] = set()
        for record in records:
            sample_id = record.get("sample_id")
            source_id = record.get("source_image_id")
            bbox = record.get("bbox")
            if (
                not isinstance(sample_id, str)
                or sample_id in sample_split
                or not isinstance(source_id, str)
                or not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                errors.append(f"invalid object manifest record in {split}")
                continue
            sample_split[sample_id] = split
            sources.add(source_id)
            source_splits.setdefault(source_id, set()).add(split)
        source_counts[split] = len(sources)
    cross_split_sources = sorted(
        source_id
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    )
    if cross_split_sources:
        errors.append(
            f"{len(cross_split_sources)} source images cross data splits"
        )
    sensor_ids: set[str] = set()
    generator_counts: Counter[str] = Counter()
    enhancement_counts: Counter[str] = Counter()
    sensor_source_ids: set[str] = set()
    leakage_record_count = 0
    for record in sensors:
        sample_id = record.get("sample_id")
        source = record.get("source")
        if (
            not isinstance(sample_id, str)
            or sample_id in sensor_ids
            or sample_id not in sample_split
            or not isinstance(source, dict)
        ):
            errors.append("invalid or unmatched sensor record")
            continue
        sensor_ids.add(sample_id)
        source_id = source.get("source_image_id")
        if isinstance(source_id, str):
            sensor_source_ids.add(source_id)
        generator_counts[str(source.get("generator", "unknown"))] += 1
        enhancement_counts[
            str(source.get("enhancement_status", "unknown"))
        ] += 1
        if contains_leakage_key(record):
            leakage_record_count += 1
    if sensor_ids != set(sample_split):
        errors.append("object and sensor sample-id inventories differ")
    if leakage_record_count:
        errors.append(
            f"{leakage_record_count} sensor records contain leakage keys"
        )
    qwen_success = generator_counts.get("qwen3-vl-flash", 0)
    fallback = len(sensors) - qwen_success
    if qwen_success != 8639 or fallback != 15:
        errors.append("Qwen/fallback coverage differs from frozen dataset")

    result = {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "formal_paper_metric": False,
        "evidence_role": "preimplementation_data_and_leakage_feasibility",
        "object_contract": {
            "sample_count": len(sample_split),
            "split_counts": split_counts,
            "source_image_count": len(source_splits),
            "source_split_counts": source_counts,
            "cross_split_source_count": len(cross_split_sources),
        },
        "sensor_contract": {
            "record_count": len(sensors),
            "source_image_count": len(sensor_source_ids),
            "generator_counts": dict(sorted(generator_counts.items())),
            "enhancement_status_counts": dict(
                sorted(enhancement_counts.items())
            ),
            "qwen_success_count": qwen_success,
            "fallback_count": fallback,
            "qwen_coverage": qwen_success / max(1, len(sensors)),
            "leakage_record_count": leakage_record_count,
        },
        "decision": {
            "direct_bbox_aggregate_as_detector_input": "prohibited",
            "reason": (
                "Rasterizing object observations with ground-truth boxes "
                "would expose target localization to the detector."
            ),
            "approved_path": (
                "train_and_validation_teacher_fields_then_rgb_only_scene_"
                "distiller_inference"
            ),
            "approved_training_reads": [
                "train_rgb_images",
                "train_object_bboxes_for_teacher_only",
                "train_object_qwen_observations",
                "validation_rgb_images",
                "validation_teacher_fields_for_selection_only",
            ],
            "final_sidecar_inference_reads": ["full_rgb_image"],
            "final_sidecar_inference_forbidden_reads": [
                "annotation_files",
                "yolo_label_files",
                "object_bboxes",
                "object_classes",
                "object_qwen_records",
            ],
        },
        "claim_boundary": {
            "name": "object-Qwen-distilled image-conditioned scene sidecar",
            "independent_physical_sensor": False,
            "full_image_qwen_api_observation": False,
            "replacement_for_frozen_formal_qwen_protocol": False,
            "development_evidence_only": True,
        },
        "bindings": {name: _binding(path) for name, path in paths.items()},
        "errors": errors,
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 对象Qwen观测到整图sidecar的可行性审计",
        "",
        f"- 对象记录：{len(sample_split)}；源图像：{len(source_splits)}。",
        f"- Qwen成功：{qwen_success}；rule回退：{fallback}；泄漏字段记录：{leakage_record_count}。",
        f"- 跨split源图像：{len(cross_split_sources)}。",
        "",
        "## 决策",
        "",
        "- 禁止把按真实框铺设的对象Qwen网格直接输入检测器，因为它会泄漏目标位置。",
        "- 允许在训练集构造教师场，并以验证教师场选择RGB整图学生网络。",
        "- 最终train/val/test sidecar必须全部由学生网络只读取整图RGB后生成。",
        "- 该路径称为“对象Qwen蒸馏的图像条件整图sidecar”，不是实体传感器或整图Qwen API实测。",
        "- 本扩展只提供开发证据，不替代已冻结的正式整图Qwen三种子协议。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    if errors:
        raise ValueError("; ".join(errors))
    return result


def main() -> None:
    result = audit(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "object_contract": result["object_contract"],
                "sensor_contract": result["sensor_contract"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
