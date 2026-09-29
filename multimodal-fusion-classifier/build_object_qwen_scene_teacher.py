"""Build annotation-derived scene teacher fields from object Qwen sidecars."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mmfc.model import PROJECT_ROOT


DATA = PROJECT_ROOT / "data" / "qwen-material"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "object-qwen-scene-teacher"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build object-Qwen-derived 4x4 scene teacher fields"
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
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: invalid JSON root")
            values.append(value)
    return values


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, values: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")
    temporary.replace(path)


def aggregate_object_grids(
    objects: list[Mapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map object-local 4x4 fields into a 4x4 scene teacher field."""

    if image_width <= 0 or image_height <= 0 or not objects:
        raise ValueError("positive image shape and at least one object required")
    value_area = np.zeros((4, 4), dtype=np.float64)
    covered_area = np.zeros((4, 4), dtype=np.float64)
    scene_cell_width = float(image_width) / 4.0
    scene_cell_height = float(image_height) / 4.0
    scene_cell_area = scene_cell_width * scene_cell_height
    for record in objects:
        bbox = np.asarray(record["bbox"], dtype=np.float64)
        grid = np.asarray(record["normalized_grid"], dtype=np.float64)
        if (
            bbox.shape != (4,)
            or grid.shape != (4, 4)
            or not np.isfinite(bbox).all()
            or not np.isfinite(grid).all()
            or float(grid.min()) < 0.0
            or float(grid.max()) > 1.0
        ):
            raise ValueError("invalid bbox or object grid")
        x1, y1, x2, y2 = bbox.tolist()
        x1 = min(float(image_width), max(0.0, x1))
        x2 = min(float(image_width), max(0.0, x2))
        y1 = min(float(image_height), max(0.0, y1))
        y2 = min(float(image_height), max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("object bbox has no positive area")
        object_cell_width = (x2 - x1) / 4.0
        object_cell_height = (y2 - y1) / 4.0
        for object_y in range(4):
            oy1 = y1 + object_y * object_cell_height
            oy2 = y1 + (object_y + 1) * object_cell_height
            for object_x in range(4):
                ox1 = x1 + object_x * object_cell_width
                ox2 = x1 + (object_x + 1) * object_cell_width
                value = float(grid[object_y, object_x])
                for scene_y in range(4):
                    sy1 = scene_y * scene_cell_height
                    sy2 = (scene_y + 1) * scene_cell_height
                    overlap_y = max(0.0, min(oy2, sy2) - max(oy1, sy1))
                    if overlap_y <= 0.0:
                        continue
                    for scene_x in range(4):
                        sx1 = scene_x * scene_cell_width
                        sx2 = (scene_x + 1) * scene_cell_width
                        overlap_x = max(
                            0.0, min(ox2, sx2) - max(ox1, sx1)
                        )
                        overlap = overlap_x * overlap_y
                        if overlap <= 0.0:
                            continue
                        value_area[scene_y, scene_x] += value * overlap
                        covered_area[scene_y, scene_x] += overlap
    mean_value = np.divide(
        value_area,
        covered_area,
        out=np.zeros_like(value_area),
        where=covered_area > 0.0,
    )
    coverage = np.clip(covered_area / scene_cell_area, 0.0, 1.0)
    teacher = np.clip(mean_value * np.sqrt(coverage), 0.0, 1.0)
    return teacher.astype(np.float32), coverage.astype(np.float32)


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    paths = {
        f"{split}_manifest": DATA / f"{split}.jsonl"
        for split in SPLITS
    }
    paths.update(
        {
            "sensor_manifest": DATA / "sensors.jsonl",
            "feasibility_audit": (
                PROJECT_ROOT
                / "reports"
                / "qwen-material"
                / "OBJECT_QWEN_SCENE_DISTILLATION_FEASIBILITY.json"
            ),
            "builder_source": Path(__file__),
        }
    )
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "scene teacher inputs missing: " + ", ".join(missing)
        )
    feasibility = json.loads(
        paths["feasibility_audit"].read_text(encoding="utf-8-sig")
    )
    if (
        feasibility.get("valid") is not True
        or feasibility.get("decision", {}).get(
            "direct_bbox_aggregate_as_detector_input"
        )
        != "prohibited"
    ):
        raise ValueError("scene distillation feasibility gate is invalid")

    sensors = {
        record["sample_id"]: record
        for record in _read_jsonl(paths["sensor_manifest"])
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for split in SPLITS:
        for sample in _read_jsonl(paths[f"{split}_manifest"]):
            sensor = sensors.get(sample.get("sample_id"))
            if sensor is None:
                raise ValueError(f"missing sensor for {sample.get('sample_id')}")
            grouped[(split, sample["source_image_id"])].append(
                {
                    "sample_id": sample["sample_id"],
                    "bbox": sample["bbox"],
                    "image_width": sample["image_width"],
                    "image_height": sample["image_height"],
                    "normalized_grid": sensor["normalized_grid"],
                    "generator": sensor.get("source", {}).get(
                        "generator", "unknown"
                    ),
                }
            )

    records: list[dict[str, Any]] = []
    generator_counts: Counter[str] = Counter()
    split_counts = {split: 0 for split in SPLITS}
    object_split_counts = {split: 0 for split in SPLITS}
    for (split, source_id), objects in sorted(grouped.items()):
        widths = {int(item["image_width"]) for item in objects}
        heights = {int(item["image_height"]) for item in objects}
        if len(widths) != 1 or len(heights) != 1:
            raise ValueError(f"image shape mismatch for {source_id}")
        teacher, coverage = aggregate_object_grids(
            objects,
            image_width=next(iter(widths)),
            image_height=next(iter(heights)),
        )
        generators = Counter(str(item["generator"]) for item in objects)
        generator_counts.update(generators)
        split_counts[split] += 1
        object_split_counts[split] += len(objects)
        records.append(
            {
                "schema_version": 1,
                "source_image_id": source_id,
                "split": split,
                "image_width": next(iter(widths)),
                "image_height": next(iter(heights)),
                "grid_shape": [4, 4],
                "teacher_grid": [
                    [round(float(value), 6) for value in row]
                    for row in teacher
                ],
                "coverage_grid": [
                    [round(float(value), 6) for value in row]
                    for row in coverage
                ],
                "object_observation_count": len(objects),
                "qwen_observation_count": generators.get(
                    "qwen3.5-flash", 0
                ),
                "fallback_observation_count": (
                    len(objects)
                    - generators.get("qwen3.5-flash", 0)
                ),
                "source": {
                    "generator": "object-qwen-scene-teacher-v1",
                    "aggregation": (
                        "overlap_area_weighted_mean_times_"
                        "sqrt_scene_cell_coverage"
                    ),
                    "annotation_derived": True,
                    "direct_detector_input_permitted": False,
                },
            }
        )
    output_path = output / "teacher_fields.jsonl"
    split_paths = {
        split: output / f"{split}.jsonl" for split in SPLITS
    }
    _atomic_jsonl(output_path, records)
    for split, split_path in split_paths.items():
        _atomic_jsonl(
            split_path,
            [record for record in records if record["split"] == split],
        )
    values = np.asarray(
        [value for record in records for row in record["teacher_grid"] for value in row],
        dtype=np.float64,
    )
    coverage_values = np.asarray(
        [value for record in records for row in record["coverage_grid"] for value in row],
        dtype=np.float64,
    )
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "artifact_role": "annotation_derived_distillation_teacher_only",
        "direct_detector_input_permitted": False,
        "source_image_count": len(records),
        "object_observation_count": sum(object_split_counts.values()),
        "split_counts": split_counts,
        "object_split_counts": object_split_counts,
        "generator_counts": dict(sorted(generator_counts.items())),
        "aggregation": {
            "method": (
                "overlap_area_weighted_mean_times_sqrt_scene_cell_coverage"
            ),
            "grid_shape": [4, 4],
            "teacher_min": float(values.min()),
            "teacher_max": float(values.max()),
            "teacher_mean": float(values.mean()),
            "teacher_std": float(values.std()),
            "nonzero_fraction": float(np.mean(values > 0.0)),
            "coverage_mean": float(coverage_values.mean()),
            "coverage_nonzero_fraction": float(
                np.mean(coverage_values > 0.0)
            ),
        },
        "usage_contract": {
            "training_splits_allowed": ["train"],
            "selection_splits_allowed": ["val"],
            "test_split_training_or_selection_allowed": False,
            "final_joint_head_input_allowed": False,
            "student_prediction_required_for_joint_head": True,
        },
        "output": {
            "combined": _binding(output_path),
            "splits": {
                split: _binding(path)
                for split, path in split_paths.items()
            },
        },
        "bindings": {name: _binding(path) for name, path in paths.items()},
    }
    _atomic_json(output / "BUILD_REPORT.json", report)
    return report


def main() -> None:
    result = build(parse_args().output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
