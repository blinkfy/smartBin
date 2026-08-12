"""Prepare leakage-controlled soft-sensor targets for joint YOLO segmentation.

The object-level Qwen sidecars are converted into one spatial 4x4 target per
source image. These targets supervise an RGB-to-sensor prediction branch only;
they are never required by the detector forward pass at evaluation or
deployment time.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from mmfc.model import CLIPPrototypeEncoder, PROJECT_ROOT
from mmfc.training import load_config, read_jsonl, resolve_project_path


SPLITS = ("train", "val", "test")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "data" / "qwen-material"
DEFAULT_YOLO_EXPORT = PROJECT_ROOT / "data" / "yolo-seg-unique-linked"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "joint-multimodal-seg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build image-level Qwen soft-sensor distillation targets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--sensor-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "sensors.jsonl",
    )
    parser.add_argument("--yolo-export", type=Path, default=DEFAULT_YOLO_EXPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--clip-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _write_or_validate(path: Path, payload: bytes) -> str:
    """Write a new artifact or fail closed if an existing artifact differs."""

    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to overwrite non-identical prepared artifact: {path}"
            )
        return "validated"
    _atomic_write_bytes(path, payload)
    return "created"


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    return text.encode("utf-8")


def _as_grid(value: object, *, name: str) -> np.ndarray:
    grid = np.asarray(value, dtype=np.float32)
    if grid.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4,4], got {list(grid.shape)}")
    if not np.isfinite(grid).all():
        raise ValueError(f"{name} contains non-finite values")
    if float(grid.min()) < 0.0 or float(grid.max()) > 1.0:
        raise ValueError(f"{name} must be normalized to [0,1]")
    return grid


def _bilinear_sample(grid: np.ndarray, x: float, y: float) -> float:
    """Sample a 4x4 object grid at normalized object coordinates."""

    px = min(3.0, max(0.0, x * 4.0 - 0.5))
    py = min(3.0, max(0.0, y * 4.0 - 0.5))
    x0, y0 = int(math.floor(px)), int(math.floor(py))
    x1, y1 = min(3, x0 + 1), min(3, y0 + 1)
    wx, wy = px - x0, py - y0
    top = float(grid[y0, x0]) * (1.0 - wx) + float(grid[y0, x1]) * wx
    bottom = float(grid[y1, x0]) * (1.0 - wx) + float(grid[y1, x1]) * wx
    return top * (1.0 - wy) + bottom * wy


def rasterize_sensor_target(
    objects: Sequence[tuple[Sequence[float], np.ndarray]],
    *,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize object Qwen grids into a full-image 4x4 soft-sensor target."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    weighted_value = np.zeros((4, 4), dtype=np.float64)
    covered_area = np.zeros((4, 4), dtype=np.float64)
    cell_area = 1.0 / 16.0

    for bbox, raw_grid in objects:
        if len(bbox) != 4:
            raise ValueError("bbox must contain four values")
        grid = _as_grid(raw_grid, name="object sensor grid")
        x1 = min(1.0, max(0.0, float(bbox[0]) / image_width))
        y1 = min(1.0, max(0.0, float(bbox[1]) / image_height))
        x2 = min(1.0, max(0.0, float(bbox[2]) / image_width))
        y2 = min(1.0, max(0.0, float(bbox[3]) / image_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox has non-positive normalized area")

        for gy in range(4):
            cell_y1, cell_y2 = gy / 4.0, (gy + 1) / 4.0
            iy1, iy2 = max(y1, cell_y1), min(y2, cell_y2)
            if iy2 <= iy1:
                continue
            for gx in range(4):
                cell_x1, cell_x2 = gx / 4.0, (gx + 1) / 4.0
                ix1, ix2 = max(x1, cell_x1), min(x2, cell_x2)
                if ix2 <= ix1:
                    continue
                area = (ix2 - ix1) * (iy2 - iy1)
                local_x = ((ix1 + ix2) * 0.5 - x1) / (x2 - x1)
                local_y = ((iy1 + iy2) * 0.5 - y1) / (y2 - y1)
                value = _bilinear_sample(grid, local_x, local_y)
                weighted_value[gy, gx] += area * value
                covered_area[gy, gx] += area

    coverage = np.clip(covered_area / cell_area, 0.0, 1.0)
    mean_value = np.divide(
        weighted_value,
        covered_area,
        out=np.zeros_like(weighted_value),
        where=covered_area > 0,
    )
    signal = np.clip(mean_value * coverage, 0.0, 1.0)
    return signal.astype(np.float32), coverage.astype(np.float32)


def build_image_targets(
    *,
    manifest_dir: Path,
    sensor_manifest: Path,
    duplicate_sample_ids: set[str],
) -> list[dict[str, Any]]:
    sensors: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(sensor_manifest):
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in sensors:
            raise ValueError(f"invalid or duplicate sensor sample id: {sample_id}")
        sensors[sample_id] = record

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_samples: set[str] = set()
    for split in SPLITS:
        for record in read_jsonl(manifest_dir / f"{split}.jsonl"):
            sample_id = str(record.get("sample_id", ""))
            if not sample_id or sample_id in seen_samples:
                raise ValueError(f"invalid or duplicate object sample id: {sample_id}")
            seen_samples.add(sample_id)
            if sample_id not in duplicate_sample_ids:
                grouped[(split, str(record["source_image_id"]))].append(record)

    if set(sensors) != seen_samples:
        missing = sorted(seen_samples - set(sensors))[:5]
        extra = sorted(set(sensors) - seen_samples)[:5]
        raise ValueError(f"sensor/object sample mismatch; missing={missing}, extra={extra}")

    targets: list[dict[str, Any]] = []
    for (split, source_id), records in sorted(grouped.items()):
        image_paths = {str(record["image_path"]) for record in records}
        dimensions = {
            (int(record["image_width"]), int(record["image_height"]))
            for record in records
        }
        if len(image_paths) != 1 or len(dimensions) != 1:
            raise ValueError(f"inconsistent source metadata: {source_id}")
        width, height = next(iter(dimensions))
        objects: list[tuple[Sequence[float], np.ndarray]] = []
        qwen_success = 0
        for record in sorted(records, key=lambda item: str(item["sample_id"])):
            sensor = sensors[str(record["sample_id"])]
            grid = _as_grid(sensor.get("normalized_grid"), name="normalized_grid")
            objects.append((record["bbox"], grid))
            source = sensor.get("source", {})
            if (
                isinstance(source, Mapping)
                and source.get("enhancement_status") == "success"
            ):
                qwen_success += 1
        signal, coverage = rasterize_sensor_target(
            objects,
            image_width=width,
            image_height=height,
        )
        targets.append(
            {
                "schema_version": 1,
                "source_image_id": source_id,
                "split": split,
                "image_path": next(iter(image_paths)),
                "image_width": width,
                "image_height": height,
                "normalized_grid": np.round(signal, 8).tolist(),
                "coverage_grid": np.round(coverage, 8).tolist(),
                "object_count": len(records),
                "qwen_success_count": qwen_success,
                "rule_fallback_count": len(records) - qwen_success,
                "target_role": "training_only_rgb_to_soft_sensor_distillation",
            }
        )
    return targets


def build_text_prototypes(
    class_descriptions: Sequence[str],
    *,
    model_name: str,
    local_files_only: bool,
) -> np.ndarray:
    encoder = CLIPPrototypeEncoder(
        model_name,
        freeze_layers=12,
        local_files_only=local_files_only,
    )
    encoder.eval()
    with torch.inference_mode():
        prototypes = encoder(class_descriptions, torch.device("cpu"))
    result = prototypes.detach().cpu().numpy().astype(np.float32)
    expected = (len(class_descriptions), 512)
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(
            f"CLIP prototypes must have shape {expected}, got {result.shape}"
        )
    return result


def _npy_bytes(array: np.ndarray) -> bytes:
    import io

    handle = io.BytesIO()
    np.save(handle, array, allow_pickle=False)
    return handle.getvalue()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_project_path(args.config)
    manifest_dir = resolve_project_path(args.manifest_dir)
    sensor_manifest = resolve_project_path(args.sensor_manifest)
    yolo_export = resolve_project_path(args.yolo_export)
    output = resolve_project_path(args.output)
    export_report_path = yolo_export / "export_report.json"
    required = [
        config_path,
        sensor_manifest,
        export_report_path,
        *(manifest_dir / f"{split}.jsonl" for split in SPLITS),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing preparation inputs: {missing}")

    config = load_config(config_path)
    classes = config.get("classes", [])
    class_names = [str(item["name"]) for item in classes]
    descriptions = [str(item["description"]) for item in classes]
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    if class_names != list(export_report["class_names"]):
        raise ValueError("config class order differs from YOLO export")
    duplicate_sample_ids = {
        str(sample_id)
        for group in export_report.get("duplicate_object_groups", [])
        for sample_id in group.get("duplicate_sample_ids", [])
    }

    targets = build_image_targets(
        manifest_dir=manifest_dir,
        sensor_manifest=sensor_manifest,
        duplicate_sample_ids=duplicate_sample_ids,
    )
    expected_sources = int(export_report["source_image_count"])
    if len(targets) != expected_sources:
        raise ValueError(
            f"prepared {len(targets)} image targets, expected {expected_sources}"
        )
    split_counts = Counter(str(record["split"]) for record in targets)
    if dict(sorted(split_counts.items())) != dict(
        sorted(export_report["split_source_counts"].items())
    ):
        raise ValueError("prepared split source counts differ from YOLO export")

    clip_model = str(config["model"]["clip_model"])
    prototypes = build_text_prototypes(
        descriptions,
        model_name=clip_model,
        local_files_only=bool(args.clip_local_files_only),
    )
    targets_path = output / "image_sensor_targets.jsonl"
    prototypes_path = output / "text_prototypes.npy"
    target_state = _write_or_validate(targets_path, _jsonl_bytes(targets))
    prototype_state = _write_or_validate(prototypes_path, _npy_bytes(prototypes))

    target_values = np.asarray(
        [record["normalized_grid"] for record in targets], dtype=np.float32
    )
    coverage_values = np.asarray(
        [record["coverage_grid"] for record in targets], dtype=np.float32
    )
    input_bindings = {
        path.resolve().as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in required
    }
    report = {
        "schema_version": 1,
        "protocol": "joint_yolo_soft_sensor_target_preparation",
        "valid": True,
        "output": {
            "target_manifest": targets_path.resolve().as_posix(),
            "target_manifest_sha256": sha256_file(targets_path),
            "text_prototypes": prototypes_path.resolve().as_posix(),
            "text_prototypes_sha256": sha256_file(prototypes_path),
            "target_state": target_state,
            "prototype_state": prototype_state,
        },
        "dataset": {
            "source_image_count": len(targets),
            "split_source_counts": dict(sorted(split_counts.items())),
            "canonical_object_count": sum(
                int(record["object_count"]) for record in targets
            ),
            "removed_exact_duplicate_count": len(duplicate_sample_ids),
            "qwen_success_count": sum(
                int(record["qwen_success_count"]) for record in targets
            ),
            "rule_fallback_count": sum(
                int(record["rule_fallback_count"]) for record in targets
            ),
        },
        "target_statistics": {
            "signal_mean": float(target_values.mean()),
            "signal_std": float(target_values.std()),
            "signal_nonzero_fraction": float((target_values > 0).mean()),
            "coverage_mean": float(coverage_values.mean()),
            "coverage_nonzero_fraction": float((coverage_values > 0).mean()),
        },
        "text": {
            "clip_model": clip_model,
            "local_files_only": bool(args.clip_local_files_only),
            "class_count": len(class_names),
            "class_names": class_names,
            "prototype_shape": list(prototypes.shape),
            "prototype_dtype": str(prototypes.dtype),
        },
        "forward_contract": {
            "training": (
                "image targets supervise the RGB-to-soft-sensor branch through "
                "an auxiliary loss"
            ),
            "evaluation": (
                "the detector forward pass receives RGB only and predicts its "
                "own 4x4 soft-sensor grid"
            ),
            "forbidden": (
                "ground-truth boxes, labels, and prepared target grids must not "
                "be detector inputs during metric evaluation"
            ),
        },
        "limitations": [
            "Targets are distilled from object-level Qwen/rule sidecars and annotations.",
            "They are not synchronized hardware sensor measurements.",
            "The resulting model remains RGB plus text plus predicted soft-sensor, not RGB-D.",
        ],
        "input_bindings": input_bindings,
    }
    report_path = output / "PREPARATION_REPORT.json"
    _atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> None:
    report = prepare(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
