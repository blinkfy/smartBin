"""Generate annotation-free full-image sidecars with the distilled student."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from build_scene_sensor_sidecars import discover_images
from mmfc.model import PROJECT_ROOT
from mmfc.scene_distillation import SceneQwenDistiller, image_to_tensor
from mmfc.scene_sensors import simulate_scene_rule_sensor


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "scene-qwen-distiller-v1" / "best.pt"
)
DEFAULT_IMAGES = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "images"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "scene-qwen-distilled-sidecars"
)
DEFAULT_CALIBRATION = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "object-qwen-scene-distillation"
    / "CALIBRATION_PARAMETERS.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-image Qwen-distilled scene sidecars"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--calibration", type=Path, default=DEFAULT_CALIBRATION
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--property-seed", type=int, default=20260729)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _portable(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": _portable(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in sorted(values, key=lambda item: item["source_image_id"]):
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


class SceneInferenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, images_root: Path, *, image_size: int) -> None:
        self.records = discover_images(images_root)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = _resolve(Path(record["image_path"]))
        with Image.open(path) as image:
            tensor = image_to_tensor(image, self.image_size)
            width, height = image.size
        return {
            "image": tensor,
            "source_image_id": record["source_image_id"],
            "split": record["split"],
            "image_path": record["image_path"],
            "image_width": int(width),
            "image_height": int(height),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "metadata": [
            {key: value for key, value in item.items() if key != "image"}
            for item in batch
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be positive")
    if int(args.workers) < 0 or int(args.workers) > 16:
        raise ValueError("--workers must be in range 0..16")
    checkpoint_path = _resolve(args.checkpoint)
    calibration_path = _resolve(args.calibration)
    images_root = _resolve(args.images_root)
    output = _resolve(args.output)
    if not checkpoint_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError(
            "checkpoint or frozen calibration parameters are missing"
        )
    device = torch.device(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8-sig")
    )
    if (
        calibration.get("valid") is not True
        or calibration.get("state") != "frozen_before_test_evaluation"
        or calibration.get("bindings", {}).get("checkpoint", {}).get(
            "sha256"
        )
        != _sha256(checkpoint_path)
    ):
        raise ValueError("invalid or mismatched calibration parameters")
    slope = float(calibration["parameters"]["slope"])
    intercept = float(calibration["parameters"]["intercept"])
    if (
        checkpoint.get("architecture") != "SceneQwenDistiller"
        or int(checkpoint.get("image_size", 0)) < 64
    ):
        raise ValueError("invalid scene distiller checkpoint")
    model = SceneQwenDistiller().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_size = int(checkpoint["image_size"])
    dataset = SceneInferenceDataset(images_root, image_size=image_size)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": _collate,
    }
    if int(args.workers):
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**kwargs)
    predictions: dict[str, list[list[float]]] = {}
    started = time.monotonic()
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with amp_context:
                grids = (
                    model(images).float() * slope + intercept
                ).clamp(0.0, 1.0).cpu().numpy()
            for metadata, grid in zip(batch["metadata"], grids):
                predictions[metadata["source_image_id"]] = [
                    [round(float(value), 6) for value in row]
                    for row in grid[0]
                ]
    if len(predictions) != len(dataset):
        raise ValueError("prediction coverage differs from image inventory")

    records: list[dict[str, Any]] = []
    for image_record in dataset.records:
        source_id = image_record["source_image_id"]
        path = _resolve(Path(image_record["image_path"]))
        with Image.open(path) as image:
            rule = simulate_scene_rule_sensor(
                image, source_id, seed=int(args.property_seed)
            )
            width, height = image.size
        grid = predictions[source_id]
        records.append(
            {
                "schema_version": 1,
                "source_image_id": source_id,
                "split": image_record["split"],
                "image_path": image_record["image_path"],
                "image_width": int(width),
                "image_height": int(height),
                "grid_shape": [4, 4],
                "grid": grid,
                "normalized_grid": grid,
                "properties": rule["properties"],
                "source": {
                    "generator": "object-qwen-scene-distiller-v1",
                    "observation_scope": "full_image",
                    "full_image_only": True,
                    "annotation_inputs": [],
                    "inputs": [
                        "full_rgb_image",
                        "frozen_scene_distiller",
                    ],
                    "student_checkpoint_sha256": _sha256(checkpoint_path),
                    "calibration_sha256": _sha256(calibration_path),
                    "object_qwen_teacher_used_at_inference": False,
                    "property_generator": "scene-rule-v1-image-only",
                },
            }
        )
    observations = output / "observations.jsonl"
    _atomic_jsonl(observations, records)
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "val", "test")
    }
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "mode": "object-qwen-scene-distilled",
        "model": "SceneQwenDistiller",
        "source_contract": {
            "full_image_only": True,
            "annotation_files_read": False,
            "label_files_read": False,
            "bbox_inputs": False,
            "polygon_inputs": False,
            "class_inputs": False,
        },
        "source_image_count": len(records),
        "split_counts": split_counts,
        "qwen_success_count": 0,
        "fallback_count": 0,
        "distilled_prediction_count": len(records),
        "object_qwen_teacher_api_success_count": 8639,
        "object_qwen_teacher_fallback_count": 15,
        "output": _binding(observations),
        "bindings": {
            "checkpoint": _binding(checkpoint_path),
            "calibration": _binding(calibration_path),
            "predictor_source": _binding(Path(__file__)),
        },
        "runtime": {
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "device": str(device),
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": datetime.now().astimezone().isoformat(),
        },
        "limitations": [
            (
                "The grid is an RGB-only student prediction distilled from "
                "annotation-derived object-crop Qwen teacher fields."
            ),
            "It is not a direct full-image Qwen API observation.",
            "It is not a synchronized physical sensor measurement.",
        ],
    }
    _atomic_json(output / "BUILD_REPORT.json", report)
    return report


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
