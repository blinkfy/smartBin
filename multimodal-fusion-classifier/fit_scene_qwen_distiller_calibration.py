"""Fit one global affine calibration on validation predictions only."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmfc.model import PROJECT_ROOT
from mmfc.scene_distillation import (
    SceneQwenDistiller,
    SceneTeacherDataset,
    prediction_metrics,
)


ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "object-qwen-scene-distillation"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "scene-qwen-distiller-v1" / "best.pt"
)
DEFAULT_TEACHER = (
    PROJECT_ROOT / "data" / "object-qwen-scene-teacher" / "val.jsonl"
)
DEFAULT_IMAGES = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "images"
)
DEFAULT_RECOVERY_PROTOCOL = ROOT / "CALIBRATION_RECOVERY_PROTOCOL.json"
DEFAULT_OUTPUT = ROOT / "CALIBRATION_PARAMETERS.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit validation-only affine scene-distiller calibration"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument(
        "--recovery-protocol",
        type=Path,
        default=DEFAULT_RECOVERY_PROTOCOL,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.inference_mode()
def _collect(
    model: SceneQwenDistiller,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    coverages: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            prediction = model(images)
        predictions.append(prediction.float().cpu().numpy())
        teachers.append(batch["teacher"].numpy())
        coverages.append(batch["coverage"].numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(teachers),
        np.concatenate(coverages),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.batch_size) < 1 or int(args.workers) < 0:
        raise ValueError("invalid batch size or worker count")
    paths = {
        "checkpoint": _resolve(args.checkpoint),
        "val_teacher": _resolve(args.teacher),
        "recovery_protocol": _resolve(args.recovery_protocol),
        "fit_source": Path(__file__).resolve(),
    }
    output = _resolve(args.output)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "calibration fit inputs missing: " + ", ".join(missing)
        )
    protocol = json.loads(
        paths["recovery_protocol"].read_text(encoding="utf-8-sig")
    )
    if (
        protocol.get("valid") is not True
        or protocol.get("state")
        != "frozen_before_calibration_fit_and_test_evaluation"
        or protocol.get("calibration", {}).get("method")
        != "global_nonnegative_affine_least_squares_then_clip_0_1"
        or protocol.get("calibration", {}).get("fit_split") != "val"
        or protocol.get("calibration", {}).get("test_read_during_fit")
        is not False
    ):
        raise ValueError("calibration recovery protocol is invalid")
    device = torch.device(args.device)
    checkpoint = torch.load(
        paths["checkpoint"], map_location=device, weights_only=True
    )
    model = SceneQwenDistiller().to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = SceneTeacherDataset(
        paths["val_teacher"],
        _resolve(args.images_root),
        split="val",
        image_size=int(checkpoint["image_size"]),
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.workers),
        "pin_memory": device.type == "cuda",
    }
    if int(args.workers):
        loader_kwargs.update(
            {"persistent_workers": True, "prefetch_factor": 2}
        )
    prediction, teacher, coverage = _collect(
        model, DataLoader(**loader_kwargs), device=device
    )
    x = prediction.reshape(-1).astype(np.float64)
    y = teacher.reshape(-1).astype(np.float64)
    centered = x - x.mean()
    denominator = float(np.square(centered).sum())
    slope = (
        float(np.dot(centered, y - y.mean()) / denominator)
        if denominator > 0.0
        else 0.0
    )
    slope = min(5.0, max(0.0, slope))
    intercept = min(1.0, max(-1.0, float(y.mean() - slope * x.mean())))
    calibrated = np.clip(
        slope * prediction.astype(np.float64) + intercept,
        0.0,
        1.0,
    )
    train_mean_grid = checkpoint["train_mean_grid"].cpu().numpy()
    raw_metrics = prediction_metrics(
        prediction,
        teacher,
        coverage,
        train_mean_grid=train_mean_grid,
    )
    calibrated_metrics = prediction_metrics(
        calibrated,
        teacher,
        coverage,
        train_mean_grid=train_mean_grid,
    )
    acceptance = {
        "validation_mse_below_train_mean_grid_baseline": (
            calibrated_metrics["mse"]
            < calibrated_metrics["train_mean_grid_baseline_mse"]
        ),
        "validation_elementwise_spearman_above_zero": (
            calibrated_metrics["elementwise_spearman"] > 0.0
        ),
    }
    result = {
        "schema_version": 1,
        "state": (
            "frozen_before_test_evaluation"
            if all(acceptance.values())
            else "failed_validation_acceptance"
        ),
        "valid": all(acceptance.values()),
        "formal_paper_metric": False,
        "method": "global_nonnegative_affine_least_squares_then_clip_0_1",
        "fit_split": "val",
        "test_teacher_read": False,
        "parameters": {
            "slope": slope,
            "intercept": intercept,
            "slope_bounds": [0.0, 5.0],
            "intercept_bounds": [-1.0, 1.0],
            "output_clip": [0.0, 1.0],
        },
        "validation": {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
        },
        "acceptance": acceptance,
        "bindings": {name: _binding(path) for name, path in paths.items()},
    }
    _atomic_json(output, result)
    if not result["valid"]:
        raise ValueError("validation-only affine calibration failed")
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
