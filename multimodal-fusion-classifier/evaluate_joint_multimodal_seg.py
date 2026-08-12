"""Audit the joint YOLO head and its RGB-predicted soft-sensor branch."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path
from prepare_joint_multimodal_seg import sha256_file
from train_yolo_seg import _json_safe, _runtime_environment


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg"
    / "yolo11s-seg-joint-v1-seed42"
    / "weights"
    / "best.pt"
)
DEFAULT_TARGETS = (
    PROJECT_ROOT
    / "data"
    / "joint-multimodal-seg"
    / "image_sensor_targets.jsonl"
)
DEFAULT_DATA = PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "dataset.yaml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
    / "yolo11s-seg-joint-v1-seed42"
    / "diagnostics"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate joint-head sensor diagnostics and branch masking."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sensor-targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-detector-ablation", action="store_true")
    return parser.parse_args()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_test_targets(path: Path) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "test":
                continue
            source_id = str(record.get("source_image_id", ""))
            if not source_id or source_id in seen:
                raise ValueError(f"invalid source id at line {line_number}")
            seen.add(source_id)
            records.append(record)
    records.sort(key=lambda item: str(item["source_image_id"]))
    if not records:
        raise ValueError("no test soft-sensor targets found")
    return records


def _letterbox_rgb(image: Image.Image, size: int) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    ratio = min(size / height, size / width, 1.0)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    if (resized_width, resized_height) != (width, height):
        image = image.resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left = round((size - resized_width) / 2 - 0.1)
    top = round((size - resized_height) / 2 - 0.1)
    canvas.paste(image, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


def _batches(
    values: Sequence[dict[str, Any]], batch_size: int
) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or first.std() <= 0 or second.std() <= 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def sensor_prediction_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    coverage: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, Any]:
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (predicted, target, coverage, uncertainty)
    ]
    predicted, target, coverage, uncertainty = arrays
    if not (
        predicted.shape
        == target.shape
        == coverage.shape
        == uncertainty.shape
    ):
        raise ValueError("sensor diagnostic arrays must have identical shape")
    if predicted.ndim != 4 or predicted.shape[1:] != (1, 4, 4):
        raise ValueError("sensor diagnostic arrays must have shape [N,1,4,4]")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("sensor diagnostic arrays contain non-finite values")

    error = predicted - target
    absolute_error = np.abs(error)
    covered = coverage > 0
    uncovered = ~covered
    per_image_mae = absolute_error.mean(axis=(1, 2, 3))

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        return float(values[mask].mean()) if mask.any() else None

    return {
        "sample_count": int(predicted.shape[0]),
        "cell_count": int(predicted.size),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "pearson_predicted_target": _pearson(predicted, target),
        "covered_cell_mae": masked_mean(absolute_error, covered),
        "uncovered_cell_mae": masked_mean(absolute_error, uncovered),
        "covered_cell_fraction": float(covered.mean()),
        "per_image_mae_mean": float(per_image_mae.mean()),
        "per_image_mae_std": float(per_image_mae.std(ddof=1)),
        "uncertainty_mean": float(uncertainty.mean()),
        "uncertainty_absolute_error_pearson": _pearson(
            uncertainty, absolute_error
        ),
    }


def _predict_sensor_targets(
    *,
    checkpoint: Path,
    targets: Sequence[dict[str, Any]],
    image_size: int,
    batch_size: int,
    device_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from ultralytics import YOLO

    if image_size <= 0 or batch_size <= 0:
        raise ValueError("image size and batch size must be positive")
    device = torch.device(
        f"cuda:{device_name}"
        if torch.cuda.is_available() and str(device_name).isdigit()
        else device_name
    )
    wrapper = YOLO(str(checkpoint))
    model = wrapper.model.to(device).eval()
    predicted_batches = []
    target_batches = []
    coverage_batches = []
    uncertainty_batches = []
    rows: list[dict[str, Any]] = []

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    for records in _batches(targets, batch_size):
        images = []
        for record in records:
            image_path = resolve_project_path(str(record["image_path"]))
            with Image.open(image_path) as image:
                images.append(_letterbox_rgb(image, image_size))
        batch = (
            torch.from_numpy(np.stack(images))
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
        with torch.inference_mode(), autocast:
            _ = model(batch)
        auxiliary = model.joint_fusion.last_aux
        predicted = (
            auxiliary["predicted_sensor_grid"].detach().float().cpu().numpy()
        )
        uncertainty = (
            auxiliary["sensor_uncertainty"].detach().float().cpu().numpy()
        )
        target = np.asarray(
            [record["normalized_grid"] for record in records], dtype=np.float32
        )[:, None]
        coverage = np.asarray(
            [record["coverage_grid"] for record in records], dtype=np.float32
        )[:, None]
        predicted_batches.append(predicted)
        target_batches.append(target)
        coverage_batches.append(coverage)
        uncertainty_batches.append(uncertainty)
        for index, record in enumerate(records):
            rows.append(
                {
                    "source_image_id": str(record["source_image_id"]),
                    "predicted_grid": np.round(predicted[index, 0], 8).tolist(),
                    "target_grid": record["normalized_grid"],
                    "coverage_grid": record["coverage_grid"],
                    "uncertainty_grid": np.round(
                        uncertainty[index, 0], 8
                    ).tolist(),
                    "mae": float(np.abs(predicted[index, 0] - target[index, 0]).mean()),
                }
            )

    predicted_all = np.concatenate(predicted_batches)
    target_all = np.concatenate(target_batches)
    coverage_all = np.concatenate(coverage_batches)
    uncertainty_all = np.concatenate(uncertainty_batches)
    metrics = sensor_prediction_metrics(
        predicted_all,
        target_all,
        coverage_all,
        uncertainty_all,
    )
    metrics["forward_input_keys"] = ["RGB"]
    metrics["target_grid_passed_to_model"] = False
    metrics["external_sensor_passed_to_model"] = False
    metrics["branch_parameters"] = {
        "text_scale": [
            float(value)
            for value in model.joint_fusion.text_scale.detach().float().cpu()
        ],
        "sensor_scale": [
            float(value)
            for value in model.joint_fusion.sensor_scale.detach().float().cpu()
        ],
        "text_output_l2": [
            float(module.weight.detach().float().norm().cpu())
            for module in model.joint_fusion.text_output
        ],
        "sensor_output_l2": [
            float(module.weight.detach().float().norm().cpu())
            for module in model.joint_fusion.sensor_output
        ],
    }
    return metrics, rows


def _detector_masking_evaluation(
    *,
    checkpoint: Path,
    data: Path,
    output: Path,
    image_size: int,
    batch_size: int,
    device: str,
    workers: int,
) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    modes = {
        "full": (True, True),
        "no_text": (False, True),
        "no_sensor": (True, False),
        "vision_only": (False, False),
    }
    results: dict[str, Any] = {}
    for mode, (text_enabled, sensor_enabled) in modes.items():
        wrapper = YOLO(str(checkpoint))
        with torch.no_grad():
            if not text_enabled:
                wrapper.model.joint_fusion.text_scale.zero_()
            if not sensor_enabled:
                wrapper.model.joint_fusion.sensor_scale.zero_()
        metrics = wrapper.val(
            data=str(data),
            split="test",
            imgsz=image_size,
            batch=batch_size,
            device=device,
            workers=workers,
            project=str(output / "detector-masking"),
            name=mode,
            exist_ok=False,
            plots=False,
            save_json=False,
            verbose=False,
        )
        results[mode] = {
            "text_enabled": text_enabled,
            "predicted_sensor_enabled": sensor_enabled,
            "metrics": _json_safe(metrics.results_dict),
            "speed_ms_per_image": _json_safe(metrics.speed),
        }

    full = results["full"]["metrics"]
    for mode, result in results.items():
        if mode == "full":
            continue
        metrics = result["metrics"]
        result["full_minus_mode"] = {
            key: float(full[key]) - float(metrics[key])
            for key in (
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95(M)",
                "metrics/mAP50(B)",
                "metrics/mAP50(M)",
            )
        }
    return {
        "protocol": "same_checkpoint_conditional_branch_masking",
        "independent_retraining": False,
        "interpretation": (
            "diagnostic branch contribution only; this is not a matched "
            "independently trained ablation"
        ),
        "modes": results,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolve_project_path(args.checkpoint)
    targets_path = resolve_project_path(args.sensor_targets)
    data = resolve_project_path(args.data)
    output = resolve_project_path(args.output)
    for path in (checkpoint, targets_path, data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite diagnostics: {output}")

    targets = _read_test_targets(targets_path)
    sensor_metrics, prediction_rows = _predict_sensor_targets(
        checkpoint=checkpoint,
        targets=targets,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        device_name=str(args.device),
    )
    predictions_path = output / "sensor_predictions.jsonl"
    _atomic_write_text(
        predictions_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in prediction_rows
        ),
    )
    masking = None
    if not args.skip_detector_ablation:
        masking = _detector_masking_evaluation(
            checkpoint=checkpoint,
            data=data,
            output=output,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            device=str(args.device),
            workers=int(args.workers),
        )
    report = {
        "schema_version": 1,
        "protocol": "joint_head_rgb_only_forward_audit",
        "valid": True,
        "checkpoint": {
            "path": checkpoint.resolve().as_posix(),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "sensor_target_manifest": {
            "path": targets_path.resolve().as_posix(),
            "size_bytes": targets_path.stat().st_size,
            "sha256": sha256_file(targets_path),
        },
        "sensor_prediction": sensor_metrics,
        "detector_branch_masking": masking,
        "predictions": {
            "path": predictions_path.resolve().as_posix(),
            "size_bytes": predictions_path.stat().st_size,
            "sha256": sha256_file(predictions_path),
            "sample_count": len(prediction_rows),
        },
        "runtime": _runtime_environment(),
        "claim_boundaries": [
            "The metric forward path receives RGB only.",
            "The 4x4 target is used after forward solely for diagnostics.",
            "Branch masking is a same-checkpoint diagnostic, not retraining.",
            "Soft-sensor targets are not hardware measurements.",
        ],
    }
    report_path = output / "evaluation.json"
    _atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> None:
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
