"""Evaluate frozen validation-only calibration on the test teacher split."""

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

from fit_scene_qwen_distiller_calibration import _collect
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
    PROJECT_ROOT / "data" / "object-qwen-scene-teacher" / "test.jsonl"
)
DEFAULT_IMAGES = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "images"
)
DEFAULT_CALIBRATION = ROOT / "CALIBRATION_PARAMETERS.json"
DEFAULT_OUTPUT = ROOT / "CALIBRATION_EVALUATION.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen scene-distiller affine calibration"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument(
        "--calibration", type=Path, default=DEFAULT_CALIBRATION
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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "checkpoint": _resolve(args.checkpoint),
        "test_teacher": _resolve(args.teacher),
        "calibration": _resolve(args.calibration),
        "evaluator_source": Path(__file__).resolve(),
    }
    output = _resolve(args.output)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "calibration evaluation inputs missing: " + ", ".join(missing)
        )
    calibration = json.loads(
        paths["calibration"].read_text(encoding="utf-8-sig")
    )
    if (
        calibration.get("valid") is not True
        or calibration.get("state") != "frozen_before_test_evaluation"
        or calibration.get("test_teacher_read") is not False
        or not all(calibration.get("acceptance", {}).values())
        or calibration.get("bindings", {}).get("checkpoint", {}).get(
            "sha256"
        )
        != _sha256(paths["checkpoint"])
    ):
        raise ValueError("calibration parameters are not frozen and valid")
    device = torch.device(args.device)
    checkpoint = torch.load(
        paths["checkpoint"], map_location=device, weights_only=True
    )
    model = SceneQwenDistiller().to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = SceneTeacherDataset(
        paths["test_teacher"],
        _resolve(args.images_root),
        split="test",
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
    parameters = calibration["parameters"]
    calibrated = np.clip(
        float(parameters["slope"]) * prediction.astype(np.float64)
        + float(parameters["intercept"]),
        0.0,
        1.0,
    )
    mean_grid = checkpoint["train_mean_grid"].cpu().numpy()
    raw_metrics = prediction_metrics(
        prediction,
        teacher,
        coverage,
        train_mean_grid=mean_grid,
    )
    calibrated_metrics = prediction_metrics(
        calibrated,
        teacher,
        coverage,
        train_mean_grid=mean_grid,
    )
    acceptance = {
        "validation_acceptance_preserved": all(
            calibration["acceptance"].values()
        ),
        "test_mse_below_train_mean_grid_baseline": (
            calibrated_metrics["mse"]
            < calibrated_metrics["train_mean_grid_baseline_mse"]
        ),
        "test_elementwise_spearman_above_zero": (
            calibrated_metrics["elementwise_spearman"] > 0.0
        ),
    }
    result = {
        "schema_version": 1,
        "state": "complete" if all(acceptance.values()) else "failed",
        "valid": all(acceptance.values()),
        "formal_paper_metric": False,
        "evidence_role": (
            "post_failure_validation_calibrated_distillation_diagnostic"
        ),
        "parameter_fit_split": "val",
        "evaluation_split": "test",
        "test_used_to_fit_parameters": False,
        "parameters": parameters,
        "validation": calibration["validation"],
        "test": {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
        },
        "acceptance": acceptance,
        "bindings": {name: _binding(path) for name, path in paths.items()},
        "claim_boundary": (
            "The accepted affine calibration recovers absolute scale while "
            "preserving an image-conditioned, nonphysical development sidecar."
        ),
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 对象Qwen整图蒸馏校准评估",
        "",
        f"- 状态：{'通过' if result['valid'] else '未通过'}；正式论文主指标：否。",
        f"- 全局仿射：`clip({parameters['slope']:.6f} * prediction + {parameters['intercept']:.6f}, 0, 1)`。",
        f"- 验证校准MSE：{calibration['validation']['calibrated']['mse']:.6f}；训练均值网格基线：{calibration['validation']['calibrated']['train_mean_grid_baseline_mse']:.6f}。",
        f"- 测试校准MSE：{calibrated_metrics['mse']:.6f}；训练均值网格基线：{calibrated_metrics['train_mean_grid_baseline_mse']:.6f}。",
        f"- 测试Spearman：{calibrated_metrics['elementwise_spearman']:.6f}。",
        "",
        "参数只在验证集拟合并先行冻结，测试集未参与参数拟合。本结果仍是图像条件软sidecar开发诊断。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    if not result["valid"]:
        raise ValueError("frozen affine calibration failed test acceptance")
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
