"""Freeze an additive validation-only calibration recovery after raw failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT


ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "object-qwen-scene-distillation"
)
DEFAULT_OUTPUT = ROOT / "CALIBRATION_RECOVERY_PROTOCOL.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the scene-distiller calibration recovery"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve() if output.is_absolute() else (PROJECT_ROOT / output).resolve()
    paths = {
        "original_protocol": ROOT / "PROTOCOL.json",
        "failed_training_report": ROOT / "TRAINING.json",
        "best_checkpoint": (
            PROJECT_ROOT / "checkpoints" / "scene-qwen-distiller-v1" / "best.pt"
        ),
        "val_teacher": (
            PROJECT_ROOT / "data" / "object-qwen-scene-teacher" / "val.jsonl"
        ),
        "test_teacher": (
            PROJECT_ROOT / "data" / "object-qwen-scene-teacher" / "test.jsonl"
        ),
        "fit_source": (
            PROJECT_ROOT / "fit_scene_qwen_distiller_calibration.py"
        ),
        "evaluation_source": (
            PROJECT_ROOT / "evaluate_scene_qwen_distiller_calibration.py"
        ),
        "revised_predictor_source": (
            PROJECT_ROOT / "predict_scene_qwen_distiller.py"
        ),
        "recovery_builder": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "calibration recovery inputs missing: " + ", ".join(missing)
        )
    failed = json.loads(
        paths["failed_training_report"].read_text(encoding="utf-8-sig")
    )
    if (
        failed.get("valid") is not False
        or failed.get("state") != "failed_acceptance"
        or failed.get("acceptance", {}).get(
            "validation_beats_train_mean_grid_baseline"
        )
        is not False
        or failed.get("acceptance", {}).get(
            "test_beats_train_mean_grid_baseline"
        )
        is not False
        or failed.get("acceptance", {}).get(
            "validation_positive_spearman"
        )
        is not True
        or failed.get("acceptance", {}).get("test_positive_spearman")
        is not True
    ):
        raise ValueError("raw distillation failure contract differs")
    result = {
        "schema_version": 1,
        "state": "frozen_before_calibration_fit_and_test_evaluation",
        "valid": True,
        "formal_paper_metric": False,
        "recovery_role": "post_raw_scale_failure_validation_only_calibration",
        "original_failure_preserved": True,
        "observed_before_recovery": {
            "validation_raw_mse": failed["validation"]["mse"],
            "validation_baseline_mse": failed["validation"][
                "train_mean_grid_baseline_mse"
            ],
            "validation_spearman": failed["validation"][
                "elementwise_spearman"
            ],
            "test_raw_mse": failed["test"]["mse"],
            "test_baseline_mse": failed["test"][
                "train_mean_grid_baseline_mse"
            ],
            "test_spearman": failed["test"]["elementwise_spearman"],
        },
        "calibration": {
            "method": (
                "global_nonnegative_affine_least_squares_then_clip_0_1"
            ),
            "formula": "clip(a * raw_prediction + b, 0, 1)",
            "fit_split": "val",
            "slope_bounds": [0.0, 5.0],
            "intercept_bounds": [-1.0, 1.0],
            "test_read_during_fit": False,
            "parameter_file_frozen_before_test": True,
        },
        "acceptance": {
            "validation_calibrated_mse_below_train_mean_grid_baseline": True,
            "test_calibrated_mse_below_train_mean_grid_baseline": True,
            "validation_and_test_spearman_above_zero": True,
        },
        "post_acceptance": {
            "sidecar_generation_applies_frozen_calibration": True,
            "joint_head_protocol_still_separate": True,
        },
        "bindings": {name: _binding(path) for name, path in paths.items()},
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 对象Qwen整图蒸馏校准恢复协议",
        "",
        "- 原始40轮学生训练与MSE接受失败保持不变。",
        "- 仅允许在验证集拟合一个非负全局仿射参数，并裁剪至[0,1]。",
        "- 参数文件必须先于测试评估落盘冻结；测试集不参与拟合。",
        "- 校准后验证和测试MSE均须优于训练均值网格基线，否则终止。",
        "- 本恢复仍是开发证据，不修改正式整图Qwen协议。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return result


def main() -> None:
    print(json.dumps(build(parse_args().output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
