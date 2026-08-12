"""Summarize parameter and validation-speed evidence for the joint head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
JOINT_ROOT = REPORTS_ROOT / "joint-multimodal-seg"
DEFAULT_OUTPUT = JOINT_ROOT / "DEPLOYMENT_PROFILE.json"
EVALUATIONS = {
    "rgb": REPORTS_ROOT / "yolo-seg" / "evaluation.json",
    **{
        f"joint_seed{seed}": (
            JOINT_ROOT
            / f"yolo11s-seg-joint-v1-seed{seed}"
            / "evaluation.json"
        )
        for seed in (42, 43, 44)
    },
}
CHECKPOINTS = {
    "rgb": (
        PROJECT_ROOT
        / "checkpoints"
        / "yolo-seg"
        / "yolo11s-seg-qwen-split"
        / "weights"
        / "best.pt"
    ),
    **{
        f"joint_seed{seed}": (
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg"
            / f"yolo11s-seg-joint-v1-seed{seed}"
            / "weights"
            / "best.pt"
        )
        for seed in (42, 43, 44)
    },
}
RUNTIME_KEYS = ("python", "torch", "ultralytics", "cuda_runtime", "gpu")
SPEED_KEYS = ("preprocess", "inference", "postprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the joint-head deployment profile"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _parameter_counts() -> dict[str, int]:
    from ultralytics import YOLO

    counts: dict[str, int] = {}
    for name, path in CHECKPOINTS.items():
        model = YOLO(str(path)).model
        counts[name] = int(sum(parameter.numel() for parameter in model.parameters()))
    return counts


def _summary(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("speed values must be finite and positive")
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_profile(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    parameter_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    output_path = _resolve(output_path)
    documents = {name: _read_json(path) for name, path in EVALUATIONS.items()}
    baseline = documents["rgb"]
    joint_names = [f"joint_seed{seed}" for seed in (42, 43, 44)]
    baseline_runtime = baseline.get("runtime", {})
    if not isinstance(baseline_runtime, Mapping):
        raise ValueError("RGB runtime metadata is missing")
    for name in joint_names:
        document = documents[name]
        runtime = document.get("runtime", {})
        if any(runtime.get(key) != baseline_runtime.get(key) for key in RUNTIME_KEYS):
            raise ValueError(f"runtime mismatch: {name}")
        if document.get("forward_contract", {}).get("metric_forward_inputs") != [
            "RGB"
        ]:
            raise ValueError(f"formal metric forward is not RGB-only: {name}")

    speed_rows: dict[str, dict[str, float]] = {}
    for name, document in documents.items():
        raw = document.get("speed_ms_per_image")
        if not isinstance(raw, Mapping):
            raise ValueError(f"speed metadata is missing: {name}")
        row = {key: float(raw[key]) for key in SPEED_KEYS}
        if not all(math.isfinite(value) and value > 0.0 for value in row.values()):
            raise ValueError(f"invalid speed metadata: {name}")
        row["pipeline_without_loss"] = sum(row.values())
        speed_rows[name] = row

    counts = dict(parameter_counts or _parameter_counts())
    if set(counts) != set(CHECKPOINTS):
        raise ValueError("parameter-count labels do not match checkpoints")
    if any(not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError("parameter counts must be positive integers")
    joint_counts = [counts[name] for name in joint_names]
    if len(set(joint_counts)) != 1:
        raise ValueError("joint checkpoint parameter counts differ by seed")
    joint_parameter_count = joint_counts[0]
    rgb_parameter_count = counts["rgb"]
    added_parameters = joint_parameter_count - rgb_parameter_count
    declared_joint_counts = {
        int(documents[name].get("parameters", {}).get("joint_fusion", -1))
        for name in joint_names
    }
    if declared_joint_counts != {added_parameters}:
        raise ValueError("joint parameter delta differs from recorded adapter count")

    joint_speed = {
        key: _summary([speed_rows[name][key] for name in joint_names])
        for key in (*SPEED_KEYS, "pipeline_without_loss")
    }
    rgb_inference = speed_rows["rgb"]["inference"]
    rgb_pipeline = speed_rows["rgb"]["pipeline_without_loss"]
    report = {
        "schema_version": 1,
        "valid": True,
        "scope": "descriptive_ultralytics_validation_speed_and_parameters",
        "comparison_identity": {
            "environment_equal": True,
            "runtime": {
                key: baseline_runtime.get(key) for key in RUNTIME_KEYS
            },
            "test_source_images": 570,
            "test_objects": 825,
            "image_size": 640,
            "joint_seeds": [42, 43, 44],
            "formal_metric_forward_inputs": ["RGB"],
        },
        "checkpoint_parameters_unfused": {
            "rgb": rgb_parameter_count,
            "joint_per_seed": joint_parameter_count,
            "joint_added": added_parameters,
            "joint_added_percent_of_rgb": (
                100.0 * added_parameters / rgb_parameter_count
            ),
        },
        "speed_ms_per_image": {
            "rgb_single_validation_run": speed_rows["rgb"],
            "joint_three_seed_summary": joint_speed,
            "joint_inference_mean_overhead_ms": (
                joint_speed["inference"]["mean"] - rgb_inference
            ),
            "joint_inference_mean_overhead_percent": (
                100.0
                * (joint_speed["inference"]["mean"] - rgb_inference)
                / rgb_inference
            ),
            "joint_pipeline_mean_overhead_ms": (
                joint_speed["pipeline_without_loss"]["mean"] - rgb_pipeline
            ),
            "joint_pipeline_mean_overhead_percent": (
                100.0
                * (
                    joint_speed["pipeline_without_loss"]["mean"]
                    - rgb_pipeline
                )
                / rgb_pipeline
            ),
        },
        "interpretation": {
            "parameter_overhead_is_adapter_only": True,
            "latency_is_amortized_validator_speed": True,
            "single_sample_latency_claim_supported": False,
            "patent_15_2_ms_reproduction_claim_supported": False,
            "note": (
                "Ultralytics validation speed is amortized over the fixed test "
                "run. It is suitable for a same-environment descriptive "
                "comparison, not a controlled single-image latency benchmark."
            ),
        },
        "source_bindings": {
            "builder_source": _binding(Path(__file__)),
            **{
                f"evaluation_{name}": _binding(path)
                for name, path in EVALUATIONS.items()
            },
            **{
                f"checkpoint_{name}": _binding(path)
                for name, path in CHECKPOINTS.items()
            },
        },
    }
    _atomic_write(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 联合头参数量与验证速度",
        "",
        "- 状态：有效",
        "- 环境：NVIDIA GeForce RTX 4090，PyTorch 2.6.0+cu126，Ultralytics 8.4.70",
        "- 正式前向输入：RGB",
        "",
        "| 指标 | RGB | 完整联合头 | 变化 |",
        "|---|---:|---:|---:|",
        (
            f"| 未融合检查点参数量 | {rgb_parameter_count:,} | "
            f"{joint_parameter_count:,} | +{added_parameters:,} "
            f"({100.0 * added_parameters / rgb_parameter_count:.2f}%) |"
        ),
        (
            f"| 推理 ms/图 | {rgb_inference:.3f} | "
            f"{joint_speed['inference']['mean']:.3f}±"
            f"{joint_speed['inference']['sample_std']:.3f} | "
            f"+{report['speed_ms_per_image']['joint_inference_mean_overhead_percent']:.2f}% |"
        ),
        (
            f"| 预处理+推理+后处理 ms/图 | {rgb_pipeline:.3f} | "
            f"{joint_speed['pipeline_without_loss']['mean']:.3f}±"
            f"{joint_speed['pipeline_without_loss']['sample_std']:.3f} | "
            f"+{report['speed_ms_per_image']['joint_pipeline_mean_overhead_percent']:.2f}% |"
        ),
        "",
        (
            "速度来自相同环境、固定 570 张测试源图上的 Ultralytics 验证器摊销值。"
            "RGB 只有一次正式验证，联合头统计来自种子 42/43/44；这不是受控的"
            "单样本延迟实验，不能用于宣称复现专利中的 15.2 ms。"
        ),
        "",
    ]
    _atomic_write(output_path.with_suffix(".md"), "\n".join(markdown))
    return report


def main() -> None:
    args = parse_args()
    report = build_profile(args.output)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": report["valid"],
                "output": _resolve(args.output).as_posix(),
                "joint_added_parameters": report[
                    "checkpoint_parameters_unfused"
                ]["joint_added"],
                "joint_inference_overhead_percent": report[
                    "speed_ms_per_image"
                ]["joint_inference_mean_overhead_percent"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
