"""Build the audited paper-readiness statement for the joint Segment head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
JOINT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
)
DEFAULT_OUTPUT = JOINT_ROOT / "PAPER_READINESS.json"
SUMMARY_PATH = JOINT_ROOT / "MULTISEED_SUMMARY.json"
ABLATION_ROOT = JOINT_ROOT / "retrained-ablations"
ABLATION_PATH = ABLATION_ROOT / "OUTCOME.json"
PROTOCOL_PATH = ABLATION_ROOT / "PROTOCOL.json"
DEPLOYMENT_PATH = JOINT_ROOT / "DEPLOYMENT_PROFILE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the joint-head paper-readiness statement"
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


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_readiness(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    summary = _read_json(SUMMARY_PATH)
    ablation = _read_json(ABLATION_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    deployment = _read_json(DEPLOYMENT_PATH)
    if (
        summary.get("valid") is not True
        or summary.get("run_count") != 3
        or summary.get("seeds") != [42, 43, 44]
        or ablation.get("valid") is not True
        or ablation.get("formal_run_count") != 6
        or ablation.get("independent_retraining") is not True
        or protocol.get("status") != "frozen_before_formal_ablation_training"
        or deployment.get("valid") is not True
    ):
        raise ValueError("joint-head evidence is not complete and valid")

    full_box = summary["aggregate"]["metrics/mAP50-95(B)"]
    full_mask = summary["aggregate"]["metrics/mAP50-95(M)"]
    text_metrics = ablation["aggregate"]["text_only"]["metrics"]
    sensor_metrics = ablation["aggregate"]["sensor_only"]["metrics"]
    text_box = text_metrics["metrics/mAP50-95(B)"]
    text_mask = text_metrics["metrics/mAP50-95(M)"]
    sensor_box = sensor_metrics["metrics/mAP50-95(B)"]
    sensor_mask = sensor_metrics["metrics/mAP50-95(M)"]
    sensor_mask_paired = ablation["aggregate"]["sensor_only"][
        "paired_full_minus_mode"
    ]["metrics/mAP50-95(M)"]["full_minus_mode_by_seed"]
    report = {
        "schema_version": 2,
        "status": (
            "internal_joint_head_multiseed_and_independent_ablation_evidence_ready"
            "_with_claim_limits"
        ),
        "valid": True,
        "source_bindings": {
            "multiseed_summary": _binding(SUMMARY_PATH),
            "retrained_ablation_protocol": _binding(PROTOCOL_PATH),
            "retrained_ablation_outcome": _binding(ABLATION_PATH),
            "deployment_profile": _binding(DEPLOYMENT_PATH),
            "builder_source": _binding(Path(__file__)),
        },
        "run_counts": {
            "full_joint_head": 3,
            "text_only": 3,
            "sensor_only": 3,
            "independently_retrained_branch_controls": 6,
        },
        "seeds": [42, 43, 44],
        "primary_result": {
            "box_map50_95_mean": full_box["mean"],
            "box_map50_95_sample_std": full_box["sample_std"],
            "mask_map50_95_mean": full_mask["mean"],
            "mask_map50_95_sample_std": full_mask["sample_std"],
            "box_mean_delta_percentage_points_vs_rgb": (
                full_box["mean_minus_rgb_baseline_percentage_points"]
            ),
            "mask_mean_delta_percentage_points_vs_rgb": (
                full_mask["mean_minus_rgb_baseline_percentage_points"]
            ),
        },
        "independent_branch_ablation": {
            "text_only": {
                "box_mean": text_box["mean"],
                "mask_mean": text_mask["mean"],
                "box_delta_percentage_points_vs_rgb": (
                    text_box["mean_minus_rgb_baseline_percentage_points"]
                ),
                "mask_delta_percentage_points_vs_rgb": (
                    text_mask["mean_minus_rgb_baseline_percentage_points"]
                ),
                "full_minus_text_box_percentage_points": (
                    text_box["full_mean_minus_mode_mean_percentage_points"]
                ),
                "full_minus_text_mask_percentage_points": (
                    text_mask["full_mean_minus_mode_mean_percentage_points"]
                ),
            },
            "sensor_only": {
                "box_mean": sensor_box["mean"],
                "mask_mean": sensor_mask["mean"],
                "box_delta_percentage_points_vs_rgb": (
                    sensor_box["mean_minus_rgb_baseline_percentage_points"]
                ),
                "mask_delta_percentage_points_vs_rgb": (
                    sensor_mask["mean_minus_rgb_baseline_percentage_points"]
                ),
                "full_minus_sensor_box_percentage_points": (
                    sensor_box["full_mean_minus_mode_mean_percentage_points"]
                ),
                "full_minus_sensor_mask_percentage_points": (
                    sensor_mask["full_mean_minus_mode_mean_percentage_points"]
                ),
                "full_minus_sensor_mask_by_seed": sensor_mask_paired,
                "full_minus_sensor_mask_positive_seed_count": sum(
                    value > 0.0 for value in sensor_mask_paired
                ),
            },
        },
        "evidence_status": {
            "three_independent_full_training_seeds": True,
            "six_independently_retrained_branch_controls": True,
            "same_checkpoint_masking_used_as_formal_ablation": False,
            "formal_metric_forward_is_rgb_only": True,
            "both_single_branches_improve_map50_95_over_rgb_mean": (
                text_box["mean_minus_rgb_baseline"] > 0.0
                and text_mask["mean_minus_rgb_baseline"] > 0.0
                and sensor_box["mean_minus_rgb_baseline"] > 0.0
                and sensor_mask["mean_minus_rgb_baseline"] > 0.0
            ),
            "stable_positive_text_synergy_over_sensor_only_mask": False,
            "external_blind_test": False,
            "hardware_sensor_measurement": False,
            "rgb_d_input": False,
            "full_patent_reproduction": False,
        },
        "deployment_profile": {
            "rgb_parameter_count": deployment[
                "checkpoint_parameters_unfused"
            ]["rgb"],
            "joint_parameter_count": deployment[
                "checkpoint_parameters_unfused"
            ]["joint_per_seed"],
            "joint_added_parameter_percent": deployment[
                "checkpoint_parameters_unfused"
            ]["joint_added_percent_of_rgb"],
            "joint_inference_mean_ms_per_image": deployment[
                "speed_ms_per_image"
            ]["joint_three_seed_summary"]["inference"]["mean"],
            "joint_inference_mean_overhead_percent": deployment[
                "speed_ms_per_image"
            ]["joint_inference_mean_overhead_percent"],
            "amortized_validator_speed_not_single_sample_latency": True,
        },
        "recommended_wording": (
            "在固定 source-grouped 内部测试集的三个独立训练种子上，文本分支和"
            "图像条件软传感器分支独立重训后均高于 RGB 基线。完整联合头相对仅文本"
            "变体的 box/mask mAP50-95 均值变化为 +0.038/+0.094 个百分点，相对"
            "仅软传感器变体为 +0.020/+0.007 个百分点；后者 mask 差异在种子间"
            "符号不一致。因此结果支持两个分支各自有效及完整头的小幅性能变化，"
            "不支持强协同、统计显著性、外部泛化或真实硬件传感器主张。"
        ),
    }
    _atomic_write(output_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = build_readiness(args.output)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": report["valid"],
                "output": _resolve(args.output).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
