"""Build the auditable addendum for the object-Qwen-distilled v6 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from train_external_sensor_joint_seg import _atomic_write_text
from train_patent_aligned_qwen_distilled import (
    DEFAULT_NAME,
    DEFAULT_PROJECT,
    DEFAULT_REPORT,
    REGISTERED_PROTOCOL_NAME,
    RUN_PROTOCOL_NAME,
)


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
RUN_REPORT = DEFAULT_REPORT / DEFAULT_NAME
RUN_CHECKPOINT = DEFAULT_PROJECT / DEFAULT_NAME / "weights" / "best.pt"
REGISTERED_PROTOCOL = DEFAULT_REPORT / "DEVELOPMENT_PROTOCOL.json"
RUN_PROTOCOL = RUN_REPORT / "PROTOCOL.json"
EVALUATION = RUN_REPORT / "evaluation.json"
CONDITIONS = RUN_REPORT / "RELIABILITY_CONDITIONS.json"
DISTILLATION_REPORT = (
    REPORTS / "object-qwen-scene-distillation"
)
CALIBRATION_EVALUATION = (
    DISTILLATION_REPORT / "CALIBRATION_EVALUATION.json"
)
FEASIBILITY = (
    REPORTS / "OBJECT_QWEN_SCENE_DISTILLATION_FEASIBILITY.json"
)
SCENE_BUILD = (
    PROJECT_ROOT
    / "data"
    / "scene-qwen-distilled-sidecars"
    / "BUILD_REPORT.json"
)
V5_CONDITIONS = (
    REPORTS
    / "joint-multimodal-seg-spatial"
    / "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
    / "RELIABILITY_CONDITIONS.json"
)
V6_RULE_CONDITIONS = (
    REPORTS
    / "joint-multimodal-seg-patent"
    / "yolo11s-seg-joint-patent-stack-rule-dev-seed42"
    / "RELIABILITY_CONDITIONS.json"
)
DEFAULT_OUTPUT = REPORTS / "OBJECT_QWEN_DISTILLED_JOINT_ADDENDUM.json"
INPUTS = {
    "registered_protocol": REGISTERED_PROTOCOL,
    "run_protocol": RUN_PROTOCOL,
    "evaluation": EVALUATION,
    "conditions": CONDITIONS,
    "checkpoint": RUN_CHECKPOINT,
    "calibration_evaluation": CALIBRATION_EVALUATION,
    "feasibility_audit": FEASIBILITY,
    "scene_build": SCENE_BUILD,
    "v5_rule_conditions": V5_CONDITIONS,
    "v6_rule_conditions": V6_RULE_CONDITIONS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the object-Qwen-distilled joint-model addendum"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _metric(condition: dict[str, Any], modality: str) -> float:
    return float(
        condition["metrics"][f"metrics/mAP50-95({modality})"]
    )


def _condition_summary(report: dict[str, Any]) -> dict[str, Any]:
    clean = report["conditions"]["clean_matched"]
    result: dict[str, Any] = {}
    for name, condition in report["conditions"].items():
        result[name] = {
            "box_map50_95": _metric(condition, "B"),
            "mask_map50_95": _metric(condition, "M"),
            "clean_minus_condition_box": (
                _metric(clean, "B") - _metric(condition, "B")
            ),
            "clean_minus_condition_mask": (
                _metric(clean, "M") - _metric(condition, "M")
            ),
            "raw_reliability_mean": float(
                condition["gate"]["sensor_raw_reliability"]["mean"]
            ),
            "effective_reliability_mean": float(
                condition["gate"]["sensor_reliability"]["mean"]
            ),
        }
    return result


def _matched_summary(report: dict[str, Any]) -> dict[str, float]:
    clean = report["conditions"]["clean_matched"]
    rgb = report["conditions"]["rgb_fallback"]
    shuffled = report["conditions"]["shuffled"]
    return {
        "box_map50_95": _metric(clean, "B"),
        "mask_map50_95": _metric(clean, "M"),
        "clean_minus_rgb_box": _metric(clean, "B") - _metric(rgb, "B"),
        "clean_minus_rgb_mask": _metric(clean, "M") - _metric(rgb, "M"),
        "clean_minus_shuffled_box": (
            _metric(clean, "B") - _metric(shuffled, "B")
        ),
        "clean_minus_shuffled_mask": (
            _metric(clean, "M") - _metric(shuffled, "M")
        ),
    }


def build() -> dict[str, Any]:
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("addendum inputs missing: " + ", ".join(missing))

    registered = _read(REGISTERED_PROTOCOL)
    run_protocol = _read(RUN_PROTOCOL)
    evaluation = _read(EVALUATION)
    conditions = _read(CONDITIONS)
    calibration = _read(CALIBRATION_EVALUATION)
    feasibility = _read(FEASIBILITY)
    scene_build = _read(SCENE_BUILD)
    v5_conditions = _read(V5_CONDITIONS)
    v6_rule_conditions = _read(V6_RULE_CONDITIONS)
    checkpoint_sha = sha256_file(RUN_CHECKPOINT)

    if (
        registered.get("protocol") != REGISTERED_PROTOCOL_NAME
        or registered.get("formal_paper_metric") is not False
        or run_protocol.get("protocol") != RUN_PROTOCOL_NAME
        or run_protocol.get("formal_paper_metric") is not False
        or evaluation.get("valid") is not True
        or evaluation.get("formal_paper_metric") is not False
        or conditions.get("valid") is not True
        or conditions.get("formal_paper_metric") is not False
        or evaluation.get("checkpoint", {}).get("sha256") != checkpoint_sha
        or conditions.get("bindings", {})
        .get("checkpoint", {})
        .get("sha256")
        != checkpoint_sha
        or calibration.get("valid") is not True
        or scene_build.get("valid") is not True
        or scene_build.get("source_contract", {}).get(
            "annotation_files_read"
        )
        is not False
        or feasibility.get("valid") is not True
    ):
        raise ValueError("object-Qwen-distilled evidence bindings are invalid")

    distilled = _matched_summary(conditions)
    v5 = _matched_summary(v5_conditions)
    v6_rule = _matched_summary(v6_rule_conditions)
    equivalence_tolerance = 0.0005
    accuracy_contribution = any(
        distilled[key] > equivalence_tolerance
        for key in (
            "clean_minus_rgb_box",
            "clean_minus_rgb_mask",
            "clean_minus_shuffled_box",
            "clean_minus_shuffled_mask",
        )
    )
    test_calibrated = calibration["test"]["calibrated"]
    test_baseline_mse = float(
        test_calibrated["train_mean_grid_baseline_mse"]
    )
    grid_mse_reduction = 1.0 - (
        float(test_calibrated["mse"]) / test_baseline_mse
    )
    training = evaluation["training"]

    return {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_role": (
            "object_qwen_distilled_patent_stack_development_negative_result"
        ),
        "data_lineage": {
            "object_teacher_sample_count": int(
                scene_build["object_qwen_teacher_api_success_count"]
                + scene_build["object_qwen_teacher_fallback_count"]
            ),
            "object_qwen_success_count": int(
                scene_build["object_qwen_teacher_api_success_count"]
            ),
            "object_rule_fallback_count": int(
                scene_build["object_qwen_teacher_fallback_count"]
            ),
            "source_image_count": int(scene_build["source_image_count"]),
            "teacher_uses_annotations": True,
            "generated_scene_sidecar_uses_annotations": False,
            "joint_inference_uses_annotations": False,
            "direct_full_image_qwen_api": False,
            "physical_sensor_measurement": False,
        },
        "scene_grid_diagnostic": {
            "test_calibrated_mse": float(test_calibrated["mse"]),
            "test_mean_grid_baseline_mse": test_baseline_mse,
            "relative_mse_reduction": grid_mse_reduction,
            "test_spearman": float(
                test_calibrated["elementwise_spearman"]
            ),
            "test_positive_iou": float(
                test_calibrated["positive_iou_at_0_05"]
            ),
            "calibration_fit_split": "val",
            "test_used_for_calibration_fit": False,
        },
        "joint_training": {
            "registered_epochs_max": int(
                registered["training"]["epochs"]
            ),
            "completed_epochs": int(training["completed_epochs"]),
            "ultralytics_best_epoch": 6,
            "early_stopping_patience": int(
                registered["training"]["patience"]
            ),
            "checkpoint": _binding(RUN_CHECKPOINT),
        },
        "joint_test": {
            "test_source_image_count": int(
                conditions["test_source_image_count"]
            ),
            "conditions": _condition_summary(conditions),
            "gate_ordering_all_passed": bool(
                conditions["gate_ordering_all_passed"]
            ),
            "metric_equivalence_tolerance": equivalence_tolerance,
            "matched_sensor_accuracy_contribution_detected": (
                accuracy_contribution
            ),
        },
        "development_comparison": {
            "object_qwen_distilled_v6": distilled,
            "rule_sidecar_v5": v5,
            "rule_sidecar_v6": v6_rule,
            "distilled_minus_rule_v5": {
                "box_map50_95": (
                    distilled["box_map50_95"] - v5["box_map50_95"]
                ),
                "mask_map50_95": (
                    distilled["mask_map50_95"] - v5["mask_map50_95"]
                ),
            },
            "distilled_minus_rule_v6": {
                "box_map50_95": (
                    distilled["box_map50_95"] - v6_rule["box_map50_95"]
                ),
                "mask_map50_95": (
                    distilled["mask_map50_95"] - v6_rule["mask_map50_95"]
                ),
            },
            "comparison_scope": (
                "single-seed development diagnostics; no significance claim"
            ),
        },
        "interpretation": {
            "scene_grid_is_reconstructable_beyond_mean_baseline": True,
            "reliability_gate_discriminates_registered_corruptions": bool(
                conditions["gate_ordering_all_passed"]
            ),
            "independent_segmentation_information_demonstrated": False,
            "supports_compatibility_claim": True,
            "supports_accuracy_improvement_claim": False,
            "recommended_manuscript_use": (
                "Report as a transparent negative development result: the "
                "RGB-distilled Qwen scene grid is recoverable and accepted "
                "by the reliability stack, but is redundant with the RGB "
                "backbone for detection and segmentation accuracy."
            ),
        },
        "remaining_formal_requirement": (
            "Execute the unchanged preregistered three-seed direct "
            "full-image Qwen protocol after credentials become available."
        ),
        "bindings": {
            **{name: _binding(path) for name, path in INPUTS.items()},
            "builder_source": _binding(Path(__file__)),
        },
    }


def _pp(value: float) -> str:
    return f"{value * 100:+.4f}"


def write(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    report = build()
    _atomic_write_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    grid = report["scene_grid_diagnostic"]
    joint = report["joint_test"]
    comparisons = report["development_comparison"]
    distilled = comparisons["object_qwen_distilled_v6"]
    rows = []
    for name in ("clean_matched", "rgb_fallback", "shuffled", "missing",
                 "noise_0_15", "bias_drift_0_25", "stuck"):
        value = joint["conditions"][name]
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{value['box_map50_95']:.6f}",
                    f"{value['mask_map50_95']:.6f}",
                    _pp(value["clean_minus_condition_box"]),
                    _pp(value["clean_minus_condition_mask"]),
                    f"{value['raw_reliability_mean']:.4f}",
                ]
            )
            + " |"
        )
    markdown = [
        "# 目标级 Qwen 蒸馏侧车联合实验附录",
        "",
        "## 结论",
        "",
        (
            "该实验是单种子开发证据，不是正式论文指标。整图 RGB 学生"
            "能够重建目标级 Qwen 教师的场景网格，但加入 v6 后没有产生"
            "可测的检测或分割增益，因此只支持链路兼容性，不支持精度提升"
            "主张。"
        ),
        "",
        "## 网格诊断",
        "",
        (
            f"- 测试 MSE：{grid['test_calibrated_mse']:.6f}；均值基线："
            f"{grid['test_mean_grid_baseline_mse']:.6f}；相对下降："
            f"{grid['relative_mse_reduction'] * 100:.2f}%。"
        ),
        (
            f"- Spearman：{grid['test_spearman']:.4f}；正值区域 IoU："
            f"{grid['test_positive_iou']:.4f}。"
        ),
        "- 校准参数只在验证集拟合，测试教师未参与拟合。",
        "",
        "## 联合模型测试",
        "",
        "| 条件 | Box mAP50-95 | Mask mAP50-95 | Clean差值Box/pp | Clean差值Mask/pp | 原始可靠性 |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        (
            f"matched 相对 RGB fallback：Box "
            f"{_pp(distilled['clean_minus_rgb_box'])} pp，Mask "
            f"{_pp(distilled['clean_minus_rgb_mask'])} pp；相对 shuffled："
            f"Box {_pp(distilled['clean_minus_shuffled_box'])} pp，Mask "
            f"{_pp(distilled['clean_minus_shuffled_mask'])} pp。"
        ),
        "",
        "## 表述边界",
        "",
        "- 教师构建使用目标标注，但最终侧车生成和联合推理不读取标注。",
        "- 这是目标级 Qwen 蒸馏结果，不是整图 Qwen API 直接观测。",
        "- 数据是图像条件软传感器估计，不是物理传感器实测。",
        "- 正式结论仍需执行已冻结的三种子整图 Qwen 协议。",
        "",
    ]
    _atomic_write_text(output.with_suffix(".md"), "\n".join(markdown))
    return report


def main() -> None:
    report = write(parse_args().output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "formal_paper_metric": report["formal_paper_metric"],
                "grid_relative_mse_reduction": report[
                    "scene_grid_diagnostic"
                ]["relative_mse_reduction"],
                "gate_ordering_all_passed": report["joint_test"][
                    "gate_ordering_all_passed"
                ],
                "accuracy_contribution_detected": report["joint_test"][
                    "matched_sensor_accuracy_contribution_detected"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
