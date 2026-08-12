"""Build the evidence index for the object-Qwen-distilled development path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from train_external_sensor_joint_seg import _atomic_write_text
from train_patent_aligned_qwen_distilled import DEFAULT_NAME, DEFAULT_REPORT


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
DISTILLATION = REPORTS / "object-qwen-scene-distillation"
RUN_REPORT = DEFAULT_REPORT / DEFAULT_NAME
SIDECARS = PROJECT_ROOT / "data" / "scene-qwen-distilled-sidecars"
DEFAULT_OUTPUT = (
    REPORTS / "OBJECT_QWEN_DISTILLED_EXTENSION_INDEX.json"
)
EVIDENCE = {
    "feasibility_json": (
        REPORTS / "OBJECT_QWEN_SCENE_DISTILLATION_FEASIBILITY.json"
    ),
    "feasibility_md": (
        REPORTS / "OBJECT_QWEN_SCENE_DISTILLATION_FEASIBILITY.md"
    ),
    "distillation_protocol_json": DISTILLATION / "PROTOCOL.json",
    "distillation_protocol_md": DISTILLATION / "PROTOCOL.md",
    "distillation_training": DISTILLATION / "TRAINING.json",
    "calibration_recovery_protocol_json": (
        DISTILLATION / "CALIBRATION_RECOVERY_PROTOCOL.json"
    ),
    "calibration_recovery_protocol_md": (
        DISTILLATION / "CALIBRATION_RECOVERY_PROTOCOL.md"
    ),
    "calibration_parameters": DISTILLATION / "CALIBRATION_PARAMETERS.json",
    "calibration_evaluation_json": (
        DISTILLATION / "CALIBRATION_EVALUATION.json"
    ),
    "calibration_evaluation_md": (
        DISTILLATION / "CALIBRATION_EVALUATION.md"
    ),
    "scene_sidecar_build": SIDECARS / "BUILD_REPORT.json",
    "scene_sidecar_validation": SIDECARS / "VALIDATION_REPORT.json",
    "scene_sidecar_shuffle": SIDECARS / "SHUFFLE_REPORT.json",
    "joint_development_protocol_json": (
        DEFAULT_REPORT / "DEVELOPMENT_PROTOCOL.json"
    ),
    "joint_development_protocol_md": (
        DEFAULT_REPORT / "DEVELOPMENT_PROTOCOL.md"
    ),
    "joint_preflight_protocol": DEFAULT_REPORT / "PREFLIGHT_PROTOCOL.json",
    "joint_run_protocol": RUN_REPORT / "PROTOCOL.json",
    "joint_evaluation": RUN_REPORT / "evaluation.json",
    "joint_conditions": RUN_REPORT / "RELIABILITY_CONDITIONS.json",
    "joint_addendum_json": (
        REPORTS / "OBJECT_QWEN_DISTILLED_JOINT_ADDENDUM.json"
    ),
    "joint_addendum_md": (
        REPORTS / "OBJECT_QWEN_DISTILLED_JOINT_ADDENDUM.md"
    ),
    "feasibility_source": (
        PROJECT_ROOT / "audit_object_qwen_scene_distillation_feasibility.py"
    ),
    "teacher_builder_source": (
        PROJECT_ROOT / "build_object_qwen_scene_teacher.py"
    ),
    "distillation_module_source": (
        PROJECT_ROOT / "mmfc" / "scene_distillation.py"
    ),
    "distillation_trainer_source": (
        PROJECT_ROOT / "train_scene_qwen_distiller.py"
    ),
    "distillation_predictor_source": (
        PROJECT_ROOT / "predict_scene_qwen_distiller.py"
    ),
    "calibration_fitter_source": (
        PROJECT_ROOT / "fit_scene_qwen_distiller_calibration.py"
    ),
    "calibration_evaluator_source": (
        PROJECT_ROOT / "evaluate_scene_qwen_distiller_calibration.py"
    ),
    "derangement_source": (
        PROJECT_ROOT / "prepare_object_qwen_scene_derangement.py"
    ),
    "joint_protocol_builder_source": (
        PROJECT_ROOT / "prepare_object_qwen_scene_joint_protocol.py"
    ),
    "joint_trainer_source": (
        PROJECT_ROOT / "train_patent_aligned_qwen_distilled.py"
    ),
    "joint_evaluator_source": (
        PROJECT_ROOT
        / "evaluate_patent_aligned_qwen_distilled_conditions.py"
    ),
    "joint_addendum_builder_source": (
        PROJECT_ROOT / "build_object_qwen_distilled_joint_addendum.py"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the distilled-Qwen development evidence index"
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


def build() -> dict[str, Any]:
    missing = [name for name, path in EVIDENCE.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "distilled extension evidence missing: " + ", ".join(missing)
        )
    addendum = _read(EVIDENCE["joint_addendum_json"])
    calibration = _read(EVIDENCE["calibration_evaluation_json"])
    conditions = _read(EVIDENCE["joint_conditions"])
    scene_build = _read(EVIDENCE["scene_sidecar_build"])
    if (
        addendum.get("valid") is not True
        or addendum.get("formal_paper_metric") is not False
        or addendum.get("interpretation", {}).get(
            "supports_accuracy_improvement_claim"
        )
        is not False
        or calibration.get("valid") is not True
        or conditions.get("valid") is not True
        or scene_build.get("valid") is not True
    ):
        raise ValueError("distilled extension evidence is not internally valid")

    return {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "object_qwen_distilled_scene_sidecar_development_extension",
        "readiness": {
            "object_qwen_teacher_audit": "complete",
            "annotation_free_scene_sidecar_inference": "complete",
            "scene_grid_calibration": "complete",
            "v6_joint_training": "complete",
            "registered_condition_evaluation": "complete",
            "reliability_gate_discrimination": "demonstrated",
            "segmentation_accuracy_gain": "not_demonstrated",
            "direct_full_image_qwen_three_seed_study": (
                "preregistered_awaiting_credentials"
            ),
            "physical_sensor_validation": "not_implemented",
            "external_blind_validation": "not_implemented",
        },
        "headline_metrics": {
            "grid_test_mse": addendum["scene_grid_diagnostic"][
                "test_calibrated_mse"
            ],
            "grid_test_mean_baseline_mse": addendum[
                "scene_grid_diagnostic"
            ]["test_mean_grid_baseline_mse"],
            "grid_relative_mse_reduction": addendum[
                "scene_grid_diagnostic"
            ]["relative_mse_reduction"],
            "grid_test_spearman": addendum["scene_grid_diagnostic"][
                "test_spearman"
            ],
            "joint_clean_box_map50_95": addendum[
                "development_comparison"
            ]["object_qwen_distilled_v6"]["box_map50_95"],
            "joint_clean_mask_map50_95": addendum[
                "development_comparison"
            ]["object_qwen_distilled_v6"]["mask_map50_95"],
            "clean_minus_rgb_box": addendum[
                "development_comparison"
            ]["object_qwen_distilled_v6"]["clean_minus_rgb_box"],
            "clean_minus_rgb_mask": addendum[
                "development_comparison"
            ]["object_qwen_distilled_v6"]["clean_minus_rgb_mask"],
            "clean_minus_shuffled_box": addendum[
                "development_comparison"
            ]["object_qwen_distilled_v6"]["clean_minus_shuffled_box"],
            "clean_minus_shuffled_mask": addendum[
                "development_comparison"
            ]["object_qwen_distilled_v6"]["clean_minus_shuffled_mask"],
        },
        "claim_decision": {
            "may_claim": [
                (
                    "An object-Qwen teacher can supervise an annotation-free "
                    "RGB scene-grid student in this dataset."
                ),
                (
                    "The v6 reliability gate ranks clean distilled sidecars "
                    "above registered observed corruptions."
                ),
                (
                    "The distilled sidecar path is compatible with the "
                    "patent-aligned native detection/segmentation stack."
                ),
            ],
            "must_not_claim": [
                "The distilled sidecar improves detection or segmentation.",
                "The sidecars are direct full-image Qwen API observations.",
                "The sidecars are synchronized physical sensor measurements.",
                "This single-seed development run is a formal paper metric.",
            ],
            "patent_requirement_status_changed": False,
        },
        "next_decision": {
            "recommended_paper_role": (
                "transparent negative development ablation"
            ),
            "formal_next_step": (
                "run the unchanged direct full-image Qwen three-seed protocol"
            ),
            "highest_value_external_step": (
                "collect independent hardware or non-image sensor observations"
            ),
        },
        "evidence_bindings": {
            name: _binding(path) for name, path in EVIDENCE.items()
        },
        "builder_source": _binding(Path(__file__)),
    }


def write(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    index = build()
    _atomic_write_text(
        output,
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
    )
    metrics = index["headline_metrics"]
    markdown = [
        "# 目标级 Qwen 蒸馏开发扩展索引",
        "",
        "状态：完整、有效、非正式论文指标。",
        "",
        "## 已完成",
        "",
        "- 目标级 Qwen 教师数据与泄漏风险审计。",
        "- 标注派生教师网格、整图 RGB 学生训练及验证集校准。",
        "- 5,990 张图的无标注场景侧车生成与同划分乱序对照。",
        "- 固定 v5 父模型到 v6 专利融合栈的单种子联合适配。",
        "- matched、fallback、shuffled、missing、noise、bias、stuck 条件评估。",
        "",
        "## 核心数值",
        "",
        (
            f"- 网格测试 MSE {metrics['grid_test_mse']:.6f}，均值基线 "
            f"{metrics['grid_test_mean_baseline_mse']:.6f}，相对下降 "
            f"{metrics['grid_relative_mse_reduction'] * 100:.2f}%。"
        ),
        (
            f"- 联合测试 Box/Mask mAP50-95："
            f"{metrics['joint_clean_box_map50_95']:.6f}/"
            f"{metrics['joint_clean_mask_map50_95']:.6f}。"
        ),
        (
            "- matched 相对 RGB fallback 和 shuffled 的差异均小于 "
            "0.002 个百分点，未检测到精度贡献。"
        ),
        "",
        "## 结论边界",
        "",
        "- 可作为软传感器蒸馏链路与可靠性门控兼容性的开发证据。",
        "- 不能作为检测/分割精度提升证据。",
        "- 不能替代三种子整图 Qwen 正式协议或硬件传感器实验。",
        "",
        "完整来源和 SHA-256 见同名 JSON。",
        "",
    ]
    _atomic_write_text(output.with_suffix(".md"), "\n".join(markdown))
    return index


def main() -> None:
    index = write(parse_args().output)
    print(
        json.dumps(
            {
                "state": index["state"],
                "valid": index["valid"],
                "formal_paper_metric": index["formal_paper_metric"],
                "evidence_binding_count": len(
                    index["evidence_bindings"]
                ),
                "segmentation_accuracy_gain": index["readiness"][
                    "segmentation_accuracy_gain"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
