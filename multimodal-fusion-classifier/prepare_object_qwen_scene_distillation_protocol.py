"""Freeze the object-Qwen scene-distillation development protocol."""

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
DEFAULT_OUTPUT = ROOT / "PROTOCOL.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the object-Qwen scene-distillation protocol"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def build_protocol(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    teacher_root = (
        PROJECT_ROOT / "data" / "object-qwen-scene-teacher"
    )
    paths = {
        "feasibility_audit": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "OBJECT_QWEN_SCENE_DISTILLATION_FEASIBILITY.json"
        ),
        "teacher_report": teacher_root / "BUILD_REPORT.json",
        "train_teacher": teacher_root / "train.jsonl",
        "val_teacher": teacher_root / "val.jsonl",
        "test_teacher": teacher_root / "test.jsonl",
        "distillation_module": (
            PROJECT_ROOT / "mmfc" / "scene_distillation.py"
        ),
        "trainer_source": PROJECT_ROOT / "train_scene_qwen_distiller.py",
        "predictor_source": (
            PROJECT_ROOT / "predict_scene_qwen_distiller.py"
        ),
        "protocol_builder": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "distillation protocol inputs missing: " + ", ".join(missing)
        )
    feasibility = json.loads(
        paths["feasibility_audit"].read_text(encoding="utf-8-sig")
    )
    teacher = json.loads(
        paths["teacher_report"].read_text(encoding="utf-8-sig")
    )
    if (
        feasibility.get("valid") is not True
        or feasibility.get("decision", {}).get(
            "direct_bbox_aggregate_as_detector_input"
        )
        != "prohibited"
        or teacher.get("valid") is not True
        or teacher.get("direct_detector_input_permitted") is not False
        or teacher.get("split_counts")
        != {"train": 4823, "val": 597, "test": 570}
        or teacher.get("usage_contract", {}).get(
            "test_split_training_or_selection_allowed"
        )
        is not False
    ):
        raise ValueError("distillation protocol source gate is invalid")

    protocol = {
        "schema_version": 1,
        "state": "frozen_before_training",
        "valid": True,
        "formal_paper_metric": False,
        "protocol": "object_qwen_scene_distillation_development_v1",
        "purpose": (
            "Distill annotation-derived object-crop Qwen fields into an "
            "annotation-free RGB-only full-image 4x4 sidecar."
        ),
        "data_contract": {
            "teacher_source_image_counts": teacher["split_counts"],
            "teacher_object_counts": teacher["object_split_counts"],
            "teacher_qwen_success_count": teacher["generator_counts"][
                "qwen3-vl-flash"
            ],
            "teacher_rule_fallback_count": teacher["generator_counts"][
                "rule-v1"
            ],
            "teacher_aggregation": teacher["aggregation"]["method"],
            "physical_split_files": True,
            "train_reads": ["train.jsonl", "train_rgb_images"],
            "selection_reads": ["val.jsonl", "val_rgb_images"],
            "test_read_time": "only_after_best_checkpoint_is_frozen",
            "direct_teacher_to_joint_head": False,
        },
        "model": {
            "architecture": "SceneQwenDistiller",
            "channels": [24, 32, 48, 64, 96],
            "input": "normalized_full_rgb_image",
            "output": "sigmoid_1x4x4",
            "pretrained_weights": False,
            "parameter_selection_after_results": False,
        },
        "loss": {
            "weighted_mse": "weight=1+4*sqrt(coverage)",
            "l1_weight": 0.1,
            "spatial_gradient_l1_weight": 0.2,
        },
        "runtime": {
            "seed": 42,
            "image_size": 256,
            "batch_size": 256,
            "workers": 4,
            "epochs": 40,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "patience": 10,
            "optimizer": "AdamW",
            "schedule": "CosineAnnealingLR",
            "amp": True,
            "device": "cuda",
        },
        "selection": {
            "metric": "validation_composite_loss",
            "lower_is_better": True,
            "test_used_for_selection": False,
        },
        "frozen_acceptance": {
            "validation_mse_below_train_mean_grid_baseline": True,
            "test_mse_below_train_mean_grid_baseline": True,
            "validation_elementwise_spearman_above_zero": True,
            "test_elementwise_spearman_above_zero": True,
        },
        "post_acceptance_action": {
            "generate_all_split_sidecars_from_rgb_only_student": True,
            "validate_with_scene_sidecar_validator": True,
            "joint_head_training_requires_separate_frozen_protocol": True,
        },
        "prohibited": [
            "feeding annotation-derived teacher grids to the detector",
            "reading test teacher fields before checkpoint selection",
            "calling the distilled sidecar a physical sensor",
            "calling the distilled sidecar a direct full-image Qwen API result",
            "replacing the preregistered formal full-image Qwen protocol",
        ],
        "bindings": {name: _binding(path) for name, path in paths.items()},
    }
    _atomic_text(
        output,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 对象Qwen整图蒸馏开发协议",
        "",
        "- 状态：训练前冻结；正式论文主指标：否。",
        "- 教师场：训练4823、验证597、测试570张源图；Qwen对象观测8639条，rule回退15条。",
        "- 学生输入：仅整图RGB；输出：1×4×4连续场。",
        "- 训练：seed 42，256输入，batch 256，workers 4，最多40轮，patience 10，AMP。",
        "- 选模：仅验证复合损失；测试教师场在best检查点冻结后才读取。",
        "- 接受条件：验证/测试MSE均优于训练均值网格基线，且Spearman均大于0。",
        "",
        "真实框聚合教师场不得直接输入检测器。本协议不替代冻结的整图Qwen正式三种子协议。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return protocol


def main() -> None:
    result = build_protocol(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "protocol": result["protocol"],
                "runtime": result["runtime"],
                "frozen_acceptance": result["frozen_acceptance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
