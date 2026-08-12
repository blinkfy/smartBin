"""Freeze the v6 development protocol for distilled Qwen scene sidecars."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from train_patent_aligned_external_sensor_joint_seg import (
    DEFAULT_BASE_WEIGHTS,
)
from train_patent_aligned_qwen_distilled import (
    DEFAULT_REPORT,
    DEFAULT_SENSOR_OBSERVATIONS,
    REGISTERED_PROTOCOL_NAME,
    RUN_PROTOCOL_NAME,
)


DEFAULT_OUTPUT = DEFAULT_REPORT / "DEVELOPMENT_PROTOCOL.json"
SHUFFLED = (
    DEFAULT_SENSOR_OBSERVATIONS.parent / "SHUFFLED_OBSERVATIONS.jsonl"
)
DATASET_YAML = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "dataset.yaml"
)
TEXT_PROTOTYPES = (
    PROJECT_ROOT / "data" / "joint-multimodal-seg" / "text_prototypes.npy"
)
DISTILLATION_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "object-qwen-scene-distillation"
)
DISTILLER_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "scene-qwen-distiller-v1" / "best.pt"
)


SOURCE_PATHS = {
    "protocol_builder": Path(__file__),
    "feasibility_audit": (
        PROJECT_ROOT / "audit_object_qwen_scene_distillation_feasibility.py"
    ),
    "teacher_builder": (
        PROJECT_ROOT / "build_object_qwen_scene_teacher.py"
    ),
    "distillation_module": PROJECT_ROOT / "mmfc" / "scene_distillation.py",
    "distiller_trainer": PROJECT_ROOT / "train_scene_qwen_distiller.py",
    "distiller_predictor": PROJECT_ROOT / "predict_scene_qwen_distiller.py",
    "calibration_fitter": (
        PROJECT_ROOT / "fit_scene_qwen_distiller_calibration.py"
    ),
    "calibration_evaluator": (
        PROJECT_ROOT / "evaluate_scene_qwen_distiller_calibration.py"
    ),
    "derangement_builder": (
        PROJECT_ROOT / "prepare_object_qwen_scene_derangement.py"
    ),
    "joint_trainer": PROJECT_ROOT / "train_patent_aligned_qwen_distilled.py",
    "condition_evaluator": (
        PROJECT_ROOT
        / "evaluate_patent_aligned_qwen_distilled_conditions.py"
    ),
    "scene_observations": DEFAULT_SENSOR_OBSERVATIONS,
    "scene_build": DEFAULT_SENSOR_OBSERVATIONS.parent / "BUILD_REPORT.json",
    "scene_validation": (
        DEFAULT_SENSOR_OBSERVATIONS.parent / "VALIDATION_REPORT.json"
    ),
    "shuffled_observations": SHUFFLED,
    "shuffle_report": SHUFFLED.parent / "SHUFFLE_REPORT.json",
    "distiller_checkpoint": DISTILLER_CHECKPOINT,
    "distiller_training_report": DISTILLATION_REPORT / "TRAINING.json",
    "calibration_parameters": (
        DISTILLATION_REPORT / "CALIBRATION_PARAMETERS.json"
    ),
    "calibration_evaluation": (
        DISTILLATION_REPORT / "CALIBRATION_EVALUATION.json"
    ),
    "dataset_yaml": DATASET_YAML,
    "text_prototypes": TEXT_PROTOTYPES,
    "fixed_v5_parent_checkpoint": DEFAULT_BASE_WEIGHTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the object-Qwen-distilled v6 development protocol"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build_protocol() -> dict[str, Any]:
    missing = [
        name for name, path in SOURCE_PATHS.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "distilled development inputs missing: " + ", ".join(missing)
        )
    build = json.loads(
        SOURCE_PATHS["scene_build"].read_text(encoding="utf-8-sig")
    )
    validation = json.loads(
        SOURCE_PATHS["scene_validation"].read_text(encoding="utf-8-sig")
    )
    calibration = json.loads(
        SOURCE_PATHS["calibration_evaluation"].read_text(
            encoding="utf-8-sig"
        )
    )
    shuffle = json.loads(
        SOURCE_PATHS["shuffle_report"].read_text(encoding="utf-8-sig")
    )
    observations_sha = sha256_file(DEFAULT_SENSOR_OBSERVATIONS)
    if (
        build.get("valid") is not True
        or validation.get("valid") is not True
        or build.get("mode") != "object-qwen-scene-distilled"
        or validation.get("mode") != "object-qwen-scene-distilled"
        or build.get("output", {}).get("sha256") != observations_sha
        or validation.get("observations_sha256") != observations_sha
        or calibration.get("valid") is not True
        or calibration.get("formal_paper_metric") is not False
        or shuffle.get("valid") is not True
        or shuffle.get("self_assignment_count") != 0
    ):
        raise ValueError("distilled evidence does not pass the frozen gate")

    return {
        "schema_version": 1,
        "protocol": REGISTERED_PROTOCOL_NAME,
        "run_protocol": RUN_PROTOCOL_NAME,
        "status": "development_protocol_frozen_before_training",
        "formal_paper_metric": False,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Adapt the patent-aligned v6 stack to an RGB-only scene grid "
            "student distilled from object-crop Qwen teacher observations."
        ),
        "lineage": {
            "experiment_type": (
                "single_seed_fixed_parent_distilled_sidecar_adaptation"
            ),
            "seed": 42,
            "parent_checkpoint": _binding(DEFAULT_BASE_WEIGHTS),
            "teacher_source": "object-crop Qwen and deterministic fallback",
            "teacher_uses_training_annotations": True,
            "student_inputs": ["full RGB image"],
            "student_checkpoint": _binding(DISTILLER_CHECKPOINT),
            "student_selection_split": "val",
            "calibration_fit_split": "val",
            "joint_inference_uses_teacher": False,
            "joint_inference_uses_annotations": False,
            "test_teacher_diagnostic_already_run": True,
        },
        "data": {
            "mode": "object-qwen-scene-distilled",
            "source_image_count": 5990,
            "split_counts": {"train": 4823, "val": 597, "test": 570},
            "scene_observations": _binding(DEFAULT_SENSOR_OBSERVATIONS),
            "shuffled_observations": _binding(SHUFFLED),
            "direct_full_image_qwen_api": False,
            "physical_sensor_measurement": False,
            "annotation_inputs_to_generated_sidecar": False,
        },
        "training": {
            "seed": 42,
            "epochs": 20,
            "patience": 5,
            "batch_size": 32,
            "image_size": 640,
            "workers": 4,
            "freeze_layers": 24,
            "amp": True,
            "learning_rate": 0.0001,
            "weight_decay": 0.0005,
        },
        "evaluation": {
            "test_source_image_count": 570,
            "conditions": [
                "clean matched",
                "RGB fallback",
                "within-split shuffled",
                "missing",
                "noise std 0.15",
                "bias/drift strength 0.25",
                "stuck",
            ],
            "primary_metrics": ["box mAP50-95", "mask mAP50-95"],
            "checkpoint_selection_uses_test": False,
            "development_only_due_prior_test_diagnostic_access": True,
        },
        "claim_boundaries": [
            (
                "This is a single-seed development experiment and is not "
                "part of the frozen three-seed formal full-image Qwen study."
            ),
            (
                "The sidecar is an RGB-only student prediction distilled "
                "from annotation-derived object-crop Qwen teacher fields."
            ),
            (
                "Annotations and teacher records are absent from sidecar "
                "generation and joint-model inference."
            ),
            (
                "The Qwen-derived values are image-conditioned soft-sensor "
                "estimates, not synchronized physical measurements."
            ),
            "No RGB-D or depth input is introduced.",
            (
                "The 19 classes are general waste rather than the patent's "
                "medical-waste taxonomy."
            ),
            (
                "The test teacher field was used only in a preregistered "
                "post-fit calibration diagnostic, so all joint results stay "
                "development evidence."
            ),
        ],
        "bindings": {
            name: _binding(path) for name, path in SOURCE_PATHS.items()
        },
    }


def write_protocol(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    protocol = build_protocol()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8-sig"))
        old = dict(existing)
        new = dict(protocol)
        old.pop("frozen_at", None)
        new.pop("frozen_at", None)
        if old != new:
            raise FileExistsError(
                "refusing to replace a drifted frozen development protocol"
            )
        return existing
    _atomic_text(
        output,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 目标级 Qwen 蒸馏侧车 v6 开发协议",
        "",
        "- 状态：训练前冻结；非正式论文指标。",
        "- 单种子 42；20 轮；batch=32；workers=4。",
        "- 固定父模型：v5 spatial contrastive reliability。",
        "- 场景网格：目标级 Qwen 教师蒸馏出的整图 RGB 学生预测。",
        "- 联合训练与推理不读取标注、目标框或教师记录。",
        "- 条件：matched、RGB fallback、shuffled、missing、noise、bias、stuck。",
        "",
        (
            "该实验用于验证 Qwen 辅助软传感器链路的可行性，不能替代"
            "已冻结的三种子整图 Qwen 正式协议，也不能表述为硬件实测。"
        ),
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return protocol


def main() -> None:
    protocol = write_protocol(parse_args().output)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": True,
                "protocol": protocol["protocol"],
                "status": protocol["status"],
                "formal_paper_metric": protocol["formal_paper_metric"],
                "source_binding_count": len(protocol["bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
