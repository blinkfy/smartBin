"""Freeze the planned full-image Qwen sidecar and v2 training protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
)
DEFAULT_OUTPUT = REPORT_ROOT / "QWEN_FORMAL_PROTOCOL.json"
SOURCE_PATHS = {
    "protocol_builder": (
        PROJECT_ROOT / "prepare_external_sensor_qwen_protocol.py"
    ),
    "scene_sensor_module": PROJECT_ROOT / "mmfc" / "scene_sensors.py",
    "scene_sensor_builder": PROJECT_ROOT / "build_scene_sensor_sidecars.py",
    "scene_sensor_validator": (
        PROJECT_ROOT / "validate_scene_sensor_sidecars.py"
    ),
    "external_joint_module": (
        PROJECT_ROOT / "mmfc" / "joint_seg_external.py"
    ),
    "external_joint_trainer": (
        PROJECT_ROOT / "train_external_sensor_joint_seg.py"
    ),
    "condition_evaluator": (
        PROJECT_ROOT / "evaluate_external_sensor_conditions.py"
    ),
    "worker_equivalence_auditor": (
        PROJECT_ROOT / "audit_external_sensor_worker_equivalence.py"
    ),
    "worker_equivalence_report": (
        REPORT_ROOT / "WORKER_EQUIVALENCE.json"
    ),
    "formal_runner": (
        PROJECT_ROOT / "run_external_sensor_qwen_formal.py"
    ),
    "formal_aggregator": (
        PROJECT_ROOT / "aggregate_external_sensor_formal.py"
    ),
    "text_prototypes": (
        PROJECT_ROOT
        / "data"
        / "joint-multimodal-seg"
        / "text_prototypes.npy"
    ),
    "dataset_yaml": (
        PROJECT_ROOT
        / "data"
        / "yolo-seg-unique-linked"
        / "dataset.yaml"
    ),
    "base_checkpoint": (
        PROJECT_ROOT
        / "checkpoints"
        / "yolo-seg"
        / "yolo11s-seg-qwen-split"
        / "weights"
        / "best.pt"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full-image Qwen external-sensor protocol"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def build_protocol(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    missing = [name for name, path in SOURCE_PATHS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "external-sensor protocol sources are missing: "
            + ", ".join(missing)
        )
    protocol = {
        "schema_version": 1,
        "protocol": "full_image_qwen_external_sensor_joint_seg_v2",
        "status": "preregistered_awaiting_api_credentials_and_data",
        "purpose": (
            "Generate annotation-free full-image Qwen soft-sensor observations "
            "and evaluate direct 4x4 sensor conditioning of the Segment head."
        ),
        "data_generation": {
            "source_images": (
                "data/yolo-seg-unique-linked/images/{train,val,test}"
            ),
            "expected_source_image_count": 5990,
            "expected_split_counts": {
                "train": 4823,
                "val": 597,
                "test": 570,
            },
            "mode": "qwen",
            "model": "qwen3-vl-flash",
            "api_key_environment_variable_name": "DASHSCOPE_API_KEY",
            "api_key_value_persisted": False,
            "output": "data/scene-qwen-sensor-sidecars",
            "workers": 2,
            "retries": 5,
            "timeout_seconds": 90,
            "resume": True,
            "minimum_qwen_coverage": 0.99,
            "required_valid_record_fraction": 1.0,
            "fallback_policy": (
                "Retain deterministic full-image scene-rule-v1 records only "
                "after retries; disclose count and exclude no record."
            ),
        },
        "forbidden_inputs": [
            "YOLO label files",
            "LabelMe annotations",
            "bounding boxes",
            "polygons or masks",
            "class names or class indices",
            (
                "data/joint-multimodal-seg/image_sensor_targets.jsonl "
                "annotation-projected grids"
            ),
        ],
        "training": {
            "architecture": (
                "ExternalSensorJointSegmentationModel with RGB backbone, "
                "CLIP prototypes, and direct full-image 4x4 sidecar"
            ),
            "seeds": [42, 43, 44],
            "epochs": 20,
            "patience": 5,
            "batch_size": 32,
            "image_size": 640,
            "workers": 4,
            "worker_equivalence_evidence": (
                "reports/qwen-material/joint-multimodal-seg-external/"
                "WORKER_EQUIVALENCE.json"
            ),
            "freeze_layers": 24,
            "amp": True,
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0005,
            "inclusion_rule": (
                "Include every completed formal seed; do not select by test result."
            ),
        },
        "fusion": {
            "semantic_dim": 64,
            "sensor_hidden_dim": 256,
            "text_temperature": 0.1,
            "sensor_reconstruction_weight": 0.25,
            "sensor_calibration_weight": 0.05,
        },
        "evaluation": {
            "primary_metrics": [
                "box mAP50-95",
                "mask mAP50-95",
            ],
            "required_conditions": [
                "external Qwen full-image sidecar",
                "same-checkpoint RGB predicted-sensor fallback",
                "same-checkpoint shuffled-sidecar negative control",
            ],
            "summary": [
                "mean",
                "sample standard deviation",
                "same-seed paired differences",
            ],
            "test_forward_inputs": [
                "RGB",
                "annotation-free full-image 4x4 soft sensor",
            ],
        },
        "execution_commands": {
            "build": (
                "python build_scene_sensor_sidecars.py --mode qwen "
                "--api-key-env DASHSCOPE_API_KEY "
                "--output data/scene-qwen-sensor-sidecars --resume "
                "--workers 2 --retries 5 --timeout 90 "
                "--checkpoint-every 20 --progress-every 20"
            ),
            "validate": (
                "python validate_scene_sensor_sidecars.py "
                "--output data/scene-qwen-sensor-sidecars"
            ),
            "train_template": (
                "python train_external_sensor_joint_seg.py "
                "--sensor-observations "
                "data/scene-qwen-sensor-sidecars/observations.jsonl "
                "--name yolo11s-seg-joint-external-qwen-seed{seed} "
                "--epochs 20 --patience 5 --batch-size 32 "
                "--workers 4 --freeze-layers 24 --seed {seed}"
            ),
            "shuffled_control_template": (
                "python evaluate_external_sensor_conditions.py "
                "--checkpoint checkpoints/joint-multimodal-seg-external/"
                "yolo11s-seg-joint-external-qwen-seed{seed}/weights/best.pt "
                "--sensor-observations "
                "data/scene-qwen-sensor-sidecars/observations.jsonl "
                "--output reports/qwen-material/"
                "joint-multimodal-seg-external/"
                "yolo11s-seg-joint-external-qwen-seed{seed}/conditions "
                "--batch-size 64 --workers 4"
            ),
            "formal_runner": (
                "python run_external_sensor_qwen_formal.py"
            ),
            "preflight_only": (
                "python run_external_sensor_qwen_formal.py --preflight-only"
            ),
        },
        "current_gate": {
            "api_credentials_available": False,
            "qwen_full_image_records_built": False,
            "formal_training_started": False,
            "rule_smoke_is_formal_evidence": False,
        },
        "claim_boundaries": [
            "Qwen output is an image-conditioned soft sensor, not hardware.",
            "No RGB-D input is introduced.",
            "The rule one-epoch smoke is excluded from formal paper tables.",
            "External blind validation and physical synchronization remain absent.",
        ],
        "source_bindings": {
            name: _binding(path) for name, path in SOURCE_PATHS.items()
        },
    }
    _atomic_write(
        output_path,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 整图 Qwen 外部传感器正式协议",
        "",
        "- 状态：已预注册，等待 API 环境变量与整图数据",
        "- 目标：让 4×4 整图软传感器直接进入联合 Segment head",
        "- 图像数：5,990（train/val/test = 4,823/597/570）",
        "- 正式训练种子：42、43、44",
        "- 正式训练：batch=32，workers=4（逐张量等价性基准支持）",
        "- rule 一轮 smoke：仅集成验证，不进入正式表格",
        "",
        "数据生成禁止读取标签文件、LabelMe 标注、bbox、polygon、mask、类别名称或"
        "旧的标注投影网格。Qwen 输出仍是图像条件估计，不是真实硬件测量。",
        "",
        "执行命令已写入同名 JSON；密钥只从 `DASHSCOPE_API_KEY` 读取且不落盘。",
        "统一入口：`python run_external_sensor_qwen_formal.py`；只检查门禁可添加 "
        "`--preflight-only`。",
        "",
    ]
    _atomic_write(output_path.with_suffix(".md"), "\n".join(markdown))
    return protocol


def main() -> None:
    protocol = build_protocol(parse_args().output)
    print(
        json.dumps(
            {
                "state": protocol["status"],
                "valid": True,
                "source_binding_count": len(protocol["source_bindings"]),
                "api_key_value_persisted": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
