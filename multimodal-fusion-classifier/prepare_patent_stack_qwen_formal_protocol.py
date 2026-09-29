"""Freeze the additive Qwen formal-adaptation protocol for the v6 stack."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from train_patent_aligned_external_sensor_joint_seg import (
    DEFAULT_BASE_WEIGHTS,
)


PROTOCOL_NAME = "full_image_qwen_patent_stack_adaptation_v6"
RUN_PROTOCOL_NAME = (
    "joint_text_external_scene_sensor_patent_stack_qwen_adaptation_v6"
)
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent-qwen"
)
DEFAULT_OUTPUT = REPORT_ROOT / "QWEN_FORMAL_EXTENSION_PROTOCOL.json"
OLD_PROTOCOL = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
    / "QWEN_FORMAL_PROTOCOL.json"
)
DATA_OUTPUT = PROJECT_ROOT / "data" / "scene-qwen-sensor-sidecars"
DATASET_YAML = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "dataset.yaml"
)
TEXT_PROTOTYPES = (
    PROJECT_ROOT / "data" / "joint-multimodal-seg" / "text_prototypes.npy"
)
WORKER_EQUIVALENCE = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
    / "WORKER_EQUIVALENCE.json"
)


SOURCE_PATHS = {
    "protocol_builder": Path(__file__),
    "legacy_v2_protocol": OLD_PROTOCOL,
    "scene_sensor_module": PROJECT_ROOT / "mmfc" / "scene_sensors.py",
    "scene_sensor_builder": PROJECT_ROOT / "build_scene_sensor_sidecars.py",
    "scene_sensor_validator": (
        PROJECT_ROOT / "validate_scene_sensor_sidecars.py"
    ),
    "derangement_builder": (
        PROJECT_ROOT / "prepare_qwen_sensor_derangement.py"
    ),
    "patent_stack_model": PROJECT_ROOT / "mmfc" / "joint_seg_patent.py",
    "spatial_model": PROJECT_ROOT / "mmfc" / "joint_seg_spatial.py",
    "contrastive_model": (
        PROJECT_ROOT / "mmfc" / "joint_seg_contrastive.py"
    ),
    "reliability_model": PROJECT_ROOT / "mmfc" / "joint_seg_reliable.py",
    "external_sensor_model": (
        PROJECT_ROOT / "mmfc" / "joint_seg_external.py"
    ),
    "base_joint_model": PROJECT_ROOT / "mmfc" / "joint_seg.py",
    "development_trainer": (
        PROJECT_ROOT / "train_patent_aligned_external_sensor_joint_seg.py"
    ),
    "formal_adaptation_trainer": (
        PROJECT_ROOT / "train_patent_aligned_qwen_formal.py"
    ),
    "formal_condition_evaluator": (
        PROJECT_ROOT
        / "evaluate_patent_aligned_qwen_formal_conditions.py"
    ),
    "formal_aggregator": (
        PROJECT_ROOT / "aggregate_patent_stack_qwen_formal.py"
    ),
    "formal_runner": PROJECT_ROOT / "run_patent_stack_qwen_formal.py",
    "lineage_auditor": (
        PROJECT_ROOT / "audit_patent_stack_qwen_lineage.py"
    ),
    "dataset_yaml": DATASET_YAML,
    "text_prototypes": TEXT_PROTOTYPES,
    "fixed_v5_parent_checkpoint": DEFAULT_BASE_WEIGHTS,
    "worker_equivalence": WORKER_EQUIVALENCE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the additive v6 Qwen formal protocol"
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


def build_protocol() -> dict[str, Any]:
    missing = [
        name for name, path in SOURCE_PATHS.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "formal extension inputs missing: " + ", ".join(missing)
        )
    old_protocol = json.loads(OLD_PROTOCOL.read_text(encoding="utf-8-sig"))
    if (
        old_protocol.get("protocol")
        != "full_image_qwen_external_sensor_joint_seg_v2"
        or old_protocol.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
    ):
        raise ValueError("legacy v2 protocol identity has drifted")

    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "run_protocol": RUN_PROTOCOL_NAME,
        "status": "preregistered_awaiting_api_credentials_and_data",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Adapt the patent-aligned v6 native Segment fusion stack from "
            "one fixed v5 rule-developed parent to annotation-free "
            "full-image Qwen soft-sensor observations, then evaluate every "
            "registered seed under matched, fallback, shuffled, and "
            "software-degraded conditions."
        ),
        "relationship_to_legacy_v2": {
            "legacy_protocol": old_protocol["protocol"],
            "legacy_protocol_modified": False,
            "extension_is_additive": True,
            "architecture_overlap": (
                "Both use the same annotation-free full-image Qwen sidecars; "
                "v2 trains the direct external-sensor head from an RGB "
                "checkpoint, while this extension adapts the v6 stack from "
                "a fixed v5 reliability-development parent."
            ),
        },
        "lineage": {
            "experiment_type": "fixed_parent_multiseed_qwen_adaptation",
            "independent_end_to_end_replicates": False,
            "shared_parent_across_seeds": True,
            "parent_architecture": (
                "spatial contrastive reliability joint Segment head v5"
            ),
            "parent_sensor_source": "scene-rule-v1",
            "parent_development_seed": 42,
            "parent_checkpoint": _binding(DEFAULT_BASE_WEIGHTS),
            "child_architecture": (
                "patent-aligned three-adapter FiLM cosine-attention "
                "dynamic-multiscale native Segment stack v6"
            ),
            "interpretation": (
                "Seeds quantify stochastic variation in Qwen adaptation "
                "from one fixed parent; they do not estimate full-pipeline "
                "variance from independent RGB pretraining."
            ),
        },
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
            "model": "qwen3.5-flash",
            "api_key_environment_variable_name": "DASHSCOPE_API_KEY",
            "api_key_value_persisted": False,
            "output": DATA_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
            "workers": 2,
            "retries": 5,
            "timeout_seconds": 90,
            "resume": True,
            "minimum_qwen_coverage": 0.99,
            "required_valid_record_fraction": 1.0,
            "fallback_policy": (
                "Retain deterministic full-image scene-rule-v1 records only "
                "after retries, disclose their count, and exclude no record."
            ),
            "forbidden_inputs": [
                "YOLO label files",
                "LabelMe annotations",
                "bounding boxes",
                "polygons or masks",
                "class names or class indices",
                (
                    "data/joint-multimodal-seg/"
                    "image_sensor_targets.jsonl"
                ),
            ],
        },
        "derangement_control": {
            "method": "within_split_one_position_cyclic_derangement",
            "seed_required": False,
            "self_assignments_allowed": False,
            "split_membership_preserved": True,
            "sensor_payload_multiset_preserved": True,
            "output": (
                DATA_OUTPUT / "SHUFFLED_OBSERVATIONS.jsonl"
            ).relative_to(PROJECT_ROOT).as_posix(),
            "report": (
                DATA_OUTPUT / "SHUFFLE_REPORT.json"
            ).relative_to(PROJECT_ROOT).as_posix(),
        },
        "training": {
            "architecture": RUN_PROTOCOL_NAME,
            "seeds": [42, 43, 44],
            "epochs": 20,
            "patience": 5,
            "batch_size": 32,
            "image_size": 640,
            "workers": 4,
            "worker_equivalence_evidence": (
                WORKER_EQUIVALENCE.relative_to(PROJECT_ROOT).as_posix()
            ),
            "freeze_layers": 24,
            "amp": True,
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0005,
            "cache": "false",
            "inclusion_rule": (
                "Include every registered seed that completes the frozen "
                "pipeline; do not select checkpoints or seeds by test result."
            ),
            "output_project": (
                "checkpoints/joint-multimodal-seg-patent-qwen"
            ),
            "report_root": REPORT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        },
        "fusion": {
            "semantic_dim": 64,
            "sensor_hidden_dim": 256,
            "text_temperature": 0.1,
            "sensor_reconstruction_weight": 0.25,
            "sensor_calibration_weight": 0.05,
            "reliability_calibration_weight": 0.5,
            "reliability_hidden_dim": 32,
            "reliability_init": 0.85,
            "mismatch_probability": 0.35,
            "patent_feature_dim": 512,
            "patent_attention_dim": 128,
            "patent_attention_heads": 8,
            "patent_dropout": 0.3,
            "dynamic_multiscale_initial_prior": [0.3, 0.5, 0.2],
        },
        "software_sensor_degradation": {
            "missing_probability": 0.1,
            "noise_probability": 0.5,
            "noise_std_low": 0.02,
            "noise_std_high": 0.15,
            "bias_probability": 0.25,
            "bias_strength": 0.15,
            "stuck_probability": 0.15,
            "quality_scale": 0.25,
        },
        "evaluation": {
            "primary_metrics": [
                "box mAP50-95",
                "mask mAP50-95",
            ],
            "primary_conditions": [
                "clean matched Qwen full-image sidecar",
                "same-checkpoint RGB predicted-sensor fallback",
                "same-checkpoint within-split shuffled Qwen sidecar",
            ],
            "secondary_robustness_conditions": [
                "missing",
                "noise std 0.15",
                "bias/drift strength 0.25",
                "stuck",
            ],
            "summary": [
                "mean",
                "sample standard deviation",
                "same-seed paired differences",
            ],
            "test_source_image_count": 570,
            "evaluation_batch_size": 64,
            "test_forward_inputs": [
                "RGB",
                "annotation-free full-image 4x4 soft sensor",
            ],
        },
        "claim_boundaries": [
            (
                "The Qwen outputs are image-conditioned soft-sensor "
                "estimates, not synchronized hardware measurements."
            ),
            "No RGB-D or depth input is introduced.",
            (
                "The three seeds are adaptation replicates sharing one "
                "rule-developed v5 parent, not independent end-to-end runs."
            ),
            (
                "The 19-class taxonomy is general waste rather than the "
                "medical-waste taxonomy in the patent document."
            ),
            (
                "Software degradation evaluates simulated corruption and "
                "does not establish physical sensor robustness."
            ),
            (
                "Results become formal extension metrics only after all "
                "three registered seeds and conditions pass aggregation."
            ),
        ],
        "current_gate": {
            "api_credentials_available": False,
            "qwen_full_image_records_built": False,
            "formal_adaptation_started": False,
            "rule_development_metrics_are_formal": False,
        },
        "source_bindings": {
            name: _binding(path) for name, path in SOURCE_PATHS.items()
        },
    }


def write_protocol(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    protocol = build_protocol()
    payload = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8-sig"))
        comparable_existing = dict(existing)
        comparable_new = dict(protocol)
        comparable_existing.pop("frozen_at", None)
        comparable_new.pop("frozen_at", None)
        if comparable_existing != comparable_new:
            raise FileExistsError(
                "refusing to replace a drifted preregistered protocol"
            )
        return existing
    _atomic_text(output, payload)
    markdown = [
        "# v6 Qwen 正式适配扩展协议",
        "",
        "- 状态：已预注册，等待 API 凭据和整图 Qwen 数据。",
        "- 种子：42、43、44；每次最多20轮；batch=32；workers=4。",
        "- 父模型：固定的 v5 rule-sidecar 机制开发检查点。",
        "- 子模型：v6 专利对齐融合栈。",
        "- 主对照：matched Qwen、RGB fallback、同划分乱序 Qwen。",
        "- 次要稳健性条件：missing、noise、bias/drift、stuck。",
        "",
        (
            "三个种子只度量从同一父检查点进行 Qwen 适配时的随机性，"
            "不代表三次独立端到端训练。Qwen 输出仍是图像条件软传感器，"
            "不是硬件实测。"
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
                "seeds": protocol["training"]["seeds"],
                "shared_parent": protocol["lineage"][
                    "shared_parent_across_seeds"
                ],
                "source_binding_count": len(protocol["source_bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
