"""Build the additive index for reliability-aware external sensor work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORTS / "RELIABLE_EXTERNAL_EXTENSION_INDEX.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reliable external-sensor extension index"
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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build_index(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    paths = {
        "builder_source": Path(__file__),
        "base_external_index": REPORTS / "EXTERNAL_SENSOR_EXTENSION_INDEX.json",
        "development_audit": REPORTS / "SPATIAL_RELIABILITY_AUDIT.json",
        "development_audit_md": REPORTS / "SPATIAL_RELIABILITY_AUDIT.md",
        "qwen_formal_protocol": (
            REPORTS
            / "joint-multimodal-seg-external"
            / "QWEN_FORMAL_PROTOCOL.json"
        ),
        "v3_module": PROJECT_ROOT / "mmfc" / "joint_seg_reliable.py",
        "v4_mismatch_module": (
            PROJECT_ROOT / "mmfc" / "joint_seg_contrastive.py"
        ),
        "v5_spatial_module": PROJECT_ROOT / "mmfc" / "joint_seg_spatial.py",
        "v3_trainer": (
            PROJECT_ROOT / "train_reliable_external_sensor_joint_seg.py"
        ),
        "v4_trainer": (
            PROJECT_ROOT
            / "train_contrastive_reliable_external_sensor_joint_seg.py"
        ),
        "v5_trainer": (
            PROJECT_ROOT
            / "train_spatial_contrastive_external_sensor_joint_seg.py"
        ),
        "v3_evaluator": (
            PROJECT_ROOT / "evaluate_reliable_external_conditions.py"
        ),
        "v4_evaluator": (
            PROJECT_ROOT / "evaluate_contrastive_external_conditions.py"
        ),
        "v5_evaluator": (
            PROJECT_ROOT / "evaluate_spatial_external_conditions.py"
        ),
        "separation_diagnostic": (
            PROJECT_ROOT / "diagnose_spatial_sensor_separation.py"
        ),
        "auditor": (
            PROJECT_ROOT / "audit_spatial_reliability_extension.py"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "reliable extension inputs missing: " + ", ".join(missing)
        )
    base = _read_json(paths["base_external_index"])
    audit = _read_json(paths["development_audit"])
    qwen = _read_json(paths["qwen_formal_protocol"])
    if base.get("valid") is not True or base.get("state") != "complete":
        raise ValueError("base external extension index is not valid")
    if (
        audit.get("valid") is not True
        or audit.get("state") != "complete"
        or audit.get("formal_paper_metric") is not False
        or audit.get("selected_development_model", {}).get("selected")
        != "spatial_v5_primary"
        or audit.get("selected_development_model", {}).get(
            "calibrated_rejected"
        )
        is not True
    ):
        raise ValueError("spatial reliability audit is not valid")
    if (
        qwen.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
        or qwen.get("current_gate", {}).get(
            "qwen_full_image_records_built"
        )
        is not False
        or qwen.get("current_gate", {}).get("formal_training_started")
        is not False
    ):
        raise ValueError("formal Qwen gate is inconsistent")

    test = audit["primary_v5_test"]
    diagnostic = audit["matched_shuffled_diagnostic"]
    index = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "reliability_aware_external_sensor_joint_head_v3_v5",
        "formal_paper_metric": False,
        "capability_status": {
            "software_missing_noise_bias_drift_stuck": "implemented_and_tested",
            "availability_gated_rgb_fallback": "implemented_and_tested",
            "cross_modal_consistency_gate": "implemented_and_tested",
            "label_free_mismatched_sensor_negatives": "implemented_and_tested",
            "spatial_rgb_sensor_prediction": "implemented_and_tested",
            "matched_shuffled_auc_diagnostic": "implemented_and_tested",
            "overcalibration_rejection_rule": "implemented_and_applied",
            "full_image_qwen_sidecars": "awaiting_api_credentials",
            "three_seed_qwen_training": "not_started",
            "hardware_sensor_array": "not_implemented",
            "rgb_d_input": "not_implemented",
        },
        "selected_development_result": {
            "model": "spatial_v5_primary",
            "seed_count": 1,
            "training_epochs": 5,
            "sensor_source": "scene-rule-v1",
            "source_image_count": 570,
            "clean_minus_rgb_fallback": test[
                "clean_minus_rgb_fallback"
            ],
            "clean_minus_shuffled": test["clean_minus_shuffled"],
            "discrepancy_auc_shuffled_positive": diagnostic[
                "discrepancy_auc_shuffled_positive"
            ],
            "raw_reliability_auc_shuffled_positive": diagnostic[
                "raw_reliability_auc_shuffled_positive"
            ],
            "formal_performance_evidence": False,
        },
        "governance": {
            "overcalibrated_candidate_promoted": False,
            "rule_sidecar_result_promoted_to_formal": False,
            "qwen_formal_protocol_modified": False,
            "checkpoint_in_extension_archive": False,
            "source_images_in_extension_archive": False,
        },
        "next_gate": {
            "required_environment_variable": "DASHSCOPE_API_KEY",
            "minimum_qwen_coverage": 0.99,
            "planned_training_seeds": [42, 43, 44],
            "training_workers": 4,
            "required_conditions": [
                "matched",
                "fallback",
                "shuffled",
            ],
        },
        "claim_boundaries": audit["claim_boundaries"],
        "bindings": {
            name: _binding(path) for name, path in paths.items()
        },
    }
    _atomic_text(
        output,
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 可靠性外部传感器扩展索引",
        "",
        "- 状态：开发证据完整；正式论文指标：否。",
        "- 主开发模型：v5 空间 RGB 软传感器预测 + 错配负样本 + 可靠性门控。",
        "- 测试图像：570；单 seed；scene-rule-v1 sidecar。",
        "- 干净匹配相对 RGB 回退：框 "
        f"{test['clean_minus_rgb_fallback']['metrics/mAP50-95(B)'] * 100:.3f} "
        "个百分点，掩码 "
        f"{test['clean_minus_rgb_fallback']['metrics/mAP50-95(M)'] * 100:.3f} 个百分点。",
        "- 乱序偏差 AUC："
        f"{diagnostic['discrepancy_auc_shuffled_positive']:.3f}。",
        "- 强化校准候选因精度下降且乱序未劣于匹配而被拒绝。",
        "- Qwen 全图 sidecar 与三 seed 正式训练仍等待环境变量凭据。",
        "",
        "本索引不把图像条件软传感器表述为真实硬件测量，也不把单 seed rule 结果写入正式性能表。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return index


def main() -> None:
    result = build_index(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
