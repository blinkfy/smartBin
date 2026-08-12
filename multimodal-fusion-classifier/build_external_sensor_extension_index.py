"""Build the additive evidence index for the external-sensor joint-head v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
EXTERNAL_ROOT = REPORTS_ROOT / "joint-multimodal-seg-external"
SMOKE_ROOT = (
    EXTERNAL_ROOT / "yolo11s-seg-joint-external-rule-smoke"
)
DEFAULT_OUTPUT = REPORTS_ROOT / "EXTERNAL_SENSOR_EXTENSION_INDEX.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build external-sensor joint-head extension index"
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


def _validate_bindings(
    bindings: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not bindings:
        return [f"{label} source bindings are missing"]
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(f"{label} malformed binding: {name}")
            continue
        raw_path = binding.get("path", name)
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            errors.append(f"{label} source is missing: {name}")
        elif (
            path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            errors.append(f"{label} source binding drift: {name}")
    return errors


def build_index(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    paths = {
        "builder_source": Path(__file__),
        "base_joint_index": REPORTS_ROOT / "JOINT_HEAD_EXTENSION_INDEX.json",
        "scene_sensor_document": REPORTS_ROOT / "SCENE_SENSOR_DATA.md",
        "scene_rule_build": (
            PROJECT_ROOT / "data" / "scene-sensor-sidecars" / "BUILD_REPORT.json"
        ),
        "scene_rule_validation": (
            PROJECT_ROOT
            / "data"
            / "scene-sensor-sidecars"
            / "VALIDATION_REPORT.json"
        ),
        "scene_rule_observations": (
            PROJECT_ROOT
            / "data"
            / "scene-sensor-sidecars"
            / "observations.jsonl"
        ),
        "qwen_formal_protocol": EXTERNAL_ROOT / "QWEN_FORMAL_PROTOCOL.json",
        "qwen_formal_protocol_md": EXTERNAL_ROOT / "QWEN_FORMAL_PROTOCOL.md",
        "worker_equivalence": EXTERNAL_ROOT / "WORKER_EQUIVALENCE.json",
        "worker_equivalence_md": EXTERNAL_ROOT / "WORKER_EQUIVALENCE.md",
        "smoke_protocol": SMOKE_ROOT / "PROTOCOL.json",
        "smoke_evaluation": SMOKE_ROOT / "evaluation.json",
        "smoke_outcome": SMOKE_ROOT / "OUTCOME.json",
        "smoke_condition_diagnostics": (
            SMOKE_ROOT / "conditions" / "CONDITION_DIAGNOSTICS.json"
        ),
        "smoke_shuffled_observations": (
            SMOKE_ROOT / "conditions" / "SHUFFLED_OBSERVATIONS.jsonl"
        ),
        "patent_alignment": REPORTS_ROOT / "PATENT_ALIGNMENT.md",
        "decision_brief": REPORTS_ROOT / "MANUSCRIPT_DECISION_BRIEF.md",
        "readme": PROJECT_ROOT / "README.md",
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
        "external_joint_auditor": (
            PROJECT_ROOT / "audit_external_sensor_joint_seg.py"
        ),
        "external_condition_evaluator": (
            PROJECT_ROOT / "evaluate_external_sensor_conditions.py"
        ),
        "worker_equivalence_auditor": (
            PROJECT_ROOT / "audit_external_sensor_worker_equivalence.py"
        ),
        "formal_runner": (
            PROJECT_ROOT / "run_external_sensor_qwen_formal.py"
        ),
        "formal_aggregator": (
            PROJECT_ROOT / "aggregate_external_sensor_formal.py"
        ),
        "qwen_protocol_builder": (
            PROJECT_ROOT / "prepare_external_sensor_qwen_protocol.py"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "external extension inputs are missing: " + ", ".join(missing)
        )
    base_joint = _read_json(paths["base_joint_index"])
    build = _read_json(paths["scene_rule_build"])
    validation = _read_json(paths["scene_rule_validation"])
    qwen_protocol = _read_json(paths["qwen_formal_protocol"])
    worker_equivalence = _read_json(paths["worker_equivalence"])
    smoke_protocol = _read_json(paths["smoke_protocol"])
    smoke = _read_json(paths["smoke_outcome"])
    conditions = _read_json(paths["smoke_condition_diagnostics"])
    errors: list[str] = []
    if base_joint.get("state") != "complete" or base_joint.get("valid") is not True:
        errors.append("base joint-head extension index is not valid")
    if (
        build.get("valid") is not True
        or build.get("mode") != "rule"
        or build.get("source_image_count") != 5990
        or validation.get("valid") is not True
        or validation.get("error_count") != 0
        or validation.get("source_image_count") != 5990
        or build.get("output", {}).get("sha256")
        != _sha256(paths["scene_rule_observations"])
        or validation.get("observations_sha256")
        != _sha256(paths["scene_rule_observations"])
    ):
        errors.append("annotation-free scene-rule data evidence is not valid")
    source_contract = validation.get("source_contract", {})
    if (
        source_contract.get("full_image_only") is not True
        or source_contract.get("annotation_files_read") is not False
        or source_contract.get("label_files_read") is not False
        or source_contract.get("bbox_inputs") is not False
        or source_contract.get("polygon_inputs") is not False
        or source_contract.get("class_inputs") is not False
    ):
        errors.append("scene-rule source contract is not annotation-free")
    if (
        smoke.get("valid") is not True
        or smoke.get("formal_paper_metric") is not False
        or smoke.get("training_epochs") != 1
        or smoke.get("structural_audit", {}).get(
            "rgb_backbone_input_channels"
        )
        != [3, 3]
        or smoke.get("structural_audit", {}).get(
            "external_sensor_used"
        )
        is not True
        or smoke.get("structural_audit", {}).get(
            "missing_sensor_fallback_used"
        )
        is not True
    ):
        errors.append("external-sensor smoke evidence is not valid")
    errors.extend(
        _validate_bindings(smoke.get("bindings", {}), label="smoke")
    )
    condition_contract = conditions.get("condition", {})
    if (
        conditions.get("valid") is not True
        or conditions.get("formal_paper_metric") is not False
        or conditions.get("evidence_role")
        != "integration_smoke_negative_control"
        or condition_contract.get("self_assignment_count") != 0
        or condition_contract.get("source_image_count") != 5990
        or condition_contract.get("test_source_image_count") != 570
        or condition_contract.get("split_preserved") is not True
        or condition_contract.get("sensor_distribution_preserved") is not True
    ):
        errors.append("shuffled-sidecar smoke control is not valid")
    errors.extend(
        _validate_bindings(
            conditions.get("bindings", {}),
            label="shuffled-sidecar control",
        )
    )
    if (
        qwen_protocol.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
        or qwen_protocol.get("current_gate", {}).get(
            "qwen_full_image_records_built"
        )
        is not False
        or qwen_protocol.get("current_gate", {}).get(
            "formal_training_started"
        )
        is not False
        or qwen_protocol.get("training", {}).get("seeds") != [42, 43, 44]
        or qwen_protocol.get("data_generation", {}).get(
            "api_key_value_persisted"
        )
        is not False
    ):
        errors.append("Qwen formal protocol gate is inconsistent")
    errors.extend(
        _validate_bindings(
            qwen_protocol.get("source_bindings", {}),
            label="Qwen protocol",
        )
    )
    if (
        worker_equivalence.get("valid") is not True
        or worker_equivalence.get("equivalence", {})
        .get("model_state", {})
        .get("exact")
        is not True
        or worker_equivalence.get("equivalence", {}).get(
            "training_csv_non_time_equal"
        )
        is not True
        or worker_equivalence.get("equivalence", {}).get(
            "external_and_fallback_test_metrics_equal"
        )
        is not True
        or worker_equivalence.get("timing", {}).get("speedup", 0.0) <= 1.0
        or worker_equivalence.get("recommendation", {}).get(
            "formal_training_workers"
        )
        != 4
        or qwen_protocol.get("training", {}).get("workers") != 4
    ):
        errors.append("four-worker equivalence evidence is not valid")
    errors.extend(
        _validate_bindings(
            worker_equivalence.get("source_bindings", {}),
            label="worker equivalence",
        )
    )
    if (
        smoke_protocol.get("sensor_source", {}).get("mode") != "rule"
        or smoke_protocol.get("leakage_control", {}).get(
            "annotation_conditioned_spatial_projection"
        )
        is not False
    ):
        errors.append("smoke protocol leakage boundary is inconsistent")
    if errors:
        raise ValueError("; ".join(errors))

    deltas = smoke["test"]["external_minus_fallback"]
    shuffled_deltas = conditions["metrics"]["matched_minus_shuffled"]
    index = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "external_full_image_sensor_joint_head_v2_extension",
        "capability_status": {
            "annotation_free_full_image_sidecar_builder": "implemented_and_validated",
            "external_4x4_joint_head_input": "implemented_and_smoke_tested",
            "missing_sensor_rgb_fallback": "implemented_and_smoke_tested",
            "shuffled_sidecar_negative_control": (
                "implemented_and_smoke_tested"
            ),
            "four_worker_training": "exactly_equivalent_and_enabled",
            "protocol_driven_formal_runner": "implemented",
            "three_condition_multiseed_aggregator": "implemented",
            "full_image_qwen_sidecars": "awaiting_api_credentials",
            "three_seed_qwen_training": "not_started",
            "hardware_sensor_array": "not_implemented",
            "rgb_d_input": "not_implemented",
        },
        "scene_rule_data": {
            "source_image_count": 5990,
            "split_counts": validation["split_counts"],
            "observations_sha256": validation["observations_sha256"],
            "annotation_inputs": False,
            "formal_performance_evidence": False,
        },
        "smoke_result": {
            "epochs": 1,
            "source_image_count": 570,
            "external_minus_fallback_box_map50_95": deltas[
                "metrics/mAP50-95(B)"
            ],
            "external_minus_fallback_mask_map50_95": deltas[
                "metrics/mAP50-95(M)"
            ],
            "matched_minus_shuffled_box_map50_95": shuffled_deltas[
                "metrics/mAP50-95(B)"
            ],
            "matched_minus_shuffled_mask_map50_95": shuffled_deltas[
                "metrics/mAP50-95(M)"
            ],
            "shuffled_self_assignment_count": condition_contract[
                "self_assignment_count"
            ],
            "formal_performance_evidence": False,
        },
        "next_gate": {
            "required_environment_variable": "DASHSCOPE_API_KEY",
            "minimum_qwen_coverage": 0.99,
            "required_valid_record_fraction": 1.0,
            "planned_training_seeds": [42, 43, 44],
            "training_workers": 4,
        },
        "performance_configuration": {
            "workers_from": 0,
            "workers_to": 4,
            "batch_size": 32,
            "one_epoch_training_speedup": worker_equivalence["timing"][
                "speedup"
            ],
            "model_tensor_exact_equivalence": True,
        },
        "claim_boundaries": {
            "qwen_is_hardware": False,
            "rule_smoke_is_formal_metric": False,
            "external_blind_validation": False,
            "rgb_d_input": False,
            "full_patent_reproduction": False,
        },
        "source_bindings": {
            name: _binding(path) for name, path in paths.items()
        },
    }
    _atomic_write(
        output_path,
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 整图外部传感器联合头 v2 扩展索引",
        "",
        "- 状态：接口与 rule smoke 完成，正式 Qwen 数据待 API 环境变量",
        "- 无标注整图 sidecar：5,990 条，错误 0",
        "- YOLO 主干输入：始终为 3 通道 RGB",
        "- 外部 4×4 网格：在 P3/P4/P5 进入 Segment head 前融合",
        "- 缺失回退：RGB 预测 4×4 网格",
        "- rule 一轮 smoke：不属于正式论文指标",
        (
            "- workers 0→4 等价性：595 个模型张量逐元素一致，"
            f"一轮训练提速 {worker_equivalence['timing']['speedup']:.3f}×"
        ),
        "",
        (
            "同一 smoke 检查点下，外部 rule 观测相对 RGB 回退的 box/mask "
            f"mAP50-95 变化为 {deltas['metrics/mAP50-95(B)'] * 100:+.4f}/"
            f"{deltas['metrics/mAP50-95(M)'] * 100:+.4f} 个百分点。"
            "该数值只证明链路可运行。"
        ),
        (
            "同一检查点的 split 内循环乱序负对照保持传感器分布且无自配对；"
            "匹配观测相对乱序观测的 box/mask mAP50-95 变化为 "
            f"{shuffled_deltas['metrics/mAP50-95(B)'] * 100:+.4f}/"
            f"{shuffled_deltas['metrics/mAP50-95(M)'] * 100:+.4f} "
            "个百分点。该微小差异表明一轮 rule smoke 尚未形成可主张的传感器依赖。"
        ),
        "",
        (
            "Qwen 三种子正式协议已经冻结；当前缺少 `DASHSCOPE_API_KEY`，"
            "所以 Qwen 覆盖率、正式训练和结果仍为未完成。"
        ),
        "",
    ]
    _atomic_write(output_path.with_suffix(".md"), "\n".join(markdown))
    return index


def main() -> None:
    index = build_index(parse_args().output)
    print(
        json.dumps(
            {
                "state": index["state"],
                "valid": index["valid"],
                "source_binding_count": len(index["source_bindings"]),
                "qwen_status": index["capability_status"][
                    "full_image_qwen_sidecars"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
