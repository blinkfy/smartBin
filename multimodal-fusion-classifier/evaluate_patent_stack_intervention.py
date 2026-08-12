"""Evaluate a same-checkpoint v6 final-conditioning causal intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from audit_external_sensor_joint_seg import validate_source_bindings
from evaluate_reliable_external_conditions import (
    _atomic_json,
    _binding,
    _final_gate_summary,
    _gate_collector,
    _metric_deltas,
    _read_json,
    _resolve,
)
from mmfc.model import PROJECT_ROOT
from prepare_patent_stack_intervention import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA,
    DEFAULT_OBSERVATIONS,
    DEFAULT_OUTPUT as DEFAULT_PROTOCOL,
    PROTOCOL_NAME,
    RUN_ROOT,
)
from train_yolo_seg import _json_safe, _runtime_environment


DEFAULT_OUTPUT = RUN_ROOT / "PATENT_STACK_INTERVENTION.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the v6 final-conditioning intervention"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def expected_modulator_state_keys() -> set[str]:
    return {
        f"joint_fusion.patent_scale_modulators.{index}.{suffix}"
        for index in range(3)
        for suffix in ("weight", "bias")
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = (
        tensor.detach()
        .contiguous()
        .cpu()
        .numpy()
        .tobytes(order="C")
    )
    return hashlib.sha256(value).hexdigest()


def state_hashes(model: Any) -> dict[str, str]:
    return {
        name: _tensor_sha256(tensor)
        for name, tensor in model.state_dict().items()
    }


def changed_state_keys(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> set[str]:
    if before.keys() != after.keys():
        raise ValueError("model state key set changed during intervention")
    return {name for name in before if before[name] != after[name]}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    from mmfc.joint_seg_external import load_scene_sensor_index
    from mmfc.joint_seg_patent import (
        PatentAlignedReliableJointSegmentationModel,
    )
    from mmfc.joint_seg_reliable import (
        ReliableExternalSensorSegmentationValidator,
    )

    protocol_path = _resolve(args.protocol)
    output = _resolve(args.output)
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"intervention protocol is missing: {protocol_path}"
        )
    protocol = _read_json(protocol_path)
    errors = validate_source_bindings(protocol.get("source_bindings", {}))
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or protocol.get("status")
        != "frozen_before_intervention_evaluation"
        or protocol.get("formal_paper_metric") is not False
    ):
        errors.append("intervention protocol contract mismatch")
    checkpoint = _resolve(DEFAULT_CHECKPOINT)
    observations = _resolve(DEFAULT_OBSERVATIONS)
    data = _resolve(DEFAULT_DATA)
    condition_report_path = RUN_ROOT / "RELIABILITY_CONDITIONS.json"
    condition_report = _read_json(condition_report_path)
    if (
        condition_report.get("valid") is not True
        or condition_report.get("formal_paper_metric") is not False
    ):
        errors.append("bound full-condition report is invalid")
    if errors:
        raise ValueError("; ".join(errors))

    model = YOLO(str(checkpoint))
    if not isinstance(
        model.model, PatentAlignedReliableJointSegmentationModel
    ):
        raise TypeError("checkpoint does not persist the v6 model class")
    before = state_hashes(model.model)
    original_norms: dict[str, float] = {}
    with torch.no_grad():
        for index, module in enumerate(
            model.model.joint_fusion.patent_scale_modulators
        ):
            original_norms[f"{index}.weight"] = float(
                module.weight.detach().float().norm().cpu()
            )
            original_norms[f"{index}.bias"] = float(
                module.bias.detach().float().norm().cpu()
            )
            module.weight.zero_()
            module.bias.zero_()
    after = state_hashes(model.model)
    changed = changed_state_keys(before, after)
    expected = expected_modulator_state_keys()
    if changed != expected:
        raise ValueError(
            "intervention changed unexpected state tensors: "
            f"{sorted(changed.symmetric_difference(expected))}"
        )
    zero_norms = {
        f"{index}.{suffix}": float(
            getattr(module, suffix).detach().float().norm().cpu()
        )
        for index, module in enumerate(
            model.model.joint_fusion.patent_scale_modulators
        )
        for suffix in ("weight", "bias")
    }
    if any(value != 0.0 for value in zero_norms.values()):
        raise ValueError("intervened modulator tensor is not zero")

    runtime = protocol["runtime"]
    matched_index = load_scene_sensor_index(observations)
    validators = []

    def validator_factory(*factory_args: Any, **factory_kwargs: Any):
        validator = ReliableExternalSensorSegmentationValidator(
            *factory_args,
            **factory_kwargs,
            sensor_observations=matched_index,
            degradation={"enabled": False},
            degradation_seed=int(runtime["seed"]),
        )
        validators.append(validator)
        return validator

    captured, hook = _gate_collector(model)
    try:
        metrics = model.val(
            validator=validator_factory,
            data=str(data),
            split="test",
            imgsz=int(runtime["image_size"]),
            batch=int(runtime["batch_size"]),
            device=str(runtime["device"]),
            workers=int(runtime["workers"]),
            project=str(output.parent / "intervention-runs"),
            name="head_conditioning_zeroed",
            exist_ok=True,
            plots=False,
            save_json=False,
            verbose=False,
        )
    finally:
        hook.remove()
    if (
        not validators
        or validators[0].external_sensor_sample_count != 570
    ):
        raise ValueError("intervention did not consume 570 test images")
    checkpoint_sha_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_binding = condition_report["bindings"]["checkpoint"]
    if checkpoint_sha_after != checkpoint_binding["sha256"]:
        raise ValueError("checkpoint file changed during intervention")

    full = condition_report["conditions"]["clean_matched"]
    bypass_metrics = _json_safe(metrics.results_dict)
    full_minus_bypass = _metric_deltas(
        full["metrics"], bypass_metrics
    )
    primary_effect_max_abs = max(
        abs(full_minus_bypass["metrics/mAP50-95(B)"]),
        abs(full_minus_bypass["metrics/mAP50-95(M)"]),
    )
    interpretation = (
        "measurable_descriptive_same_checkpoint_effect"
        if primary_effect_max_abs >= 1e-4
        else "negligible_at_0.01_percentage_point_resolution"
    )
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "evidence_role": "same_checkpoint_posthoc_causal_intervention",
        "retrained_ablation": False,
        "test_source_image_count": 570,
        "conditions": {
            "full_clean": full,
            "head_conditioning_zeroed": {
                "metrics": bypass_metrics,
                "speed_ms_per_image": _json_safe(metrics.speed),
                "gate": _final_gate_summary(
                    captured, sample_count=570
                ),
                "degradation": {"enabled": False},
            },
        },
        "comparison": {
            "full_clean_minus_head_conditioning_zeroed": (
                full_minus_bypass
            ),
            "maximum_absolute_box_or_mask_map50_95_effect": (
                primary_effect_max_abs
            ),
            "interpretation": interpretation,
        },
        "intervention_audit": {
            "model_class": type(model.model).__name__,
            "state_tensor_count": len(before),
            "changed_state_tensor_count": len(changed),
            "changed_state_keys": sorted(changed),
            "expected_state_keys_exact": changed == expected,
            "original_modulator_l2": original_norms,
            "intervened_modulator_l2": zero_norms,
            "all_other_state_tensors_unchanged": True,
            "modified_checkpoint_saved": False,
            "checkpoint_sha256_unchanged": True,
        },
        "runtime": _runtime_environment(),
        "claim_boundaries": protocol["claim_boundaries"],
        "bindings": {
            "checkpoint": _binding(checkpoint),
            "observations": _binding(observations),
            "data": _binding(data),
            "protocol": _binding(protocol_path),
            "full_condition_report": _binding(condition_report_path),
            "evaluator_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_json(output, result)
    markdown = [
        "# v6 同检查点头前调制干预",
        "",
        "- 有效：`true`；正式论文指标：`false`；独立重训练消融：`false`。",
        "- 同一检查点仅在内存中归零三个尺度调制器的weight与bias，共6个状态张量。",
        "- 其余模型状态逐张量不变，检查点文件哈希不变，未保存修改后检查点。",
        "- 完整相对归零条件的box/mask mAP50-95差异："
        f"{100.0 * full_minus_bypass['metrics/mAP50-95(B)']:+.3f}/"
        f"{100.0 * full_minus_bypass['metrics/mAP50-95(M)']:+.3f}个百分点。",
        f"- 解释：`{interpretation}`。",
        "",
        "该结果只回答已训练v6最终头前调制是否对任务输出存在因果作用。"
        "它不等价于独立重训组件消融，也不证明实体传感器或Qwen sidecar增益。",
        "",
    ]
    temporary = output.with_suffix(".md.tmp")
    temporary.write_text("\n".join(markdown), encoding="utf-8")
    temporary.replace(output.with_suffix(".md"))
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "changed_state_tensor_count": result[
                    "intervention_audit"
                ]["changed_state_tensor_count"],
                "comparison": result["comparison"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
