"""Audit the v6 same-checkpoint final-conditioning intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_external_sensor_joint_seg import (
    metric_deltas,
    validate_source_bindings,
)
from audit_patent_aligned_joint_seg import validate_named_bindings
from mmfc.model import PROJECT_ROOT
from prepare_patent_stack_intervention import (
    DEFAULT_OUTPUT as DEFAULT_PROTOCOL,
    PROTOCOL_NAME,
    RUN_ROOT,
)


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_INTERVENTION = RUN_ROOT / "PATENT_STACK_INTERVENTION.json"
DEFAULT_OUTPUT = REPORTS / "PATENT_STACK_INTERVENTION_AUDIT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the v6 same-checkpoint intervention"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--intervention",
        type=Path,
        default=DEFAULT_INTERVENTION,
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


def classify_contribution_pattern(
    stack_effect: Mapping[str, float],
    sidecar_effect_max_abs: float,
    *,
    resolution: float = 1e-4,
) -> str:
    stack_effect_max_abs = max(
        abs(float(stack_effect["metrics/mAP50-95(B)"])),
        abs(float(stack_effect["metrics/mAP50-95(M)"])),
    )
    if (
        stack_effect_max_abs >= resolution
        and sidecar_effect_max_abs < resolution
    ):
        return "collective_stack_effect_without_measurable_sidecar_effect"
    if (
        stack_effect_max_abs < resolution
        and sidecar_effect_max_abs < resolution
    ):
        return "neither_effect_measurable_at_resolution"
    if sidecar_effect_max_abs >= resolution:
        return "sidecar_effect_measurable_at_resolution"
    return "mixed_or_unresolved"


def audit(
    protocol_path: Path = DEFAULT_PROTOCOL,
    intervention_path: Path = DEFAULT_INTERVENTION,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    intervention_path = _resolve(intervention_path)
    output = _resolve(output)
    v6_outcome_path = RUN_ROOT / "OUTCOME.json"
    paths = {
        "protocol": protocol_path,
        "intervention": intervention_path,
        "v6_outcome": v6_outcome_path,
        "auditor_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "intervention audit inputs missing: " + ", ".join(missing)
        )
    protocol = _read_json(protocol_path)
    intervention = _read_json(intervention_path)
    v6_outcome = _read_json(v6_outcome_path)
    errors = validate_source_bindings(protocol.get("source_bindings", {}))
    errors.extend(
        validate_named_bindings(intervention.get("bindings", {}))
    )
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or protocol.get("status")
        != "frozen_before_intervention_evaluation"
        or protocol.get("formal_paper_metric") is not False
        or intervention.get("valid") is not True
        or intervention.get("formal_paper_metric") is not False
        or intervention.get("retrained_ablation") is not False
        or intervention.get("test_source_image_count") != 570
        or v6_outcome.get("valid") is not True
    ):
        errors.append("intervention protocol or result contract mismatch")
    state_audit = intervention.get("intervention_audit", {})
    if (
        state_audit.get("changed_state_tensor_count") != 6
        or state_audit.get("expected_state_keys_exact") is not True
        or state_audit.get("all_other_state_tensors_unchanged") is not True
        or state_audit.get("modified_checkpoint_saved") is not False
        or state_audit.get("checkpoint_sha256_unchanged") is not True
        or any(
            float(value) != 0.0
            for value in state_audit.get(
                "intervened_modulator_l2", {}
            ).values()
        )
    ):
        errors.append("intervention state isolation contract mismatch")
    full_metrics = intervention["conditions"]["full_clean"]["metrics"]
    zeroed_metrics = intervention["conditions"][
        "head_conditioning_zeroed"
    ]["metrics"]
    recomputed = metric_deltas(full_metrics, zeroed_metrics)
    recorded = intervention["comparison"][
        "full_clean_minus_head_conditioning_zeroed"
    ]
    if any(
        abs(float(recomputed[key]) - float(recorded[key])) > 1e-12
        for key in recomputed
    ):
        errors.append("recorded intervention deltas differ from metrics")
    sidecar_effect = float(
        v6_outcome["test"][
            "maximum_absolute_box_or_mask_map50_95_effect"
        ]
    )
    pattern = classify_contribution_pattern(recorded, sidecar_effect)
    if (
        pattern
        != "collective_stack_effect_without_measurable_sidecar_effect"
    ):
        errors.append("observed contribution pattern differs from audit rule")
    if (
        float(recorded["metrics/mAP50-95(B)"]) <= 0.0
        or float(recorded["metrics/mAP50-95(M)"]) <= 0.0
    ):
        errors.append("full conditioning is not higher on both primary metrics")

    result = {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "formal_paper_metric": False,
        "evidence_role": "same_checkpoint_posthoc_causal_intervention_audit",
        "contribution_decomposition": {
            "pattern": pattern,
            "full_stack_conditioning_effect": {
                "box_map50_95": recorded["metrics/mAP50-95(B)"],
                "mask_map50_95": recorded["metrics/mAP50-95(M)"],
                "interpretation": (
                    "measurable_descriptive_same_checkpoint_effect"
                ),
            },
            "matched_sidecar_specific_effect_max_abs": sidecar_effect,
            "matched_sidecar_interpretation": v6_outcome["test"][
                "effect_interpretation"
            ],
            "supported_conclusion": (
                "The trained collective patent-stack head conditioning "
                "has a measurable same-checkpoint task effect, while the "
                "matched external sidecar has no measurable aggregate "
                "effect at 0.01 percentage-point resolution."
            ),
            "unsupported_attribution": [
                "The collective effect cannot be attributed specifically to the sensor branch.",
                "The intervention is not an independently retrained component ablation.",
                "The result does not establish Qwen-sidecar or hardware-sensor benefit.",
            ],
        },
        "state_isolation": state_audit,
        "claim_boundaries": [
            "The intervention changes only six final modulator tensors in memory.",
            "The unmodified checkpoint file remains hash-identical.",
            "The comparison is posthoc, same-checkpoint, single-seed development evidence.",
            "The collective effect includes visual, text, sensor, adapter, attention, and scale computations.",
            "Sensor-specific task effects remain negligible under matched, fallback, and shuffled sidecars.",
        ],
        "bindings": {name: _binding(path) for name, path in paths.items()},
        "errors": errors,
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# v6 头前调制干预审计",
        "",
        f"- 有效：`{result['valid']}`；正式论文指标：`false`。",
        "- 状态隔离：仅6个预注册调制器张量在内存中变化，其余状态与检查点文件不变。",
        "- 完整调制相对归零条件：box/mask mAP50-95 "
        f"{100.0 * recorded['metrics/mAP50-95(B)']:+.3f}/"
        f"{100.0 * recorded['metrics/mAP50-95(M)']:+.3f}个百分点。",
        "- matched sidecar特异效应最大绝对值："
        f"{100.0 * sidecar_effect:.4f}个百分点，低于0.01个百分点分辨率。",
        "",
        "结论：已训练专利融合栈的整体头前调制具有可测的同检查点任务作用，"
        "但匹配sidecar本身没有可测的汇总任务效应。整体作用不能归因于传感器分支，"
        "该实验也不是独立重训练消融。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return result


def main() -> None:
    args = parse_args()
    result = audit(args.protocol, args.intervention, args.output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "pattern": result["contribution_decomposition"]["pattern"],
                "errors": result["errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
