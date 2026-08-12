"""Audit v3-v5 external-sensor reliability development evidence."""

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
from mmfc.model import PROJECT_ROOT


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
PRIMARY_NAME = (
    "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
)
CALIBRATED_NAME = (
    "yolo11s-seg-joint-spatial-calibrated-rule-dev-seed42"
)
DEFAULT_OUTPUT = REPORTS / "SPATIAL_RELIABILITY_AUDIT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit reliability-aware external sensor development"
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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _condition_metrics(report: Mapping[str, Any], name: str):
    return report["conditions"][name]["metrics"]


def select_development_candidate(
    primary: Mapping[str, Any],
    calibrated: Mapping[str, Any],
) -> dict[str, Any]:
    primary_clean = _condition_metrics(primary, "clean_matched")
    calibrated_clean = _condition_metrics(calibrated, "clean_matched")
    calibrated_shuffled = _condition_metrics(calibrated, "shuffled")
    primary_better_clean = (
        float(primary_clean["metrics/mAP50-95(B)"])
        > float(calibrated_clean["metrics/mAP50-95(B)"])
        and float(primary_clean["metrics/mAP50-95(M)"])
        > float(calibrated_clean["metrics/mAP50-95(M)"])
    )
    calibrated_shuffled_not_lower = (
        float(calibrated_shuffled["metrics/mAP50-95(B)"])
        >= float(calibrated_clean["metrics/mAP50-95(B)"])
        and float(calibrated_shuffled["metrics/mAP50-95(M)"])
        >= float(calibrated_clean["metrics/mAP50-95(M)"])
    )
    return {
        "selected": "spatial_v5_primary",
        "primary_better_on_clean_box_and_mask": primary_better_clean,
        "calibrated_shuffled_not_lower_than_clean": (
            calibrated_shuffled_not_lower
        ),
        "calibrated_rejected": (
            primary_better_clean and calibrated_shuffled_not_lower
        ),
    }


def audit(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    from mmfc.joint_seg_reliable import pack_reliable_sensor_channels
    from mmfc.joint_seg_spatial import (
        SpatialReliableExternalSensorJointSegmentationModel,
    )

    output = _resolve(output)
    spatial_root = REPORTS / "joint-multimodal-seg-spatial"
    primary_root = spatial_root / PRIMARY_NAME
    calibrated_root = spatial_root / CALIBRATED_NAME
    calibrated_eval_root = spatial_root / f"{CALIBRATED_NAME}-workers0-eval"
    paths = {
        "auditor_source": Path(__file__),
        "primary_protocol": primary_root / "PROTOCOL.json",
        "primary_evaluation": primary_root / "evaluation.json",
        "primary_conditions": primary_root / "RELIABILITY_CONDITIONS.json",
        "primary_separation": primary_root / "SENSOR_SEPARATION.json",
        "primary_checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg-spatial"
            / PRIMARY_NAME
            / "weights"
            / "best.pt"
        ),
        "v3_conditions": (
            REPORTS
            / "joint-multimodal-seg-reliable"
            / "yolo11s-seg-joint-reliable-rule-dev-seed42-retry4"
            / "RELIABILITY_CONDITIONS.json"
        ),
        "v4_conditions": (
            REPORTS
            / "joint-multimodal-seg-contrastive"
            / "yolo11s-seg-joint-reliable-contrastive-rule-dev-seed42"
            / "RELIABILITY_CONDITIONS.json"
        ),
        "calibrated_protocol": calibrated_root / "PROTOCOL.json",
        "calibrated_evaluation": calibrated_root / "evaluation.json",
        "calibrated_conditions": (
            calibrated_eval_root / "RELIABILITY_CONDITIONS.json"
        ),
        "calibrated_checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg-spatial"
            / CALIBRATED_NAME
            / "weights"
            / "best.pt"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "spatial reliability audit inputs missing: " + ", ".join(missing)
        )

    protocol = _read_json(paths["primary_protocol"])
    evaluation = _read_json(paths["primary_evaluation"])
    primary = _read_json(paths["primary_conditions"])
    separation = _read_json(paths["primary_separation"])
    v3 = _read_json(paths["v3_conditions"])
    v4 = _read_json(paths["v4_conditions"])
    calibrated_protocol = _read_json(paths["calibrated_protocol"])
    calibrated_evaluation = _read_json(paths["calibrated_evaluation"])
    calibrated = _read_json(paths["calibrated_conditions"])
    errors = validate_source_bindings(protocol.get("source_bindings", {}))
    errors.extend(
        validate_source_bindings(
            calibrated_protocol.get("source_bindings", {})
        )
    )

    primary_checkpoint_sha = _sha256(paths["primary_checkpoint"])
    calibrated_checkpoint_sha = _sha256(paths["calibrated_checkpoint"])
    if (
        protocol.get("protocol")
        != "joint_text_external_scene_sensor_spatial_yolo_segment_v5"
        or protocol.get("formal_paper_metric") is not False
        or evaluation.get("valid") is not True
        or evaluation.get("checkpoint", {}).get("sha256")
        != primary_checkpoint_sha
        or primary.get("valid") is not True
        or primary.get("test_source_image_count") != 570
        or separation.get("valid") is not True
        or separation.get("sample_count_per_population") != 570
    ):
        errors.append("primary v5 protocol or evidence binding mismatch")
    if (
        calibrated_protocol.get("protocol")
        != "joint_text_external_scene_sensor_spatial_yolo_segment_v5"
        or calibrated_evaluation.get("checkpoint", {}).get("sha256")
        != calibrated_checkpoint_sha
        or calibrated.get("valid") is not True
        or calibrated.get("test_source_image_count") != 570
    ):
        errors.append("calibrated v5 evidence binding mismatch")
    for label, report in (("v3", v3), ("v4", v4), ("v5", primary)):
        if (
            report.get("formal_paper_metric") is not False
            or report.get("state") != "complete"
            or "clean_matched" not in report.get("conditions", {})
            or "shuffled" not in report.get("conditions", {})
            or "rgb_fallback" not in report.get("conditions", {})
        ):
            errors.append(f"{label} condition evidence is incomplete")

    model_wrapper = YOLO(str(paths["primary_checkpoint"]))
    model = model_wrapper.model.eval()
    if not isinstance(
        model, SpatialReliableExternalSensorJointSegmentationModel
    ):
        errors.append("primary checkpoint does not persist the v5 model class")
    first_layer_channels: list[int] = []

    def capture_input(_module, inputs):
        first_layer_channels.append(int(inputs[0].shape[1]))

    handle = model.model[0].conv.register_forward_pre_hook(capture_input)
    rgb_uint8 = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
    grid = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4)
    packed = pack_reliable_sensor_channels(
        rgb_uint8, grid, torch.ones(1, 1)
    ).float() / 255.0
    with torch.inference_mode():
        _ = model(packed)
    external_aux = {
        key: value.detach().float().cpu()
        for key, value in model.joint_fusion.last_aux.items()
        if isinstance(value, torch.Tensor)
    }
    with torch.inference_mode():
        _ = model(packed[:, :3])
    fallback_aux = {
        key: value.detach().float().cpu()
        for key, value in model.joint_fusion.last_aux.items()
        if isinstance(value, torch.Tensor)
    }
    handle.remove()
    expected_grid = (grid * 255.0).round() / 255.0
    if first_layer_channels != [3, 3]:
        errors.append("five-channel transport reached the RGB backbone")
    if not torch.allclose(
        external_aux["observed_sensor_grid"],
        expected_grid,
        atol=1e-5,
        rtol=0.0,
    ):
        errors.append("packed sensor grid was not preserved")
    if (
        not bool((external_aux["sensor_available"] == 1).all())
        or not bool((fallback_aux["sensor_available"] == 0).all())
        or not bool((fallback_aux["sensor_reliability"] == 0).all())
    ):
        errors.append("availability or RGB fallback contract failed")
    spatial_norms = {
        name: float(parameter.detach().float().norm().cpu())
        for name, parameter in model.joint_fusion.named_parameters()
        if name.startswith("sensor_spatial")
    }
    if not spatial_norms or not all(
        torch.isfinite(torch.tensor(value)) and value > 0.0
        for value in spatial_norms.values()
    ):
        errors.append("spatial sensor predictor is inactive")

    selection = select_development_candidate(primary, calibrated)
    if not selection["calibrated_rejected"]:
        errors.append("candidate selection contract was not satisfied")
    primary_clean = _condition_metrics(primary, "clean_matched")
    primary_fallback = _condition_metrics(primary, "rgb_fallback")
    primary_shuffled = _condition_metrics(primary, "shuffled")
    v3_clean = _condition_metrics(v3, "clean_matched")
    v3_shuffled = _condition_metrics(v3, "shuffled")
    v4_clean = _condition_metrics(v4, "clean_matched")
    v4_shuffled = _condition_metrics(v4, "shuffled")
    auc = separation["metrics"]
    if not (
        0.5 < float(auc["discrepancy_auc_shuffled_positive"]) < 1.0
        and 0.5 < float(auc["consistency_auc_shuffled_positive"]) < 1.0
        and 0.5 < float(auc["raw_reliability_auc_shuffled_positive"]) < 1.0
    ):
        errors.append("matched/shuffled separation is not above chance")

    outcome = {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "formal_paper_metric": False,
        "evidence_role": "rule_sidecar_mechanism_development",
        "selected_development_model": selection,
        "structural_audit": {
            "persisted_model_class": type(model).__name__,
            "packed_input_channels": 5,
            "rgb_backbone_input_channels": first_layer_channels,
            "spatial_sensor_parameter_l2": spatial_norms,
            "clean_sensor_available": True,
            "missing_sensor_reliability_zero": True,
            "transport_grid_preserved": "8-bit quantized",
        },
        "primary_v5_test": {
            "source_image_count": 570,
            "clean_metrics": primary_clean,
            "rgb_fallback_metrics": primary_fallback,
            "shuffled_metrics": primary_shuffled,
            "clean_minus_rgb_fallback": metric_deltas(
                primary_clean, primary_fallback
            ),
            "clean_minus_shuffled": metric_deltas(
                primary_clean, primary_shuffled
            ),
            "gate_ordering_all_passed": primary.get(
                "gate_ordering_all_passed"
            ),
        },
        "matched_shuffled_diagnostic": auc,
        "development_lineage": {
            "v3_clean_minus_shuffled": metric_deltas(
                v3_clean, v3_shuffled
            ),
            "v4_clean_minus_shuffled": metric_deltas(
                v4_clean, v4_shuffled
            ),
            "v5_clean_minus_shuffled": metric_deltas(
                primary_clean, primary_shuffled
            ),
        },
        "rejected_overcalibrated_run": {
            "checkpoint_sha256": calibrated_checkpoint_sha,
            "reason": (
                "clean box and mask mAP50-95 decreased, while shuffled "
                "box and mask metrics were not lower than clean"
            ),
            "clean_minus_shuffled": metric_deltas(
                _condition_metrics(calibrated, "clean_matched"),
                _condition_metrics(calibrated, "shuffled"),
            ),
            "workers4_evaluation_incident": {
                "state": "terminated_after_dataloader_stall",
                "completed_external_condition_directories": 6,
                "replacement_workers": 0,
                "replacement_report_complete": True,
            },
        },
        "claim_boundaries": [
            "All v3-v5 results are single-seed rule-sidecar development evidence.",
            "The matched-versus-shuffled effect remains small in task metrics.",
            "AUC diagnostics show moderate, not perfect, condition separability.",
            "Software degradation does not establish hardware robustness.",
            "The observations are image-conditioned soft sensors, not physical measurements.",
            "These results are excluded from the preregistered formal Qwen table.",
        ],
        "bindings": {
            name: _binding(path) for name, path in paths.items()
        },
        "errors": errors,
    }
    _atomic_json(output, outcome)
    markdown = [
        "# 空间可靠性融合开发审计",
        "",
        f"- 有效：`{outcome['valid']}`；正式论文指标：`false`。",
        f"- 主开发模型：`{PRIMARY_NAME}`。",
        "- 结构：RGB 主干仍只接收 3 通道；4×4 观测与可用性通过独立传输通道进入联合头。",
        "- v5 干净匹配相对 RGB 回退：框 "
        f"{outcome['primary_v5_test']['clean_minus_rgb_fallback']['metrics/mAP50-95(B)'] * 100:.3f} "
        "个百分点，掩码 "
        f"{outcome['primary_v5_test']['clean_minus_rgb_fallback']['metrics/mAP50-95(M)'] * 100:.3f} 个百分点。",
        "- 匹配/乱序偏差 AUC："
        f"{auc['discrepancy_auc_shuffled_positive']:.3f}；"
        "可靠性 AUC："
        f"{auc['raw_reliability_auc_shuffled_positive']:.3f}。",
        "- 强化校准版未晋升：干净精度下降，且乱序任务指标未低于干净。",
        "",
        "该结果只能作为 rule sidecar 的机制开发证据。匹配与乱序在任务指标上的差距仍小，"
        "不能据此宣称已经验证真实硬件传感器的跨模态增益或鲁棒性。",
        "",
    ]
    markdown_path = output.with_suffix(".md")
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(markdown), encoding="utf-8")
    temporary.replace(markdown_path)
    return outcome


def main() -> None:
    result = audit(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "selected": result["selected_development_model"][
                    "selected"
                ],
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
