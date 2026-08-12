"""Audit the external-scene-sensor joint-head smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mmfc.model import PROJECT_ROOT


DEFAULT_RUN_NAME = "yolo11s-seg-joint-external-rule-smoke"
DEFAULT_REPORT_RUN = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
    / DEFAULT_RUN_NAME
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit external-sensor joint-head integration"
    )
    parser.add_argument("--report-run", type=Path, default=DEFAULT_REPORT_RUN)
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


def validate_source_bindings(
    bindings: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not bindings:
        return ["protocol source bindings are missing"]
    for raw_path, binding in bindings.items():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            errors.append(f"protocol source is missing: {raw_path}")
            continue
        if (
            not isinstance(binding, Mapping)
            or path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            errors.append(f"protocol source binding drift: {raw_path}")
    return errors


def metric_deltas(
    external: Mapping[str, float],
    fallback: Mapping[str, float],
) -> dict[str, float]:
    keys = (
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95(M)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
    )
    return {
        key: float(external[key]) - float(fallback[key]) for key in keys
    }


def audit(report_run: Path = DEFAULT_REPORT_RUN) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    from mmfc.joint_seg_external import (
        ExternalSensorJointSegmentationModel,
        pack_scene_sensor_channel,
    )

    report_run = _resolve(report_run)
    protocol_path = report_run / "PROTOCOL.json"
    evaluation_path = report_run / "evaluation.json"
    errors: list[str] = []
    if not protocol_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("protocol or evaluation is missing")
    protocol = _read_json(protocol_path)
    evaluation = _read_json(evaluation_path)
    errors.extend(validate_source_bindings(protocol.get("source_bindings", {})))
    checkpoint = Path(evaluation.get("checkpoint", {}).get("path", ""))
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size
        != evaluation.get("checkpoint", {}).get("size_bytes")
        or _sha256(checkpoint)
        != evaluation.get("checkpoint", {}).get("sha256")
    ):
        errors.append("checkpoint binding mismatch")
    sensor_path = Path(
        evaluation.get("sensor_observations", {}).get("path", "")
    )
    if (
        not sensor_path.is_file()
        or sensor_path.stat().st_size
        != evaluation.get("sensor_observations", {}).get("size_bytes")
        or _sha256(sensor_path)
        != evaluation.get("sensor_observations", {}).get("sha256")
    ):
        errors.append("sensor observation binding mismatch")
    leakage = evaluation.get("leakage_control", {})
    if (
        protocol.get("protocol")
        != "joint_text_external_scene_sensor_yolo_segment_v2"
        or evaluation.get("valid") is not True
        or evaluation.get("external_sensor", {}).get(
            "metric_source_image_count"
        )
        != 570
        or leakage.get("metric_forward_inputs")
        != ["RGB", "full_image_4x4_soft_sensor"]
        or leakage.get("ground_truth_boxes_consumed_by_sensor") is not False
        or leakage.get("ground_truth_masks_consumed_by_sensor") is not False
        or leakage.get("ground_truth_classes_consumed_by_sensor") is not False
        or leakage.get("annotation_conditioned_spatial_projection") is not False
    ):
        errors.append("metric-forward or leakage contract mismatch")
    source = protocol.get("sensor_source", {})
    if (
        source.get("mode") != "rule"
        or source.get("full_image_only") is not True
        or source.get("annotation_files_read") is not False
        or source.get("bbox_inputs") is not False
        or source.get("polygon_inputs") is not False
        or source.get("class_inputs") is not False
    ):
        errors.append("scene sensor source contract mismatch")

    model_wrapper = YOLO(str(checkpoint))
    model = model_wrapper.model.eval()
    if not isinstance(model, ExternalSensorJointSegmentationModel):
        errors.append("checkpoint does not persist the external-sensor model")
    first_layer_channels: list[int] = []

    def capture_input(_module, inputs):
        first_layer_channels.append(int(inputs[0].shape[1]))

    handle = model.model[0].conv.register_forward_pre_hook(capture_input)
    rgb_uint8 = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
    grid = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4)
    packed = pack_scene_sensor_channel(rgb_uint8, grid).float() / 255.0
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
    if first_layer_channels != [3, 3]:
        errors.append(
            f"RGB backbone input channels differ from [3,3]: "
            f"{first_layer_channels}"
        )
    if not bool((external_aux["external_sensor_used"] == 1).all()):
        errors.append("packed sensor was not used in structural forward")
    if not bool((fallback_aux["external_sensor_used"] == 0).all()):
        errors.append("RGB fallback did not use the predicted sensor")
    expected_grid = (grid * 255.0).round() / 255.0
    if not torch.allclose(
        external_aux["effective_sensor_grid"],
        expected_grid,
        atol=1e-5,
        rtol=0.0,
    ):
        errors.append("effective external grid differs after transport")
    sensor_output_norms = [
        float(module.weight.detach().norm().cpu())
        for module in model.joint_fusion.sensor_output
    ]
    if not all(value > 0.0 for value in sensor_output_norms):
        errors.append("trained sensor residual output is inactive")

    fallback = evaluation.get("rgb_fallback")
    external_metrics = evaluation["external_sensor"]["metrics"]
    if not isinstance(fallback, Mapping):
        errors.append("RGB fallback evaluation is missing")
        deltas = {}
    else:
        deltas = metric_deltas(external_metrics, fallback["metrics"])
    outcome = {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "evidence_role": "integration_smoke_only",
        "formal_paper_metric": False,
        "training_epochs": int(
            evaluation.get("training", {}).get("completed_epochs", 0)
        ),
        "sensor_source": {
            "mode": source.get("mode"),
            "full_image_only": True,
            "annotation_inputs": False,
            "qwen_success_count": 0,
        },
        "structural_audit": {
            "persisted_model_class": type(model).__name__,
            "packed_input_channels": 4,
            "rgb_backbone_input_channels": first_layer_channels,
            "external_sensor_used": True,
            "missing_sensor_fallback_used": True,
            "transport_grid_preserved": "8-bit quantized",
            "sensor_output_weight_l2": sensor_output_norms,
        },
        "test": {
            "source_image_count": 570,
            "external_sensor_metrics": external_metrics,
            "rgb_fallback_metrics": (
                fallback["metrics"] if isinstance(fallback, Mapping) else None
            ),
            "external_minus_fallback": deltas,
        },
        "claim_boundaries": [
            "One epoch is an integration smoke and is excluded from formal tables.",
            "The current observation source is scene-rule-v1, not Qwen or hardware.",
            "The sidecar is generated from the full image without annotations.",
            "No RGB-D input or synchronized physical sensor is present.",
        ],
        "bindings": {
            "protocol": _binding(protocol_path),
            "evaluation": _binding(evaluation_path),
            "checkpoint": _binding(checkpoint),
            "sensor_observations": _binding(sensor_path),
            "auditor_source": _binding(Path(__file__)),
        },
    }
    output = report_run / "OUTCOME.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return outcome


def main() -> None:
    args = parse_args()
    outcome = audit(args.report_run)
    print(
        json.dumps(
            {
                "state": "complete" if outcome["valid"] else "invalid",
                "valid": outcome["valid"],
                "errors": outcome["errors"],
                "formal_paper_metric": outcome["formal_paper_metric"],
                "output": (
                    _resolve(args.report_run) / "OUTCOME.json"
                ).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not outcome["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
