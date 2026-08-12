"""Audit the v6 patent-aligned native Segment-head development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from audit_external_sensor_joint_seg import (
    metric_deltas,
    validate_source_bindings,
)
from mmfc.model import PROJECT_ROOT
from train_patent_aligned_external_sensor_joint_seg import (
    DEFAULT_NAME,
    PROTOCOL_NAME,
)


DEFAULT_REPORT_RUN = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
)
_WORD_NAMESPACE = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}
_CLAIM_HEADER = re.compile(r"^(\d{1,2})、(?:基于|根据)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the v6 patent-aligned joint Segment head"
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
    try:
        display_path = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = resolved.as_posix()
    return {
        "path": display_path,
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


def _patent_document() -> Path:
    matches = sorted(PROJECT_ROOT.glob("2026.4.3-1*.docx"))
    if len(matches) != 1:
        raise FileNotFoundError("expected exactly one 2026.4.3-1 patent DOCX")
    return matches[0]


def extract_patent_claims(path: Path) -> dict[int, str]:
    """Read claims 1-10 from WordprocessingML without a DOCX dependency."""

    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", _WORD_NAMESPACE):
        text = "".join(
            node.text or ""
            for node in paragraph.findall(".//w:t", _WORD_NAMESPACE)
        )
        normalized = " ".join(
            text.replace("\u200b", "").replace("\xa0", " ").split()
        )
        if normalized:
            paragraphs.append(normalized)

    claims: dict[int, list[str]] = {}
    current: int | None = None
    for paragraph in paragraphs:
        match = _CLAIM_HEADER.match(paragraph)
        if match:
            number = int(match.group(1))
            if 1 <= number <= 10:
                current = number
                claims.setdefault(number, []).append(paragraph)
                continue
        if current is None:
            continue
        if current == 10 and paragraph in {
            "基于多模态识别的医疗废品识别方法",
            "技术领域",
        }:
            break
        claims[current].append(paragraph)
    return {
        number: "\n".join(claims.get(number, []))
        for number in range(1, 11)
        if claims.get(number)
    }


def validate_named_bindings(bindings: Mapping[str, Any]) -> list[str]:
    """Validate reports whose bindings are keyed by a logical name."""

    errors: list[str] = []
    if not bindings:
        return ["named evidence bindings are missing"]
    for name, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping):
            errors.append(f"invalid named binding: {name}")
            continue
        raw_path = raw_binding.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"named binding path is missing: {name}")
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            errors.append(f"named binding source is missing: {name}")
            continue
        if (
            path.stat().st_size != raw_binding.get("size_bytes")
            or _sha256(path) != raw_binding.get("sha256")
        ):
            errors.append(f"named binding drift: {name}")
    return errors


def _auxiliary_copy(model: Any) -> dict[str, Any]:
    import torch

    return {
        key: value.detach().float().cpu()
        for key, value in model.joint_fusion.last_aux.items()
        if isinstance(value, torch.Tensor)
    }


def _feature_differences(
    left: list[Any],
    right: list[Any],
) -> tuple[list[float], list[float]]:
    maximum = [
        float((left_item - right_item).abs().max())
        for left_item, right_item in zip(left, right)
    ]
    mean = [
        float((left_item - right_item).abs().mean())
        for left_item, right_item in zip(left, right)
    ]
    return maximum, mean


def audit(report_run: Path = DEFAULT_REPORT_RUN) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    from mmfc.joint_seg_patent import (
        PatentAlignedReliableJointSegmentationModel,
    )
    from mmfc.joint_seg_reliable import pack_reliable_sensor_channels

    report_run = _resolve(report_run)
    paths = {
        "protocol": report_run / "PROTOCOL.json",
        "evaluation": report_run / "evaluation.json",
        "conditions": report_run / "RELIABILITY_CONDITIONS.json",
        "patent_document": _patent_document(),
        "auditor_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "patent-aligned audit inputs missing: " + ", ".join(missing)
        )

    protocol = _read_json(paths["protocol"])
    evaluation = _read_json(paths["evaluation"])
    conditions = _read_json(paths["conditions"])
    errors = validate_source_bindings(protocol.get("source_bindings", {}))
    errors.extend(validate_named_bindings(conditions.get("bindings", {})))

    checkpoint = Path(evaluation.get("checkpoint", {}).get("path", ""))
    sensor_observations = Path(
        evaluation.get("sensor_observations", {}).get("path", "")
    )
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size
        != evaluation.get("checkpoint", {}).get("size_bytes")
        or _sha256(checkpoint)
        != evaluation.get("checkpoint", {}).get("sha256")
    ):
        errors.append("checkpoint binding mismatch")
    if (
        not sensor_observations.is_file()
        or sensor_observations.stat().st_size
        != evaluation.get("sensor_observations", {}).get("size_bytes")
        or _sha256(sensor_observations)
        != evaluation.get("sensor_observations", {}).get("sha256")
    ):
        errors.append("sensor observation binding mismatch")

    claims = extract_patent_claims(paths["patent_document"])
    expected_claim_anchors = {
        1: ("S1", "S2", "S3", "S4"),
        2: ("双边滤波", "CLAHE", "拉普拉斯", "仿射"),
        3: ("YOLO11s-seg", "CLIP", "传感器"),
        4: ("三个独立", "拼接"),
        5: ("FILM",),
        6: ("余弦相似度",),
        7: ("softmax", "下投影"),
        8: ("双向注意力", "上投影"),
        9: ("残差",),
        10: ("多尺度", "动态调整"),
    }
    for number, anchors in expected_claim_anchors.items():
        claim = claims.get(number, "")
        if not claim or any(anchor not in claim for anchor in anchors):
            errors.append(f"patent claim {number} extraction mismatch")
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or protocol.get("formal_paper_metric") is not False
        or evaluation.get("valid") is not True
        or evaluation.get("formal_paper_metric") is not False
        or conditions.get("valid") is not True
        or conditions.get("formal_paper_metric") is not False
        or conditions.get("test_source_image_count") != 570
        or conditions.get("state") != "complete"
    ):
        errors.append("v6 protocol or evaluation contract mismatch")
    architecture = protocol.get("architecture", {})
    if (
        architecture.get("image_backbone_input_channels") != 3
        or architecture.get("independent_adapters", {}).get("count") != 3
        or architecture.get("film", {}).get("gamma_initial") != 1.0
        or architecture.get("film", {}).get("beta_initial") != 0.1
        or architecture.get("cosine_modality_weighting") is not True
        or architecture.get("deep_interaction", {}).get("attention_heads")
        != 8
        or architecture.get("dynamic_multiscale_fusion", {}).get(
            "initial_prior"
        )
        != [0.3, 0.5, 0.2]
    ):
        errors.append("v6 patent-stack architecture contract mismatch")

    model_wrapper = YOLO(str(checkpoint))
    model = model_wrapper.model.eval()
    if not isinstance(model, PatentAlignedReliableJointSegmentationModel):
        errors.append("checkpoint does not persist the v6 model class")
    device = next(model.parameters()).device
    first_layer_channels: list[int] = []
    head_inputs: list[list[Any]] = []

    def capture_backbone_input(_module: Any, inputs: Any) -> None:
        first_layer_channels.append(int(inputs[0].shape[1]))

    def capture_head_input(_module: Any, inputs: Any) -> None:
        head_inputs.append(
            [
                feature.detach().float().cpu().clone()
                for feature in inputs[0]
            ]
        )

    backbone_handle = model.model[0].conv.register_forward_pre_hook(
        capture_backbone_input
    )
    head_handle = model.model[-1].register_forward_pre_hook(capture_head_input)
    rgb_uint8 = (
        torch.arange(3 * 64 * 64, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(1, 3, 64, 64)
    )
    zero_grid = torch.zeros(1, 1, 4, 4)
    one_grid = torch.ones(1, 1, 4, 4)

    def run_forward(grid: Any, availability: float) -> dict[str, Any]:
        packed = pack_reliable_sensor_channels(
            rgb_uint8,
            grid,
            torch.tensor([[availability]], dtype=torch.float32),
        )
        with torch.inference_mode():
            _ = model(packed.float().div(255.0).to(device))
        return _auxiliary_copy(model)

    observed_zero_aux = run_forward(zero_grid, 1.0)
    observed_zero_head = head_inputs[-1]
    observed_one_aux = run_forward(one_grid, 1.0)
    observed_one_head = head_inputs[-1]
    _ = run_forward(zero_grid, 0.0)
    fallback_zero_head = head_inputs[-1]
    fallback_one_aux = run_forward(one_grid, 0.0)
    fallback_one_head = head_inputs[-1]
    backbone_handle.remove()
    head_handle.remove()

    observed_max, observed_mean = _feature_differences(
        observed_zero_head, observed_one_head
    )
    fallback_max, fallback_mean = _feature_differences(
        fallback_zero_head, fallback_one_head
    )
    expected_zero = torch.zeros_like(
        observed_zero_aux["observed_sensor_grid"]
    )
    expected_one = torch.ones_like(
        observed_one_aux["observed_sensor_grid"]
    )
    if first_layer_channels != [3, 3, 3, 3]:
        errors.append("five-channel transport reached the RGB backbone")
    if not torch.equal(
        observed_zero_aux["observed_sensor_grid"], expected_zero
    ) or not torch.equal(
        observed_one_aux["observed_sensor_grid"], expected_one
    ):
        errors.append("external 4x4 sensor transport was not preserved")
    if not all(value > 1e-7 for value in observed_max):
        errors.append("observed sensor did not causally alter every head scale")
    if any(value != 0.0 for value in fallback_max):
        errors.append("missing-sensor fallback depends on packed observations")
    if (
        not bool((observed_one_aux["sensor_available"] == 1).all())
        or not bool((fallback_one_aux["sensor_available"] == 0).all())
        or not bool((fallback_one_aux["sensor_reliability"] == 0).all())
    ):
        errors.append("availability or reliability fallback contract failed")

    required_shapes = {
        "patent_modality_weights": [1, 3],
        "patent_scale_weights": [1, 3],
        "patent_interaction_weights": [1, 8, 3, 3],
        "patent_final_fusion_feature": [1, 512],
        "patent_sensor_grid_16x16": [1, 1, 16, 16],
    }
    auxiliary_shapes = {
        name: list(observed_one_aux[name].shape) for name in required_shapes
    }
    if auxiliary_shapes != required_shapes:
        errors.append("patent-stack auxiliary tensor shapes differ")
    for name in ("patent_modality_weights", "patent_scale_weights"):
        weights = observed_one_aux[name]
        if (
            not torch.isfinite(weights).all()
            or bool((weights < 0).any())
            or not torch.allclose(
                weights.sum(dim=1),
                torch.ones(weights.shape[0]),
                atol=1e-6,
                rtol=0.0,
            )
        ):
            errors.append(f"{name} is not a probability simplex")

    modulator_norms = [
        float(module.weight.detach().float().norm().cpu())
        for module in model.joint_fusion.patent_scale_modulators
    ]
    patent_parameter_norms = {
        name: float(parameter.detach().float().norm().cpu())
        for name, parameter in model.joint_fusion.named_parameters()
        if name.startswith("patent_")
    }
    if not all(value > 0.0 for value in modulator_norms):
        errors.append("zero-initialized Segment-head modulators stayed inactive")
    if not patent_parameter_norms or not all(
        torch.isfinite(torch.tensor(value)) for value in patent_parameter_norms.values()
    ):
        errors.append("patent-stack parameter norms are missing or non-finite")

    clean_metrics = conditions["conditions"]["clean_matched"]["metrics"]
    fallback_metrics = conditions["conditions"]["rgb_fallback"]["metrics"]
    shuffled_metrics = conditions["conditions"]["shuffled"]["metrics"]
    clean_minus_fallback = metric_deltas(clean_metrics, fallback_metrics)
    clean_minus_shuffled = metric_deltas(clean_metrics, shuffled_metrics)
    task_effect_max_abs = max(
        abs(clean_minus_fallback["metrics/mAP50-95(B)"]),
        abs(clean_minus_fallback["metrics/mAP50-95(M)"]),
        abs(clean_minus_shuffled["metrics/mAP50-95(B)"]),
        abs(clean_minus_shuffled["metrics/mAP50-95(M)"]),
    )
    task_effect_interpretation = (
        "negligible_at_0.01_percentage_point_resolution"
        if task_effect_max_abs < 1e-4
        else "measurable_descriptive_effect"
    )

    outcome = {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "formal_paper_metric": False,
        "evidence_role": "patent_stack_structural_mechanism_development",
        "patent_claims": {
            "extracted_count": len(claims),
            "numbers": sorted(claims),
            "all_required_anchors_present": not any(
                error.startswith("patent claim") for error in errors
            ),
        },
        "structural_audit": {
            "persisted_model_class": type(model).__name__,
            "packed_input_channels": 5,
            "rgb_backbone_input_channels": first_layer_channels,
            "transport_grid_preserved": "exact_for_0_and_1_endpoints",
            "patent_auxiliary_shapes": auxiliary_shapes,
            "observed_grid_head_feature_max_abs_difference": observed_max,
            "observed_grid_head_feature_mean_abs_difference": observed_mean,
            "missing_grid_head_feature_max_abs_difference": fallback_max,
            "missing_grid_head_feature_mean_abs_difference": fallback_mean,
            "observed_sensor_causal_path_verified": all(
                value > 1e-7 for value in observed_max
            ),
            "missing_sensor_fallback_invariant": all(
                value == 0.0 for value in fallback_max
            ),
            "zero_initialized_modulator_weight_l2_after_training": (
                modulator_norms
            ),
            "patent_parameter_l2": patent_parameter_norms,
        },
        "test": {
            "source_image_count": 570,
            "clean_metrics": clean_metrics,
            "rgb_fallback_metrics": fallback_metrics,
            "shuffled_metrics": shuffled_metrics,
            "clean_minus_rgb_fallback": clean_minus_fallback,
            "clean_minus_shuffled": clean_minus_shuffled,
            "maximum_absolute_box_or_mask_map50_95_effect": task_effect_max_abs,
            "effect_interpretation": task_effect_interpretation,
            "conclusion": (
                "structural_causal_path_verified_without_meaningful_"
                "aggregate_task_gain"
            ),
        },
        "claim_boundaries": [
            "The v6 run is a five-epoch, single-seed rule-sidecar development run.",
            "The full patent-style fusion stack is structurally active before the native Segment head.",
            "Aggregate box and mask task metrics are effectively sensor-invariant in this run.",
            "The input is RGB rather than RGB-D.",
            "The sensor observation is image-conditioned rather than a synchronized hardware measurement.",
            "The taxonomy contains 19 general-waste classes rather than medical-waste classes.",
            "The result is excluded from the preregistered formal Qwen three-seed table.",
        ],
        "bindings": {
            **{name: _binding(path) for name, path in paths.items()},
            "checkpoint": _binding(checkpoint),
            "sensor_observations": _binding(sensor_observations),
        },
        "errors": errors,
    }
    output = report_run / "OUTCOME.json"
    _atomic_json(output, outcome)
    markdown = [
        "# v6 专利融合栈结构审计",
        "",
        f"- 有效：`{outcome['valid']}`；正式论文指标：`false`。",
        f"- 持久化模型：`{type(model).__name__}`。",
        "- 原始专利权利要求：已从 DOCX 的 WordprocessingML 独立提取并校验 1-10 项。",
        "- RGB 主干输入：四次干预前向均为 3 通道；4×4 网格与 availability 仅经独立传输通道进入联合头。",
        "- 同一 RGB、观测网格 0→1 时，P3/P4/P5 头前最大绝对差异："
        + "、".join(f"{value:.6f}" for value in observed_max)
        + "。",
        "- availability=0 时，两种观测网格的 P3/P4/P5 头前差异均严格为 0。",
        "- 零初始化头前调制器训练后 L2："
        + "、".join(f"{value:.6f}" for value in modulator_norms)
        + "。",
        "- 干净匹配相对 RGB 回退的 box/mask mAP50-95 差异："
        f"{clean_minus_fallback['metrics/mAP50-95(B)'] * 100:.4f}/"
        f"{clean_minus_fallback['metrics/mAP50-95(M)'] * 100:.4f} 个百分点。",
        "",
        "结论：v6 已证明完整融合栈可因果改变原生 Segment 头输入，缺失传感器回退也满足不变性；"
        "但当前 rule sidecar 开发测试的汇总任务指标几乎不受传感器影响，因此只能作为结构和机制证据，"
        "不能宣称获得了可观的多模态性能增益，更不能替代 RGB-D 或同步实体传感器实验。",
        "",
    ]
    markdown_path = output.with_suffix(".md")
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(markdown), encoding="utf-8")
    temporary.replace(markdown_path)
    return outcome


def main() -> None:
    outcome = audit(parse_args().report_run)
    print(
        json.dumps(
            {
                "state": outcome["state"],
                "valid": outcome["valid"],
                "formal_paper_metric": outcome["formal_paper_metric"],
                "causal_path": outcome["structural_audit"][
                    "observed_sensor_causal_path_verified"
                ],
                "fallback_invariant": outcome["structural_audit"][
                    "missing_sensor_fallback_invariant"
                ],
                "task_effect": outcome["test"]["effect_interpretation"],
                "errors": outcome["errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not outcome["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
