"""Audit the formal joint-head run and write a paper-facing addendum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path
from prepare_joint_multimodal_seg import sha256_file


DEFAULT_RUN_NAME = "yolo11s-seg-joint-v1-seed42"
DEFAULT_REPORT_RUN = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
    / DEFAULT_RUN_NAME
)
DEFAULT_BASE_EVALUATION = (
    PROJECT_ROOT / "reports" / "qwen-material" / "yolo-seg" / "evaluation.json"
)
DEFAULT_PREPARATION = (
    PROJECT_ROOT
    / "data"
    / "joint-multimodal-seg"
    / "PREPARATION_REPORT.json"
)


METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formal joint YOLO outcome.")
    parser.add_argument("--report-run", type=Path, default=DEFAULT_REPORT_RUN)
    parser.add_argument(
        "--base-evaluation", type=Path, default=DEFAULT_BASE_EVALUATION
    )
    parser.add_argument("--preparation", type=Path, default=DEFAULT_PREPARATION)
    return parser.parse_args()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def compare_metrics(
    joint: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        if key not in joint or key not in base:
            raise KeyError(f"missing comparison metric: {key}")
        joint_value = float(joint[key])
        base_value = float(base[key])
        result[key] = {
            "joint": joint_value,
            "base": base_value,
            "joint_minus_base": joint_value - base_value,
            "joint_minus_base_percentage_points": (joint_value - base_value)
            * 100.0,
        }
    return result


def _read_training_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("training results.csv is empty")
    return rows


def _verify_bindings(protocol: Mapping[str, Any]) -> list[str]:
    errors = []
    for raw_path, expected in protocol.get("source_bindings", {}).items():
        path = Path(raw_path)
        if not path.is_file():
            errors.append(f"missing bound source: {path}")
            continue
        if path.stat().st_size != int(expected["size_bytes"]):
            errors.append(f"bound source size changed: {path}")
            continue
        if sha256_file(path) != str(expected["sha256"]):
            errors.append(f"bound source hash changed: {path}")
    return errors


def _model_structure(checkpoint: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model
    fusion = getattr(model, "joint_fusion", None)
    if fusion is None:
        raise ValueError("checkpoint has no joint_fusion module")
    head = model.model[-1]
    text_norms = [
        float(module.weight.detach().float().norm().cpu())
        for module in fusion.text_output
    ]
    sensor_norms = [
        float(module.weight.detach().float().norm().cpu())
        for module in fusion.sensor_output
    ]
    return {
        "model_class": type(model).__name__,
        "head_class": type(head).__name__,
        "head_sources": list(head.f),
        "head_strides": [float(value) for value in head.stride.cpu()],
        "joint_module_class": type(fusion).__name__,
        "text_prototype_shape": list(fusion.text_prototypes.shape),
        "text_output_l2": text_norms,
        "sensor_output_l2": sensor_norms,
        "text_branch_nonzero": all(value > 0 for value in text_norms),
        "sensor_branch_nonzero": all(value > 0 for value in sensor_norms),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "joint_parameter_count": sum(
            parameter.numel() for parameter in fusion.parameters()
        ),
    }


def _paper_markdown(outcome: Mapping[str, Any]) -> str:
    comparison = outcome["metric_comparison"]
    box = comparison["metrics/mAP50-95(B)"]
    mask = comparison["metrics/mAP50-95(M)"]
    sensor = outcome["sensor_diagnostics"]
    masking = outcome["same_checkpoint_branch_masking"]
    lines = [
        "# 联合检测/分割头增量结果",
        "",
        "## 方法边界",
        "",
        "本增量在独立 YOLO11s-seg 模型的 P3、P4、P5 特征进入原生 "
        "Segment 头之前注入类别文本语义和 4×4 软传感器条件。训练期的 "
        "4×4 目标由对象级 Qwen/rule sidecar 与标注几何构造，仅监督 "
        "RGB-to-soft-sensor 分支；测试前向只接收 RGB，不读取测试标注、"
        "目标框或目标网格。该实现不等同于 RGB-D 或同步硬件传感器系统。",
        "",
        "## 独立测试集结果",
        "",
        "| 模型 | Box mAP50-95 | Mask mAP50-95 |",
        "|---|---:|---:|",
        (
            f"| 原 RGB YOLO11s-seg | {box['base']:.4f} | "
            f"{mask['base']:.4f} |"
        ),
        (
            f"| 联合文本+预测软传感器头 | {box['joint']:.4f} | "
            f"{mask['joint']:.4f} |"
        ),
        (
            f"| 差值（百分点） | {box['joint_minus_base_percentage_points']:+.2f} | "
            f"{mask['joint_minus_base_percentage_points']:+.2f} |"
        ),
        "",
        "以上比较使用相同的 570 张 source-grouped 测试图和 825 个实例。"
        "联合模型从已经完成训练的 RGB 检查点初始化，并冻结原 YOLO 参数，"
        "因此这里应解释为 adapter 增量的性能保持/变化，而不是从随机初始化"
        "开始的独立多种子优越性证明。",
        "",
        "## 软传感器诊断",
        "",
        (
            f"在 570 张测试图的 9120 个网格单元上，RGB 预测软传感器目标的 "
            f"MAE 为 {sensor['mae']:.4f}，RMSE 为 {sensor['rmse']:.4f}，"
            f"预测值与目标的 Pearson 相关系数为 "
            f"{sensor['pearson_predicted_target']:.3f}。预测不确定性与绝对误差"
            f"的 Pearson 相关系数为 "
            f"{sensor['uncertainty_absolute_error_pearson']:.3f}。"
        ),
        "",
        "这些指标衡量的是对图像条件软目标的重建与误差排序能力，不代表对真实"
        "物理重量、温度、湿度、电导率或气味传感器的校准。",
        "",
        "## 同检查点遮蔽",
        "",
        "下表为同一最佳检查点在测试时将对应融合分支置零的诊断，不是独立重训练"
        "消融，因而仅用于确认已训练分支对输出的条件贡献。",
        "",
        "| 前向模式 | Box mAP50-95 | Mask mAP50-95 |",
        "|---|---:|---:|",
    ]
    for mode in ("full", "no_text", "no_sensor", "vision_only"):
        metrics = masking["modes"][mode]["metrics"]
        lines.append(
            f"| {mode} | {float(metrics['metrics/mAP50-95(B)']):.4f} | "
            f"{float(metrics['metrics/mAP50-95(M)']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 专利对齐结论",
            "",
            "本增量使融合后的多尺度特征实际进入同一个 YOLO 检测与实例分割头，"
            "补齐了此前“两阶段外部检测器+对象分类器”与专利 S3/S4 之间的主要"
            "结构差距。仍未补齐的核心条件包括 RGB-D 深度输入和同步硬件阵列采集。",
            "",
        ]
    )
    return "\n".join(lines)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    report_run = resolve_project_path(args.report_run)
    base_path = resolve_project_path(args.base_evaluation)
    preparation_path = resolve_project_path(args.preparation)
    protocol_path = report_run / "PROTOCOL.json"
    evaluation_path = report_run / "evaluation.json"
    diagnostics_path = report_run / "diagnostics" / "evaluation.json"
    required = [
        protocol_path,
        evaluation_path,
        diagnostics_path,
        base_path,
        preparation_path,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing joint audit inputs: {missing}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    checkpoint = Path(evaluation["checkpoint"]["path"])
    errors = _verify_bindings(protocol)
    warnings = list(protocol.get("claim_boundaries", []))
    if not checkpoint.is_file():
        errors.append(f"missing best checkpoint: {checkpoint}")
    elif sha256_file(checkpoint) != evaluation["checkpoint"]["sha256"]:
        errors.append("best checkpoint hash differs from evaluation binding")

    train_args = protocol["train_args"]
    results_path = (
        Path(train_args["project"]) / str(train_args["name"]) / "results.csv"
    )
    if not results_path.is_file():
        errors.append(f"missing training results: {results_path}")
        training_rows: list[dict[str, str]] = []
    else:
        training_rows = _read_training_rows(results_path)
    metric_comparison = compare_metrics(
        evaluation["metrics"], base["metrics"]
    )
    model_structure = _model_structure(checkpoint) if checkpoint.is_file() else {}
    if model_structure and not model_structure["text_branch_nonzero"]:
        errors.append("text branch output weights remain zero")
    if model_structure and not model_structure["sensor_branch_nonzero"]:
        errors.append("sensor branch output weights remain zero")
    sensor_diagnostics = diagnostics["sensor_prediction"]
    if sensor_diagnostics.get("target_grid_passed_to_model") is not False:
        errors.append("diagnostics do not prove target-free detector forward")
    masking = diagnostics.get("detector_branch_masking")
    if not isinstance(masking, Mapping):
        errors.append("missing same-checkpoint branch masking diagnostics")

    outcome = {
        "schema_version": 1,
        "protocol": "joint_multimodal_yolo_segment_formal_outcome",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "training": {
            "requested_epochs": int(train_args["epochs"]),
            "completed_epochs": len(training_rows),
            "early_stopped": len(training_rows) < int(train_args["epochs"]),
            "batch_size": int(train_args["batch"]),
            "workers": int(train_args["workers"]),
            "freeze_layers": int(train_args["freeze"]),
            "amp": bool(train_args["amp"]),
            "best_recorded_epoch": evaluation["training"][
                "best_recorded_epoch"
            ]["epoch"],
            "best_recorded_val_box_map50_95": evaluation["training"][
                "best_recorded_epoch"
            ]["metrics/mAP50-95(B)"],
            "best_recorded_val_mask_map50_95": evaluation["training"][
                "best_recorded_epoch"
            ]["metrics/mAP50-95(M)"],
            "selection_note": evaluation["training"]["selection_note"],
        },
        "dataset": preparation["dataset"],
        "metric_comparison": metric_comparison,
        "sensor_diagnostics": sensor_diagnostics,
        "same_checkpoint_branch_masking": masking,
        "model_structure": model_structure,
        "claim_status": {
            "joint_fused_detect_segment_head": not errors,
            "rgb_only_metric_forward": (
                sensor_diagnostics.get("target_grid_passed_to_model") is False
                and sensor_diagnostics.get("external_sensor_passed_to_model")
                is False
            ),
            "qwen_soft_sensor_distillation": True,
            "hardware_sensor_measurement": False,
            "rgb_d_input": False,
            "full_patent_reproduction": False,
        },
        "bindings": {
            path.resolve().as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in required
        },
    }
    outcome_path = report_run / "OUTCOME.json"
    _atomic_write_text(
        outcome_path,
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
    )
    markdown_path = report_run / "PAPER_ADDENDUM.md"
    if not errors:
        _atomic_write_text(markdown_path, _paper_markdown(outcome))
    return outcome


def main() -> None:
    print(json.dumps(audit(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
