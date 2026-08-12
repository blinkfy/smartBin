"""Build audited Chinese manuscript result sections without a title or conclusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
ABLATION_ROOT = (
    REPORTS_ROOT
    / "confirmatory-ablations"
    / "v3-no-reliability-workers0"
)
DEFAULT_OUTPUT = REPORTS_ROOT / "MANUSCRIPT_RESULTS_SECTIONS.md"
SOURCES = {
    "v3_seed_outcome": (
        REPORTS_ROOT
        / "full_cross_modal_quality"
        / "seed-replication"
        / "OUTCOME.json"
    ),
    "reliability_ablation_outcome": ABLATION_ROOT / "OUTCOME.json",
    "main_classification": REPORTS_ROOT / "paper-tables" / "main_classification.csv",
    "formal_ablation_statistics": (
        REPORTS_ROOT / "paper-tables" / "formal_ablation_statistics.csv"
    ),
    "sensor_source_retraining": (
        REPORTS_ROOT / "paper-tables" / "sensor_source_retraining.csv"
    ),
    "yolo_segmentation": (
        REPORTS_ROOT / "paper-tables" / "yolo_segmentation.csv"
    ),
    "scene_pipeline": REPORTS_ROOT / "paper-tables" / "scene_pipeline.csv",
    "joint_multiseed": (
        REPORTS_ROOT
        / "joint-multimodal-seg"
        / "MULTISEED_SUMMARY.json"
    ),
    "joint_retrained_ablations": (
        REPORTS_ROOT
        / "joint-multimodal-seg"
        / "retrained-ablations"
        / "OUTCOME.json"
    ),
    "joint_deployment_profile": (
        REPORTS_ROOT
        / "joint-multimodal-seg"
        / "DEPLOYMENT_PROFILE.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build audited manuscript result sections"
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def build_results_sections(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    for path in SOURCES.values():
        if not path.is_file():
            raise FileNotFoundError(f"results source is missing: {path}")
    seed_outcome = _read_json(SOURCES["v3_seed_outcome"])
    ablation = _read_json(SOURCES["reliability_ablation_outcome"])
    joint = _read_json(SOURCES["joint_multiseed"])
    joint_ablations = _read_json(SOURCES["joint_retrained_ablations"])
    joint_deployment = _read_json(SOURCES["joint_deployment_profile"])
    if (
        seed_outcome.get("valid") is not True
        or ablation.get("valid") is not True
        or joint.get("valid") is not True
        or joint_ablations.get("valid") is not True
        or seed_outcome.get("training", {}).get("seeds") != [42, 43, 44]
        or ablation.get("training", {}).get("seeds") != [42, 43, 44]
        or joint.get("seeds") != [42, 43, 44]
        or joint_ablations.get("formal_run_count") != 6
        or joint_deployment.get("valid") is not True
    ):
        raise ValueError("confirmatory result sources are not valid")
    main_rows = _read_csv(SOURCES["main_classification"])
    v3 = next(
        row for row in main_rows if row["model_id"] == "full_cross_modal_quality"
    )
    formal = _read_csv(SOURCES["formal_ablation_statistics"])
    sensor_rows = _read_csv(SOURCES["sensor_source_retraining"])
    yolo = _read_csv(SOURCES["yolo_segmentation"])[0]
    scene = _read_csv(SOURCES["scene_pipeline"])
    qwen = next(row for row in sensor_rows if row["training_sensor_source"] == "qwen")
    rule = next(
        row for row in sensor_rows if row["training_sensor_source"] == "rule-v1"
    )

    test_aggregate = seed_outcome["test_summary"]["aggregate"]
    matched = ablation["matched_test_comparison"]["aggregate"]
    robustness = ablation["robustness_comparison"]["rows"]
    min_holm = min(float(row["mcnemar_holm_p_value"]) for row in formal)
    comparison_rows = []
    for seed in (42, 43, 44):
        path = ABLATION_ROOT / f"seed{seed}" / "MATCHED_TEST_COMPARISON.json"
        comparison = _read_json(path)
        accuracy = comparison["paired_bootstrap"]["differences"]["accuracy"]
        macro_f1 = comparison["paired_bootstrap"]["differences"]["macro_f1"]
        comparison_rows.append(
            (
                seed,
                accuracy["point"],
                accuracy["lower"],
                accuracy["upper"],
                macro_f1["point"],
                macro_f1["lower"],
                macro_f1["upper"],
                comparison["mcnemar_exact"]["p_value"],
            )
        )
        SOURCES[f"matched_seed{seed}"] = path

    detector025 = next(
        row
        for row in scene
        if row["confidence_threshold"] == "0.25"
        and row["prediction_source"] == "detector"
    )
    detector080 = next(
        row
        for row in scene
        if row["confidence_threshold"] == "0.8"
        and row["prediction_source"] == "detector"
    )
    joint_box = joint["aggregate"]["metrics/mAP50-95(B)"]
    joint_mask = joint["aggregate"]["metrics/mAP50-95(M)"]
    text_only = joint_ablations["aggregate"]["text_only"]
    sensor_only = joint_ablations["aggregate"]["sensor_only"]
    joint_base = joint_ablations["reference"]["vision_only"]["metrics"]
    joint_full = joint_ablations["reference"]["full"]["metrics"]
    joint_speed = joint_deployment["speed_ms_per_image"]
    joint_parameters = joint_deployment["checkpoint_parameters_unfused"]
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "audited_manuscript_results_sections",
        "title_abstract_conclusion_included": False,
        "source_bindings": {
            **{name: _binding(path) for name, path in SOURCES.items()},
            "builder_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_write(
        output_path.with_suffix(".json"),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# 论文结果章节草稿",
        "",
        "本稿仅生成由审计产物支持的结果段，不包含标题、摘要、贡献排序或结论。",
        "百分比与置信区间均由绑定 JSON/CSV 自动读取，不能替代外部盲测。",
        "",
        "## 8 结果",
        "",
        "### 8.1 对象级分类与训练种子稳定性",
        "",
        (
            "预注册主模型 v3 在固定内部测试集的 seed42 运行上取得 "
            f"{_pct(float(v3['accuracy']))} Accuracy、"
            f"{_pct(float(v3['macro_f1']))} Macro F1 和 "
            f"{_pct(float(v3['ece']))} ECE。按源图聚类得到的 Accuracy "
            f"95% 描述性区间为 [{_pct(float(v3['accuracy_ci_lower']))}, "
            f"{_pct(float(v3['accuracy_ci_upper']))}]，Macro F1 区间为 "
            f"[{_pct(float(v3['macro_f1_ci_lower']))}, "
            f"{_pct(float(v3['macro_f1_ci_upper']))}]。"
        ),
        "",
        (
            "在 seed42、43 和 44 三个独立训练种子上，测试 Accuracy 为 "
            f"{_pct(test_aggregate['accuracy']['mean'])} ± "
            f"{100.0 * test_aggregate['accuracy']['sample_std']:.2f} 个百分点，"
            "Macro F1 为 "
            f"{_pct(test_aggregate['macro_f1']['mean'])} ± "
            f"{100.0 * test_aggregate['macro_f1']['sample_std']:.2f} 个百分点"
            "（均值 ± 样本标准差）。seed42 保持主模型，未按测试结果更换种子。"
        ),
        "",
        "### 8.2 可靠性子系统的匹配种子消融",
        "",
        (
            "为隔离 v3 可靠性机制，独立重训了移除潜表示预测、可靠性门控、"
            "网格质量、跨模态网格质量及相关辅助损失的三种子变体；软件退化训练、"
            "三模态编码、注意力融合、数据、优化器和 100 轮日程保持一致。"
        ),
        "",
        (
            "完整 v3 相对该消融的平均测试差异为 Accuracy "
            f"{_pp(matched['accuracy']['mean'])} 个百分点（种子间样本标准差 "
            f"{100.0 * matched['accuracy']['sample_std']:.2f}），Macro F1 "
            f"{_pp(matched['macro_f1']['mean'])} 个百分点（样本标准差 "
            f"{100.0 * matched['macro_f1']['sample_std']:.2f}）。"
            f"Accuracy 在 {matched['accuracy']['positive_seed_count']}/3 个种子"
            "上有利于完整 v3，Macro F1 在 "
            f"{matched['macro_f1']['positive_seed_count']}/3 个种子上有利于"
            "完整 v3。"
        ),
        "",
        "| 训练种子 | Accuracy 差异 (pp) [95% CI] | Macro F1 差异 (pp) [95% CI] | McNemar p |",
        "|---:|---:|---:|---:|",
        *[
            (
                f"| {seed} | {_pp(acc)} [{_pp(acc_lo)}, {_pp(acc_hi)}] | "
                f"{_pp(f1)} [{_pp(f1_lo)}, {_pp(f1_hi)}] | {mcnemar:.4f} |"
            )
            for seed, acc, acc_lo, acc_hi, f1, f1_lo, f1_hi, mcnemar
            in comparison_rows
        ],
        "",
        (
            "表中差异均为完整 v3 减去无可靠性子系统变体；区间按相同 "
            "source_image_id 簇进行 5,000 次配对 bootstrap。三种子均值用于"
            "描述训练种子变化，不进行事后最佳种子选择。"
        ),
        "",
        "### 8.3 软件传感器退化鲁棒性",
        "",
        (
            "鲁棒性比较固定 seed42 训练模型，并在 5 个匹配软件扰动种子上重复。"
            "这些扰动种子只表示退化实现的重复，不是独立训练种子。"
        ),
        "",
        "| 场景 | Accuracy 效应 (pp) | Macro F1 效应 (pp) | ECE 改善 (pp) |",
        "|---|---:|---:|---:|",
        *[
            (
                f"| {row['scenario']} | "
                f"{row['full_minus_ablation_accuracy_pp']:+.2f} | "
                f"{row['full_minus_ablation_macro_f1_pp']:+.2f} | "
                f"{row['ablation_minus_full_ece_pp']:+.2f} |"
            )
            for row in robustness
        ],
        "",
        (
            "Accuracy 与 Macro F1 的正值表示完整 v3 更优；ECE 改善的正值"
            "表示完整 v3 的 ECE 更低。该表是 seed42 的支持性描述证据。"
        ),
        "",
        "### 8.4 历史结构消融与来源分析",
        "",
        (
            "较早谱系的八项正式消融均为单训练种子。其配对 McNemar 检验经 "
            f"Holm 校正后的最小 p 值为 {min_holm:.4f}，因此不能把该套结果"
            "解释为已经覆盖训练种子方差的显著组件优势。新三种子可靠性子系统"
            "实验与该历史表分开报告。"
        ),
        "",
        (
            "独立来源重训中，Qwen sidecar 模型的 Accuracy/Macro F1 为 "
            f"{_pct(float(qwen['accuracy']))}/{_pct(float(qwen['macro_f1']))}，"
            "rule-v1 模型为 "
            f"{_pct(float(rule['accuracy']))}/{_pct(float(rule['macro_f1']))}。"
            "该来源比较仍只有一个训练种子，且两种 sidecar 均由图像条件生成，"
            "不能解释为真实硬件传感器增益。"
        ),
        "",
        "### 8.5 目标域分割与联合多模态头",
        "",
        (
            f"目标域 YOLO11s-seg 在 570 张测试源图、825 个对象上取得 box "
            f"mAP50/mAP50-95 {_pct(float(yolo['box_map50']))}/"
            f"{_pct(float(yolo['box_map50_95']))}，mask mAP50/mAP50-95 "
            f"{_pct(float(yolo['mask_map50']))}/"
            f"{_pct(float(yolo['mask_map50_95']))}。"
        ),
        "",
        (
            "在相同 source-grouped 测试集上，文本与 RGB 预测软传感器联合头的"
            "三种子 box mAP50-95 为 "
            f"{_pct(joint_box['mean'])} ± "
            f"{100.0 * joint_box['sample_std']:.2f} 个百分点，mask mAP50-95 为 "
            f"{_pct(joint_mask['mean'])} ± "
            f"{100.0 * joint_mask['sample_std']:.2f} 个百分点（均值 ± 样本"
            "标准差）。相对独立 RGB 基线的均值变化分别为 "
            f"{joint_box['mean_minus_rgb_baseline_percentage_points']:+.2f} 和 "
            f"{joint_mask['mean_minus_rgb_baseline_percentage_points']:+.2f} "
            "个百分点。联合模型从 RGB 检查点初始化并冻结原 YOLO 参数，因此"
            "该结果用于验证多模态 adapter 的可重复性能保持与变化。"
        ),
        "",
        (
            f"联合 adapter 增加 {joint_parameters['joint_added']:,} 个参数，"
            f"约为 RGB 模型参数量的 {joint_parameters['joint_added_percent_of_rgb']:.2f}%。"
            "相同环境固定测试验证器记录的联合头三种子推理时间为 "
            f"{joint_speed['joint_three_seed_summary']['inference']['mean']:.3f} ± "
            f"{joint_speed['joint_three_seed_summary']['inference']['sample_std']:.3f} ms/图，"
            f"相对单次 RGB 验证值增加 {joint_speed['joint_inference_mean_overhead_percent']:.2f}%。"
            "这些数值是批量验证摊销速度，不是受控单样本延迟测试，也不用于复现"
            "专利中的 15.2 ms。"
        ),
        "",
        "### 8.6 联合头分支的独立重训消融",
        "",
        (
            "为替代同检查点分支屏蔽的弱证据，按相同种子 42/43/44 独立重训了"
            "仅文本和仅 RGB 预测软传感器两个变体；所有六个正式 run 均按冻结"
            "协议纳入统计。"
        ),
        "",
        "| 模型 | Box mAP50-95 | Mask mAP50-95 |",
        "|---|---:|---:|",
        (
            "| 仅 RGB | "
            f"{_pct(joint_base['metrics/mAP50-95(B)'])} | "
            f"{_pct(joint_base['metrics/mAP50-95(M)'])} |"
        ),
        (
            "| 仅文本 adapter | "
            f"{_pct(text_only['metrics']['metrics/mAP50-95(B)']['mean'])} ± "
            f"{100.0 * text_only['metrics']['metrics/mAP50-95(B)']['sample_std']:.2f} pp | "
            f"{_pct(text_only['metrics']['metrics/mAP50-95(M)']['mean'])} ± "
            f"{100.0 * text_only['metrics']['metrics/mAP50-95(M)']['sample_std']:.2f} pp |"
        ),
        (
            "| 仅预测软传感器 adapter | "
            f"{_pct(sensor_only['metrics']['metrics/mAP50-95(B)']['mean'])} ± "
            f"{100.0 * sensor_only['metrics']['metrics/mAP50-95(B)']['sample_std']:.2f} pp | "
            f"{_pct(sensor_only['metrics']['metrics/mAP50-95(M)']['mean'])} ± "
            f"{100.0 * sensor_only['metrics']['metrics/mAP50-95(M)']['sample_std']:.2f} pp |"
        ),
        (
            "| 完整联合头 | "
            f"{_pct(joint_full['metrics/mAP50-95(B)'])} | "
            f"{_pct(joint_full['metrics/mAP50-95(M)'])} |"
        ),
        "",
        (
            "完整联合头相对仅文本变体的同种子平均差异为 box "
            f"{text_only['paired_full_minus_mode']['metrics/mAP50-95(B)']['mean_percentage_points']:+.2f}、"
            "mask "
            f"{text_only['paired_full_minus_mode']['metrics/mAP50-95(M)']['mean_percentage_points']:+.2f} "
            "个百分点；相对仅预测软传感器变体的差异为 box "
            f"{sensor_only['paired_full_minus_mode']['metrics/mAP50-95(B)']['mean_percentage_points']:+.2f}、"
            "mask "
            f"{sensor_only['paired_full_minus_mode']['metrics/mAP50-95(M)']['mean_percentage_points']:+.2f} "
            "个百分点。三种子均值用于描述可重复性，不替代外部验证或显著性证明。"
        ),
        "",
        "### 8.7 两阶段整图系统",
        "",
        (
            "整图链在 confidence=0.25 时的定位 Precision/Recall 为 "
            f"{_pct(float(detector025['localization_precision']))}/"
            f"{_pct(float(detector025['localization_recall']))}，端到端 "
            f"Macro F1 为 {_pct(float(detector025['macro_f1']))}；在 "
            "confidence=0.8 时相应数值为 "
            f"{_pct(float(detector080['localization_precision']))}/"
            f"{_pct(float(detector080['localization_recall']))} 和 "
            f"{_pct(float(detector080['macro_f1']))}。该结果属于 YOLO 定位后"
            "再分类的两阶段系统，与新增的联合 Segment-head 支线分别报告。"
        ),
        "",
        "### 8.8 结果解释边界",
        "",
        (
            "固定测试集已被多次检查，结果属于内部基准；Qwen/rule-v1 是图像条件"
            "软传感器而非同步硬件；当前软传感器联合头已经把融合特征送入原生 "
            "Segment head，但系统仍没有 RGB-D、同步硬件阵列或外部盲测；相对"
            "面积和 bbox 重叠是诊断代理，不是人工小目标或遮挡标签。"
        ),
        "",
    ]
    _atomic_write(output_path, "\n".join(lines))
    return report


def main() -> None:
    output = parse_args().output
    report = build_results_sections(output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "source_binding_count": len(report["source_bindings"]),
                "output": _resolve(output).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
