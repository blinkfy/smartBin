"""Build manuscript-ready method and result text for the v6 patent stack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from train_patent_aligned_external_sensor_joint_seg import DEFAULT_NAME


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
RUN_ROOT = (
    REPORTS
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
)
DEFAULT_OUTPUT = REPORTS / "PATENT_STACK_MANUSCRIPT_ADDENDUM.md"
SOURCES = {
    "requirement_matrix": REPORTS / "PATENT_REQUIREMENT_MATRIX.json",
    "v6_protocol": RUN_ROOT / "PROTOCOL.json",
    "v6_evaluation": RUN_ROOT / "evaluation.json",
    "v6_conditions": RUN_ROOT / "RELIABILITY_CONDITIONS.json",
    "v6_outcome": RUN_ROOT / "OUTCOME.json",
    "v5_reliability_audit": REPORTS / "SPATIAL_RELIABILITY_AUDIT.json",
    "worker_equivalence": (
        REPORTS
        / "joint-multimodal-seg-external"
        / "WORKER_EQUIVALENCE.json"
    ),
    "complexity_benchmark": REPORTS / "PATENT_STACK_COMPLEXITY.json",
    "intervention_audit": (
        REPORTS / "PATENT_STACK_INTERVENTION_AUDIT.json"
    ),
    "architecture_figure": (
        REPORTS
        / "joint-multimodal-seg-patent"
        / "patent_stack_architecture.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v6 patent-stack manuscript addendum"
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


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _pp(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:+.{digits}f}"


def build_addendum(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    missing = [name for name, path in SOURCES.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "patent manuscript inputs missing: " + ", ".join(missing)
        )
    matrix = _read_json(SOURCES["requirement_matrix"])
    protocol = _read_json(SOURCES["v6_protocol"])
    evaluation = _read_json(SOURCES["v6_evaluation"])
    conditions = _read_json(SOURCES["v6_conditions"])
    outcome = _read_json(SOURCES["v6_outcome"])
    v5 = _read_json(SOURCES["v5_reliability_audit"])
    workers = _read_json(SOURCES["worker_equivalence"])
    complexity = _read_json(SOURCES["complexity_benchmark"])
    intervention = _read_json(SOURCES["intervention_audit"])
    architecture_figure = _read_json(SOURCES["architecture_figure"])
    if (
        matrix.get("valid") is not True
        or protocol.get("formal_paper_metric") is not False
        or evaluation.get("valid") is not True
        or conditions.get("valid") is not True
        or outcome.get("valid") is not True
        or v5.get("valid") is not True
        or workers.get("valid") is not True
        or workers.get("equivalence", {}).get("model_state", {}).get(
            "exact"
        )
        is not True
        or workers.get("recommendation", {}).get(
            "formal_training_workers"
        )
        != 4
        or complexity.get("valid") is not True
        or complexity.get("formal_paper_metric") is not False
        or intervention.get("valid") is not True
        or intervention.get("formal_paper_metric") is not False
        or architecture_figure.get("valid") is not True
    ):
        raise ValueError("patent manuscript sources are not valid")

    v6_test = outcome["test"]
    v6_clean = v6_test["clean_metrics"]
    v6_fallback = v6_test["rgb_fallback_metrics"]
    v6_shuffled = v6_test["shuffled_metrics"]
    v5_test = v5["primary_v5_test"]
    v5_clean = v5_test["clean_metrics"]
    v5_fallback = v5_test["rgb_fallback_metrics"]
    v5_shuffled = v5_test["shuffled_metrics"]
    v5_parameters = _read_json(
        REPORTS
        / "joint-multimodal-seg-spatial"
        / "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
        / "evaluation.json"
    )["parameters"]
    v6_parameters = evaluation["parameters"]
    total_parameter_increase = (
        v6_parameters["total"] - v5_parameters["total"]
    )
    total_parameter_increase_fraction = (
        total_parameter_increase / v5_parameters["total"]
    )
    fusion_parameter_increase = (
        v6_parameters["joint_fusion"] - v5_parameters["joint_fusion"]
    )
    v6_minus_v5 = {
        key: float(v6_clean[key]) - float(v5_clean[key])
        for key in (
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(M)",
        )
    }
    coverage = matrix["summary"]["status_counts"]

    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "scope": "patent_stack_manuscript_method_result_limitations_addendum",
        "section_roles": [
            "method",
            "development_results",
            "discussion",
            "limitations",
        ],
        "v6_result": {
            "training_epochs": evaluation["training"]["completed_epochs"],
            "seed_count": 1,
            "sensor_source": "scene-rule-v1",
            "source_image_count": v6_test["source_image_count"],
            "clean_metrics": v6_clean,
            "clean_minus_rgb_fallback": v6_test[
                "clean_minus_rgb_fallback"
            ],
            "clean_minus_shuffled": v6_test["clean_minus_shuffled"],
            "effect_interpretation": v6_test["effect_interpretation"],
            "formal_performance_evidence": False,
        },
        "comparison": {
            "v5_role": "selected_reliability_development_model",
            "v6_role": "selected_patent_structure_model",
            "v6_minus_v5_clean": v6_minus_v5,
            "v6_promoted_for_task_performance": False,
        },
        "complexity_from_evaluation": {
            "v5_parameters": v5_parameters,
            "v6_parameters": v6_parameters,
            "total_parameter_increase": total_parameter_increase,
            "total_parameter_increase_fraction": (
                total_parameter_increase_fraction
            ),
            "joint_fusion_parameter_increase": fusion_parameter_increase,
        },
        "equivalent_input_pipeline_acceleration": {
            "workers_from": workers["authorized_change"]["workers"]["from"],
            "workers_to": workers["authorized_change"]["workers"]["to"],
            "state_tensor_count": workers["equivalence"]["model_state"][
                "tensor_count"
            ],
            "state_tensor_differences": workers["equivalence"][
                "model_state"
            ]["different_tensor_count"],
            "speedup": workers["timing"]["speedup"],
            "time_reduction_fraction": workers["timing"][
                "time_reduction_fraction"
            ],
        },
        "descriptive_raw_forward_complexity": {
            "configuration": complexity["configuration"],
            "models": complexity["models"],
            "v6_over_v5": complexity["v6_over_v5"],
            "formal_performance_evidence": False,
        },
        "same_checkpoint_conditioning_intervention": {
            **intervention["contribution_decomposition"],
            "retrained_ablation": False,
            "formal_performance_evidence": False,
        },
        "architecture_figure": architecture_figure["artifacts"],
        "patent_coverage": matrix["summary"],
        "source_bindings": {
            **{name: _binding(path) for name, path in SOURCES.items()},
            "v5_evaluation": _binding(
                REPORTS
                / "joint-multimodal-seg-spatial"
                / "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
                / "evaluation.json"
            ),
            "builder_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_text(
        output.with_suffix(".json"),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )

    head_differences = outcome["structural_audit"][
        "observed_grid_head_feature_max_abs_difference"
    ]
    lines = [
        "# 专利融合栈论文增量",
        "",
        "本增量只包含由现有审计产物支持的方法、开发结果与边界文本。"
        "它不覆盖已绑定的主结果稿，也不把单种子rule sidecar开发实验升级为正式结论。",
        "",
        "## 方法增量：专利对齐的联合检测分割头",
        "",
        "在可靠性联合头v5的基础上，构建专利对齐融合栈v6。YOLO11s-seg主干从"
        "P3、P4和P5提取三尺度空间特征，各尺度经全局池化和独立线性映射得到"
        "512维视觉表示，并以0.3、0.5和0.2作为可学习尺度权重的初始先验。"
        "外部4x4观测先经跨模态可靠性门控与RGB预测网格混合，再双线性上采样至"
        "16x16，通过1x1卷积、LeakyReLU和全局池化得到512维传感器表示。",
        "",
        "文本分支使用19个冻结的CLIP ViT-B/16类别原型。模型以视觉和传感器表示"
        "共同查询全部候选原型，不读取真实类别标签。视觉、文本和传感器分别经过"
        "独立的512维Adapter、LeakyReLU及dropout=0.3；随后采用初始"
        "gamma=1.0、beta=0.1的FiLM参数调制。三模态两两余弦相似度经softmax"
        "得到模态权重，特征下投影至128维后通过8头、3 token自注意力交互，"
        "再上投影回512维并执行残差连接。最终融合向量动态调整P3/P4/P5尺度权重，"
        "并生成三个尺度的FiLM参数；调制后的空间特征直接进入原生Segment头，"
        "共同预测边界框、类别与实例掩膜。",
        "",
        "该实现保留了专利的核心Adapter、FiLM、余弦权重、降维交互、残差与"
        "多尺度路径，但不是逐字公式复刻：v6采用三个模态token的联合自注意力，"
        "而非显式生成六个有向模态对后拼接；尺度使用原生P3/P4/P5，未强制重采样为"
        "8x8、16x16和32x32。对象级分类器使用CE+Focal，v6联合头则使用"
        "Ultralytics原生检测/分割损失和传感器辅助损失，两者不能写成同一损失实验。",
        "",
        "对应结构图与带边界图注见"
        "`joint-multimodal-seg-patent/patent_stack_architecture.pdf`和"
        "`PATENT_STACK_FIGURE_CAPTION.md`。",
        "",
        "## 开发结果增量",
        "",
        "v6从已选择的v5检查点初始化，在scene-rule-v1整图sidecar上训练5轮，"
        "batch size为32、DataLoader workers为4，并保持单个训练种子42。"
        f"固定570张测试源图上，干净匹配条件的box与mask mAP50-95分别为"
        f"{_pct(v6_clean['metrics/mAP50-95(B)'])}和"
        f"{_pct(v6_clean['metrics/mAP50-95(M)'])}。RGB回退条件分别为"
        f"{_pct(v6_fallback['metrics/mAP50-95(B)'])}和"
        f"{_pct(v6_fallback['metrics/mAP50-95(M)'])}；乱序观测条件分别为"
        f"{_pct(v6_shuffled['metrics/mAP50-95(B)'])}和"
        f"{_pct(v6_shuffled['metrics/mAP50-95(M)'])}。",
        "",
        "干净匹配相对RGB回退的box/mask差异仅为"
        f"{_pp(v6_test['clean_minus_rgb_fallback']['metrics/mAP50-95(B)'], 4)}/"
        f"{_pp(v6_test['clean_minus_rgb_fallback']['metrics/mAP50-95(M)'], 4)}"
        "个百分点；相对乱序观测的差异为"
        f"{_pp(v6_test['clean_minus_shuffled']['metrics/mAP50-95(B)'], 4)}/"
        f"{_pp(v6_test['clean_minus_shuffled']['metrics/mAP50-95(M)'], 4)}"
        "个百分点。该量级低于0.01个百分点，因此当前汇总任务指标可视为"
        "对sidecar近似不敏感。",
        "",
        "结构干预给出不同层面的证据：保持RGB不变并把有效4x4观测从全0改为全1时，"
        "进入Segment头的P3/P4/P5最大绝对特征差异分别为"
        + "、".join(f"{value:.6f}" for value in head_differences)
        + "；将availability置0后，三层差异均严格为0。"
        "三个零初始化头前调制器训练后均获得非零权重。由此可确认传感器路径已接入"
        "原生头且缺失回退符合不变性，但该结构成立并不等于已经获得任务性能增益。",
        "",
        "进一步在同一v6检查点内，仅将三个最终尺度调制器的weight与bias归零，"
        "共改变6个预注册状态张量，其余模型状态和检查点文件保持不变。完整调制相对"
        "归零条件的box/mask mAP50-95提高"
        f"{100.0 * intervention['contribution_decomposition']['full_stack_conditioning_effect']['box_map50_95']:+.3f}/"
        f"{100.0 * intervention['contribution_decomposition']['full_stack_conditioning_effect']['mask_map50_95']:+.3f}"
        "个百分点。这说明已训练融合栈的整体头前调制具有可测任务作用；结合"
        "clean、fallback和shuffled之间近零差异，现有证据不能把这项整体作用"
        "归因于匹配sidecar或传感器分支。该干预不是独立重训练消融。",
        "",
        "| 开发模型 | 角色 | Box mAP50-95 | Mask mAP50-95 | 干净相对回退 |",
        "|---|---|---:|---:|---:|",
        "| v5空间可靠性头 | 可靠性机制开发模型 | "
        f"{_pct(v5_clean['metrics/mAP50-95(B)'])} | "
        f"{_pct(v5_clean['metrics/mAP50-95(M)'])} | "
        f"{_pp(v5_test['clean_minus_rgb_fallback']['metrics/mAP50-95(B)'])}/"
        f"{_pp(v5_test['clean_minus_rgb_fallback']['metrics/mAP50-95(M)'])} pp |",
        "| v6专利融合栈 | 专利结构开发模型 | "
        f"{_pct(v6_clean['metrics/mAP50-95(B)'])} | "
        f"{_pct(v6_clean['metrics/mAP50-95(M)'])} | "
        f"{_pp(v6_test['clean_minus_rgb_fallback']['metrics/mAP50-95(B)'], 4)}/"
        f"{_pp(v6_test['clean_minus_rgb_fallback']['metrics/mAP50-95(M)'], 4)} pp |",
        "",
        "v6相对v5的干净box/mask mAP50-95变化为"
        f"{_pp(v6_minus_v5['metrics/mAP50-95(B)'])}/"
        f"{_pp(v6_minus_v5['metrics/mAP50-95(M)'])}个百分点。"
        "因此v5继续作为可靠性开发候选，v6仅作为更接近专利S3至S4路径的结构候选，"
        "不按任务性能晋升。",
        "",
        "## 参数与训练吞吐边界",
        "",
        f"按训练后评估对象统计，v5与v6总参数量分别为"
        f"{v5_parameters['total']:,}和{v6_parameters['total']:,}；"
        f"v6增加{total_parameter_increase:,}个参数，增幅"
        f"{100.0 * total_parameter_increase_fraction:.2f}%。"
        f"其中联合融合模块由{v5_parameters['joint_fusion']:,}增至"
        f"{v6_parameters['joint_fusion']:,}个参数。"
        "这些数字描述模型复杂度，不替代受控延迟或能耗实验。",
        "",
        "同一RTX 4090上采用batch=32、640x640与AMP FP16进行融合后原始前向"
        "基准，v5/v6分别为"
        f"{complexity['models']['v5_spatial_reliability']['timing']['median_ms_per_image']:.3f}/"
        f"{complexity['models']['v6_patent_stack']['timing']['median_ms_per_image']:.3f}"
        " ms/图；v6相对v5的GFLOPs增加"
        f"{100.0 * complexity['v6_over_v5']['gflops_fraction']:.2f}%，"
        "中位batch延迟增加"
        f"{100.0 * complexity['v6_over_v5']['median_batch_latency_fraction']:.2f}%。"
        "该计时不含DataLoader、主机到GPU传输、NMS、mask后处理和写盘，"
        "只用于同机结构开销对照。",
        "",
        "仅改变DataLoader workers从"
        f"{workers['authorized_change']['workers']['from']}到"
        f"{workers['authorized_change']['workers']['to']}的既有等效性实验中，"
        f"{workers['equivalence']['model_state']['tensor_count']}个模型状态张量"
        "逐元素完全一致，去除时间列后的训练CSV及测试指标一致，训练耗时由"
        f"{workers['timing']['baseline_training_seconds']:.2f} s降至"
        f"{workers['timing']['candidate_training_seconds']:.2f} s，速度提升"
        f"{workers['timing']['speedup']:.3f}倍。因而后续正式训练保持"
        "batch size=32及全部科学参数不变，仅采用workers=4；环境变化时重新基准。",
        "",
        "## 讨论与限制增量",
        "",
        f"专利要求矩阵拆分为{matrix['summary']['requirement_count']}项："
        f"已实现{coverage['implemented']}项、部分实现"
        f"{coverage['partially_implemented']}项、未实现"
        f"{coverage['not_implemented']}项，另有"
        f"{coverage['intentional_scope_change']}项研究范围有意变更。"
        "当前证据支持“实现了专利式融合结构并完成原生分割头接入”，"
        "不支持“完整复现专利系统”。",
        "",
        "首先，输入是RGB而非RGB-D；其次，rule与Qwen观测均由图像条件生成，"
        "不是同步实体阵列；再次，任务已调整为19类一般垃圾，不是医疗废物"
        "5大类/20细类。现有面积与框重叠分层只是小目标和遮挡代理。"
        "原始专利中的98.8%、93.3%、95.7%和15.2 ms没有可复核实验产物，"
        "不得作为本文结果。正式Qwen整图sidecar三种子协议仍待环境凭据执行，"
        "其完成前不得把v6的单种子rule结果写入正式主表。",
        "",
    ]
    _atomic_text(output, "\n".join(lines))
    return report


def main() -> None:
    result = build_addendum(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "section_roles": result["section_roles"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
