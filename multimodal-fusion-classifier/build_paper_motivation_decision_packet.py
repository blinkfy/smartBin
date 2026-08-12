"""Build a user-confirmable paper motivation and contribution decision packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORTS / "MOTIVATION_DECISION_PACKET.json"
SOURCES = {
    "v3_seed_outcome": (
        REPORTS
        / "full_cross_modal_quality"
        / "seed-replication"
        / "OUTCOME.json"
    ),
    "v3_reliability_ablation": (
        REPORTS
        / "confirmatory-ablations"
        / "v3-no-reliability-workers0"
        / "OUTCOME.json"
    ),
    "v5_reliability_audit": REPORTS / "SPATIAL_RELIABILITY_AUDIT.json",
    "v6_patent_outcome": (
        REPORTS
        / "joint-multimodal-seg-patent"
        / "yolo11s-seg-joint-patent-stack-rule-dev-seed42"
        / "OUTCOME.json"
    ),
    "v6_intervention_audit": (
        REPORTS / "PATENT_STACK_INTERVENTION_AUDIT.json"
    ),
    "patent_matrix": REPORTS / "PATENT_REQUIREMENT_MATRIX.json",
    "patent_addendum": REPORTS / "PATENT_STACK_MANUSCRIPT_ADDENDUM.json",
    "joint_multiseed": (
        REPORTS
        / "joint-multimodal-seg"
        / "MULTISEED_SUMMARY.json"
    ),
    "joint_retrained_ablations": (
        REPORTS
        / "joint-multimodal-seg"
        / "retrained-ablations"
        / "OUTCOME.json"
    ),
    "final_evidence_audit": REPORTS / "FINAL_EVIDENCE_AUDIT.json",
    "bibliography_audit": REPORTS / "BIBLIOGRAPHY_AUDIT.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paper motivation decision packet"
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


def _option_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "A",
            "recommended": True,
            "name": "低质量图像条件软传感器下的可靠多模态垃圾分类",
            "title_candidates": [
                "面向图像条件软传感器退化的可靠多模态垃圾分类方法",
                "融合跨模态一致性与不确定性的智能垃圾可靠分类方法",
            ],
            "controlling_motivation": (
                "图像条件soft sensor存在缺失、噪声、偏移或卡死时，"
                "需要在保留多模态互补性的同时抑制不可靠观测，并提供连续回退。"
            ),
            "contribution_promises": [
                "视觉-文本条件的传感器潜表示预测与缺失回退。",
                "结合availability、原始网格质量、跨模态差异和不确定性的可靠性门控。",
                "以多训练种子独立消融、同检查点干预和软件退化验证机制，而非只报告clean Accuracy。",
                "将可靠融合思想扩展至原生YOLO11s-seg联合头，并公开专利对齐边界。",
            ],
            "results_validation": [
                {
                    "promise": 1,
                    "evidence": "v3三种子主模型、潜表示与不确定性诊断",
                },
                {
                    "promise": 2,
                    "evidence": "v3三种子无可靠性子系统独立重训对照",
                },
                {
                    "promise": 3,
                    "evidence": "五种退化、多扰动种子、门控同检查点因果干预",
                },
                {
                    "promise": 4,
                    "evidence": "v5可靠性头、v6专利融合栈与头前调制干预",
                },
            ],
            "reviewer_value": (
                "主贡献由对象级三种子与独立消融支撑，联合头作为系统扩展，"
                "不依赖v6单种子或sidecar性能增益。"
            ),
            "principal_risks": [
                "soft sensor仍由图像条件生成，不是独立物理模态。",
                "固定内部测试集已用于多轮比较，缺少新采集外部盲测。",
                "不确定性相对排序改善，但绝对量值仍偏保守。",
            ],
            "selection_effect": (
                "对象级可靠性方法为主线；YOLO联合头与专利对齐作为扩展章节。"
            ),
        },
        {
            "id": "B",
            "recommended": False,
            "name": "专利对齐的可靠多模态检测与实例分割联合头",
            "title_candidates": [
                "专利对齐的可靠多模态垃圾检测与实例分割联合网络",
                "融合文本语义与软传感器可靠性的YOLO11s-seg联合头",
            ],
            "controlling_motivation": (
                "对象级复分类不能证明融合特征直接驱动检测和实例分割，"
                "需要把完整专利式融合栈接入原生Segment头并验证因果路径。"
            ),
            "contribution_promises": [
                "4×4观测与availability独立传输，RGB主干始终只接收3通道。",
                "Adapter、FiLM、余弦权重、8头交互、残差与动态尺度融合直接调制P3/P4/P5。",
                "结构干预验证sidecar到头前特征的因果路径与缺失回退不变性。",
                "同检查点调制器归零分解整体融合栈效应与sidecar特异效应。",
            ],
            "results_validation": [
                {
                    "promise": 1,
                    "evidence": "v6结构审计与4次主干通道捕获",
                },
                {
                    "promise": 2,
                    "evidence": "v6持久化检查点、模块形状和非零调制器权重",
                },
                {
                    "promise": 3,
                    "evidence": "0→1观测头前特征差异与availability=0严格不变",
                },
                {
                    "promise": 4,
                    "evidence": "完整相对调制器归零box/mask +0.418/+0.080 pp",
                },
            ],
            "reviewer_value": (
                "结构闭环和因果隔离清晰，但性能证据主要是五轮单种子rule开发实验。"
            ),
            "principal_risks": [
                "clean、fallback和shuffled任务指标近似相同，不能宣称sidecar增益。",
                "v6干净精度低于已选择的v5可靠性开发模型。",
                "缺少整图Qwen三种子、RGB-D、实体阵列和外部盲测。",
                "显式六向注意力拼接与字面8/16/32尺度只部分对齐。",
            ],
            "selection_effect": (
                "联合检测分割成为主线；对象级v3作为可靠性先导实验。"
            ),
        },
        {
            "id": "C",
            "recommended": False,
            "name": "面向一般智能垃圾分类的可靠多模态系统",
            "title_candidates": [
                "面向智能垃圾分类的可靠多模态识别与联合分割系统",
                "从对象分类到实例分割的图像条件多模态垃圾识别框架",
            ],
            "controlling_motivation": (
                "需要把对象级可靠分类、目标域分割、两阶段整图链和联合头"
                "组织为完整一般垃圾识别系统，并逐项说明与原专利的对应关系。"
            ),
            "contribution_promises": [
                "独立对象级三模态可靠分类器与Qwen/rule sidecar数据治理。",
                "目标域YOLO11s-seg与两阶段整图工程链。",
                "外部4×4 sidecar直接进入可靠联合Segment头。",
                "专利要求矩阵、复现边界和可复现证据包。",
            ],
            "results_validation": [
                {
                    "promise": 1,
                    "evidence": "对象级固定test、三种子、鲁棒性和来源实验",
                },
                {
                    "promise": 2,
                    "evidence": "YOLO-seg固定test与570图整图评估",
                },
                {
                    "promise": 3,
                    "evidence": "v2-v6联合头谱系、条件评估和因果干预",
                },
                {
                    "promise": 4,
                    "evidence": "20项专利矩阵与确定性无检查点归档",
                },
            ],
            "reviewer_value": (
                "适合学位论文或系统报告，覆盖面完整；短篇投稿可能因贡献过散而减弱主线。"
            ),
            "principal_risks": [
                "对象级Accuracy与联合头mAP属于不同任务，不能合并成一个主指标。",
                "系统没有RGB-D、实体传感器、小目标/遮挡人工标签或真实部署。",
                "模块和实验较多，短篇论文容易成为工程堆叠。",
            ],
            "selection_effect": (
                "采用系统型长文结构；A和B分别成为两个技术支柱。"
            ),
        },
    ]


def build_packet(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    missing = [name for name, path in SOURCES.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "motivation decision inputs missing: " + ", ".join(missing)
        )
    values = {name: _read_json(path) for name, path in SOURCES.items()}
    validity_checks = {
        "v3_seed_outcome": values["v3_seed_outcome"].get("valid") is True,
        "v3_reliability_ablation": (
            values["v3_reliability_ablation"].get("valid") is True
        ),
        "v5_reliability_audit": (
            values["v5_reliability_audit"].get("valid") is True
        ),
        "v6_patent_outcome": (
            values["v6_patent_outcome"].get("valid") is True
        ),
        "v6_intervention_audit": (
            values["v6_intervention_audit"].get("valid") is True
        ),
        "patent_matrix": values["patent_matrix"].get("valid") is True,
        "patent_addendum": values["patent_addendum"].get("valid") is True,
        "joint_multiseed": values["joint_multiseed"].get("valid") is True,
        "joint_retrained_ablations": (
            values["joint_retrained_ablations"].get("valid") is True
        ),
        "final_evidence_audit": (
            values["final_evidence_audit"].get("valid") is True
        ),
        "bibliography_audit": (
            values["bibliography_audit"].get("valid") is True
        ),
    }
    if not all(validity_checks.values()):
        failed = [name for name, passed in validity_checks.items() if not passed]
        raise ValueError(
            "motivation evidence is invalid: " + ", ".join(failed)
        )

    options = _option_definitions()
    if (
        sum(bool(option["recommended"]) for option in options) != 1
        or [option["id"] for option in options] != ["A", "B", "C"]
    ):
        raise ValueError("motivation option definitions are inconsistent")
    v3_delta = values["v3_reliability_ablation"][
        "matched_test_comparison"
    ]["aggregate"]
    v6_intervention = values["v6_intervention_audit"][
        "contribution_decomposition"
    ]
    packet = {
        "schema_version": 1,
        "state": "awaiting_user_confirmation",
        "valid": True,
        "formal_paper_metric": False,
        "recommended_option": "A",
        "confirmed_option": None,
        "confirmation_required": True,
        "selection_instruction": (
            "Choose A, B, or C before creating confirmed_motivation.md "
            "or fixing the final title, abstract, contribution order, "
            "and conclusion."
        ),
        "evidence_snapshot": {
            "v3_reliability_ablation_accuracy_mean_delta": v3_delta[
                "accuracy"
            ]["mean"],
            "v3_reliability_ablation_macro_f1_mean_delta": v3_delta[
                "macro_f1"
            ]["mean"],
            "v6_collective_conditioning_box_delta": v6_intervention[
                "full_stack_conditioning_effect"
            ]["box_map50_95"],
            "v6_collective_conditioning_mask_delta": v6_intervention[
                "full_stack_conditioning_effect"
            ]["mask_map50_95"],
            "v6_sidecar_specific_effect_max_abs": v6_intervention[
                "matched_sidecar_specific_effect_max_abs"
            ],
            "formal_qwen_three_seed_ready": False,
            "hardware_sensor_ready": False,
            "rgb_d_ready": False,
        },
        "options": options,
        "shared_nonnegotiable_boundaries": [
            "Qwen/rule观测必须称为图像条件soft sensor或sidecar，不得称为硬件实测。",
            "固定内部测试集不是外部盲测或真实部署精度。",
            "对象级Accuracy/Macro F1与联合头box/mask mAP必须分开报告。",
            "软件退化不等于硬件故障试验。",
            "当前系统无RGB-D、同步实体4×4阵列和人工遮挡标签。",
            "原始专利98.8%、93.3%、95.7%和15.2 ms不得继承。",
            "整图Qwen三种子正式协议完成前，rule单种子联合头不进入正式主表。",
        ],
        "paper_spine_gate": {
            "confirmed_motivation_exists": False,
            "confirmed_contribution_exists": False,
            "substantive_full_drafting_authorized": False,
            "reason": "controlling motivation requires explicit user selection",
        },
        "source_bindings": {
            **{name: _binding(path) for name, path in SOURCES.items()},
            "builder_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_text(
        output,
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
    )

    lines = [
        "# 最新论文主线决策包",
        "",
        "状态：`awaiting_user_confirmation`。本文件不会自动创建"
        "`confirmed_motivation.md`，也不会替用户固定最终标题、摘要、贡献排序或结论。",
        "",
        "## 推荐结论",
        "",
        "**推荐 A：低质量图像条件软传感器下的可靠多模态垃圾分类。**",
        "",
        "原因：A的核心贡献由对象级三训练种子、匹配种子独立重训消融、同检查点门控干预、"
        "软件退化与不确定性诊断共同支撑；v5/v6联合头可作为面向检测与实例分割的系统扩展。"
        "该主线不依赖把sidecar写成真实传感器，也不依赖v6尚未观察到的sidecar任务增益。",
        "",
    ]
    for option in options:
        suffix = "（推荐）" if option["recommended"] else ""
        lines.extend(
            [
                f"## 选项 {option['id']}：{option['name']}{suffix}",
                "",
                f"**控制性动机：** {option['controlling_motivation']}",
                "",
                "**贡献承诺：**",
                "",
                *[
                    f"{index}. {promise}"
                    for index, promise in enumerate(
                        option["contribution_promises"],
                        start=1,
                    )
                ],
                "",
                f"**审稿价值：** {option['reviewer_value']}",
                "",
                "**主要风险：**",
                "",
                *[f"- {risk}" for risk in option["principal_risks"]],
                "",
                "**标题候选：**",
                "",
                *[
                    f"- {title}"
                    for title in option["title_candidates"]
                ],
                "",
                f"**选择后章节重心：** {option['selection_effect']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 共同边界",
            "",
            *[
                f"- {boundary}"
                for boundary in packet[
                    "shared_nonnegotiable_boundaries"
                ]
            ],
            "",
            "## 需要的确认",
            "",
            "正式组稿前请选择 `A`、`B` 或 `C`。推荐选择 `A`；确认后才能建立"
            "`confirmed_motivation.md`、`confirmed_contribution.md`并固定标题、摘要、"
            "章节权重和结论。",
            "",
        ]
    )
    _atomic_text(output.with_suffix(".md"), "\n".join(lines))
    return packet


def main() -> None:
    result = build_packet(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "recommended_option": result["recommended_option"],
                "confirmed_option": result["confirmed_option"],
                "confirmation_required": result[
                    "confirmation_required"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
