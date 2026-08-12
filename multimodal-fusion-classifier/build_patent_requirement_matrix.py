"""Build an evidence-bound patent requirement matrix for paper planning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_patent_aligned_joint_seg import (
    DEFAULT_REPORT_RUN,
    extract_patent_claims,
)
from mmfc.model import PROJECT_ROOT


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "PATENT_REQUIREMENT_MATRIX.json"
)
STATUS_LABELS = {
    "implemented": "已实现",
    "partially_implemented": "部分实现",
    "not_implemented": "未实现",
    "intentional_scope_change": "有意变更范围",
    "unsupported_performance_claim": "无可复核证据",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the claim-by-claim patent requirement matrix"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def requirement_rows() -> list[dict[str, Any]]:
    """Return the curated implementation status for each material requirement."""

    return [
        {
            "id": "C1-S1",
            "claim": 1,
            "requirement": "采集深度图像、类别文本和4x4传感器阵列并完成预处理",
            "status": "partially_implemented",
            "implementation_evidence": [
                "mmfc/training.py::patent_preprocess",
                "build_scene_sensor_sidecars.py",
                "mmfc/model.py::CLIPPrototypeEncoder",
            ],
            "experimental_evidence": [
                "data/scene-sensor-sidecars/BUILD_REPORT.json",
            ],
            "boundary": "当前为RGB和图像条件soft-sensor sidecar，无深度通道或同步实体阵列。",
        },
        {
            "id": "C1-S2",
            "claim": 1,
            "requirement": "YOLO11s-seg、CLIP和传感器编码器提取特征，并以CE与Focal约束训练",
            "status": "partially_implemented",
            "implementation_evidence": [
                "mmfc/model.py::MultimodalFusionClassifier",
                "mmfc/losses.py",
                "mmfc/joint_seg_patent.py",
            ],
            "experimental_evidence": [
                "reports/qwen-material/FINAL_EVIDENCE_AUDIT.json",
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/evaluation.json",
            ],
            "boundary": "对象级分类使用CE+Focal；v6联合头使用Ultralytics原生检测/分割损失及传感器辅助损失。",
        },
        {
            "id": "C1-S3",
            "claim": 1,
            "requirement": "Adapter、FiLM、余弦权重、深层交互、残差和多尺度融合形成最终特征",
            "status": "implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_patent.py::PatentAlignedReliableFeatureFusion",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
                "reports/qwen-material/PATENT_STACK_INTERVENTION_AUDIT.json",
            ],
            "boundary": "已证明所有模块持久化并因果改变头前特征；具体运算顺序与原文部分公式不完全同构。",
        },
        {
            "id": "C1-S4",
            "claim": 1,
            "requirement": "融合特征进入原生检测与实例分割头并输出类别、位置、置信度和掩膜",
            "status": "implemented",
            "implementation_evidence": [
                "mmfc/joint_seg.py::JointMultimodalSegmentationModel",
                "mmfc/joint_seg_patent.py::PatentAlignedReliableJointSegmentationModel",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
                "reports/qwen-material/PATENT_STACK_INTERVENTION_AUDIT.json",
            ],
            "boundary": "v6直接调制P3/P4/P5后进入同一Ultralytics Segment头，不是两阶段复分类替代物。",
        },
        {
            "id": "C2-IMAGE",
            "claim": 2,
            "requirement": "双边滤波、CLAHE、拉普拉斯锐化和仿射变换",
            "status": "partially_implemented",
            "implementation_evidence": [
                "mmfc/training.py::patent_preprocess",
            ],
            "experimental_evidence": [
                "reports/qwen-material/PATENT_ALIGNMENT.md",
            ],
            "boundary": "对象级路径具备这些操作；CLAHE按亮度触发，仿射矩阵当前为单位变换，联合头未复用该CPU预处理。",
        },
        {
            "id": "C2-SENSOR",
            "claim": 2,
            "requirement": "实体传感器采集阵列数据，并上采样与归一化",
            "status": "not_implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_reliable.py::pack_reliable_sensor_channels",
                "mmfc/physical_sensor.py::calibrate_physical_observation",
                "prepare_physical_sensor_sidecars.py",
                "mmfc/physical_release.py::release_sidecars",
                "prepare_synchronized_hardware_release.py",
                "schemas/synchronized_hardware_attestation_v1.schema.json",
            ],
            "experimental_evidence": [
                "data/scene-sensor-sidecars/BUILD_REPORT.json",
                "reports/qwen-material/PHYSICAL_SENSOR_INTERFACE_AUDIT.json",
            ],
            "boundary": (
                "4x4传输、逐格校准、时间同步校验、归一化、上采样及保管人证明后"
                "盲测发布软件链已实现，但实体阵列记录仍为0，未完成硬件采集与性能验证。"
            ),
        },
        {
            "id": "C3-ENCODERS",
            "claim": 3,
            "requirement": "加载YOLO11s-seg与CLIP ViT-B/16预训练权重、冻结前层并编码传感器",
            "status": "partially_implemented",
            "implementation_evidence": [
                "mmfc/model.py::YOLOMultiScaleEncoder",
                "mmfc/model.py::CLIPPrototypeEncoder",
                "mmfc/model.py::SensorEncoder",
                "train_patent_aligned_external_sensor_joint_seg.py",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/PROTOCOL.json",
            ],
            "boundary": "对象级模型按前5/前3层冻结；v6冻结YOLO前24层并使用预计算冻结文本原型，策略不完全相同。",
        },
        {
            "id": "C4",
            "claim": 4,
            "requirement": "三个独立512维Adapter适配后在通道维拼接",
            "status": "partially_implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_patent.py::patent_adapters",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
            ],
            "boundary": "三个独立512维Adapter、LeakyReLU和dropout=0.3一致；当前保留三个token交互而非先做1536维原始拼接。",
        },
        {
            "id": "C5",
            "claim": 5,
            "requirement": "以gamma=1.0、beta=0.1初始化FiLM逐模态调制",
            "status": "implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_patent.py::patent_film_gamma",
                "mmfc/joint_seg_patent.py::patent_film_beta",
            ],
            "experimental_evidence": [
                "tests/test_joint_seg_patent.py",
            ],
            "boundary": "三组512维可学习参数已实现并随检查点持久化。",
        },
        {
            "id": "C6",
            "claim": 6,
            "requirement": "计算视觉、文本和传感器特征两两余弦相似度",
            "status": "implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_patent.py::_modality_weights",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
            ],
            "boundary": "权重在float32归一化空间计算并softmax为概率单纯形。",
        },
        {
            "id": "C7",
            "claim": 7,
            "requirement": "余弦得分softmax后加权融合，并执行512到128维下投影",
            "status": "partially_implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_patent.py::patent_down_projection",
                "mmfc/joint_seg_patent.py::_modality_weights",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
            ],
            "boundary": "两项运算均存在，但v6先下投影三个token、交互后再按余弦权重汇聚，不是原文的先汇聚再下投影。",
        },
        {
            "id": "C8",
            "claim": 8,
            "requirement": "三组模态对双向注意力、输出拼接及128到512维上投影",
            "status": "partially_implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_patent.py::patent_interaction",
                "mmfc/joint_seg_patent.py::patent_up_projection",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
            ],
            "boundary": "采用三个模态token的8头自注意力，等价覆盖双向信息交换，但未显式构造六个有向输出再拼接。",
        },
        {
            "id": "C9",
            "claim": 9,
            "requirement": "交互恢复特征与调制后特征残差连接",
            "status": "implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_patent.py::residual_tokens",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
            ],
            "boundary": "逐模态token在512维空间完成残差连接。",
        },
        {
            "id": "C10",
            "claim": 10,
            "requirement": "对8x8、16x16、32x32尺度按0.3/0.5/0.2先验动态融合",
            "status": "partially_implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_patent.py::patent_dynamic_scale_gate",
                "mmfc/joint_seg_patent.py::patent_scale_prior_logits",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
            ],
            "boundary": "对原生P3/P4/P5动态加权且先验一致；480输入下空间尺寸不是原文逐字规定的8/16/32。",
        },
        {
            "id": "DETAIL-NMS",
            "claim": None,
            "requirement": "NMS IoU=0.5且置信度阈值=0.8",
            "status": "implemented",
            "implementation_evidence": [
                "predict_scene.py::parse_args",
                "evaluate_scene_pipeline.py::parse_args",
            ],
            "experimental_evidence": [
                "reports/qwen-material/yolo-seg/integrated/evaluation.json",
            ],
            "boundary": "接口和冻结整图评估覆盖0.5/0.8；v6 mAP验证采用标准置信度扫描，不能混写为固定0.8指标。",
        },
        {
            "id": "SCOPE-TAXONOMY",
            "claim": None,
            "requirement": "医疗废物5大类、20个细分类",
            "status": "intentional_scope_change",
            "implementation_evidence": [
                "configs/default.json",
            ],
            "experimental_evidence": [
                "reports/qwen-material/DATASET_CARD.md",
            ],
            "boundary": "研究任务已明确调整为19类一般智能垃圾分类，论文标题、数据和结论必须使用新范围。",
        },
        {
            "id": "DETAIL-RGBD",
            "claim": None,
            "requirement": "RGB-D深度摄像头输入",
            "status": "not_implemented",
            "implementation_evidence": [
                "mmfc/joint_seg_reliable.py::split_reliable_sensor_channels",
            ],
            "experimental_evidence": [
                "reports/qwen-material/joint-multimodal-seg-patent/"
                "yolo11s-seg-joint-patent-stack-rule-dev-seed42/OUTCOME.json",
            ],
            "boundary": "主干前向严格为3通道RGB；额外两通道只传输网格与可用性，不是深度。",
        },
        {
            "id": "DETAIL-HARDWARE",
            "claim": None,
            "requirement": "同步4x4实体传感器阵列观测",
            "status": "not_implemented",
            "implementation_evidence": [
                "build_scene_sensor_sidecars.py",
                "mmfc/physical_sensor.py::calibrate_physical_observation",
                "prepare_physical_sensor_sidecars.py",
                "mmfc/physical_release.py::release_sidecars",
                "prepare_synchronized_hardware_release.py",
                "schemas/synchronized_hardware_attestation_v1.schema.json",
            ],
            "experimental_evidence": [
                "data/scene-sensor-sidecars/BUILD_REPORT.json",
                "reports/qwen-material/PHYSICAL_SENSOR_INTERFACE_AUDIT.json",
            ],
            "boundary": (
                "当前正式观测仍由图像统计或Qwen视觉推断生成；实体阵列接入、"
                "校准、同步及保管人证明发布契约虽已实现，但没有真实硬件记录。"
            ),
        },
        {
            "id": "DETAIL-STRATA",
            "claim": None,
            "requirement": "人工定义的小目标与遮挡测试子集",
            "status": "partially_implemented",
            "implementation_evidence": [
                "evaluate_scene_strata.py",
                "evaluate_standardized_instance_strata.py",
                "audit_standardized_instance_strata.py",
            ],
            "experimental_evidence": [
                "reports/qwen-material/standardized-instance-strata/"
                "PROTOCOL.json",
                "reports/qwen-material/standardized-instance-strata/"
                "evaluation-final-chunk64/evaluation.json",
                "reports/qwen-material/standardized-instance-strata/"
                "AUDIT.json",
            ],
            "boundary": (
                "已按COCO阈值构建可见掩膜小/中/大尺度，并按框重叠比例构建"
                "拥挤代理；仍无人工遮挡或amodal标签，且小目标和高重叠仅11/20例。"
            ),
        },
        {
            "id": "PERFORMANCE-CLAIMS",
            "claim": None,
            "requirement": "98.8%准确率、93.3%小目标、95.7%遮挡和15.2 ms",
            "status": "unsupported_performance_claim",
            "implementation_evidence": [],
            "experimental_evidence": [
                "reports/qwen-material/PATENT_ALIGNMENT.md",
            ],
            "boundary": "原始DOCX没有提供可复核清单、检查点或逐样本预测；当前论文不得继承这些数值。",
        },
    ]


def build_matrix(
    output: Path = DEFAULT_OUTPUT,
    report_run: Path = DEFAULT_REPORT_RUN,
) -> dict[str, Any]:
    output = _resolve(output)
    report_run = _resolve(report_run)
    patent_document = _patent_document()
    paths = {
        "patent_document": patent_document,
        "v6_protocol": report_run / "PROTOCOL.json",
        "v6_evaluation": report_run / "evaluation.json",
        "v6_conditions": report_run / "RELIABILITY_CONDITIONS.json",
        "v6_outcome": report_run / "OUTCOME.json",
        "v6_model_source": PROJECT_ROOT / "mmfc" / "joint_seg_patent.py",
        "object_model_source": PROJECT_ROOT / "mmfc" / "model.py",
        "preprocessing_source": PROJECT_ROOT / "mmfc" / "training.py",
        "scene_inference_source": PROJECT_ROOT / "predict_scene.py",
        "standardized_strata_evaluator": (
            PROJECT_ROOT / "evaluate_standardized_instance_strata.py"
        ),
        "standardized_strata_auditor": (
            PROJECT_ROOT / "audit_standardized_instance_strata.py"
        ),
        "standardized_strata_protocol": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "standardized-instance-strata"
            / "PROTOCOL.json"
        ),
        "standardized_strata_evaluation": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "standardized-instance-strata"
            / "evaluation-final-chunk64"
            / "evaluation.json"
        ),
        "standardized_strata_audit": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "standardized-instance-strata"
            / "AUDIT.json"
        ),
        "physical_sensor_module": (
            PROJECT_ROOT / "mmfc" / "physical_sensor.py"
        ),
        "physical_sensor_converter": (
            PROJECT_ROOT / "prepare_physical_sensor_sidecars.py"
        ),
        "physical_release_module": (
            PROJECT_ROOT / "mmfc" / "physical_release.py"
        ),
        "physical_release_converter": (
            PROJECT_ROOT / "prepare_synchronized_hardware_release.py"
        ),
        "physical_sensor_schema": (
            PROJECT_ROOT
            / "schemas"
            / "physical_sensor_observation_v1.schema.json"
        ),
        "physical_release_attestation_schema": (
            PROJECT_ROOT
            / "schemas"
            / "synchronized_hardware_attestation_v1.schema.json"
        ),
        "physical_release_document": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "SYNCHRONIZED_HARDWARE_RELEASE.md"
        ),
        "physical_sensor_interface_audit": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "PHYSICAL_SENSOR_INTERFACE_AUDIT.json"
        ),
        "builder_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "patent matrix inputs missing: " + ", ".join(missing)
        )
    outcome = _read_json(paths["v6_outcome"])
    if (
        outcome.get("valid") is not True
        or outcome.get("formal_paper_metric") is not False
    ):
        raise ValueError("v6 structural outcome is not valid development evidence")
    claims = extract_patent_claims(patent_document)
    if sorted(claims) != list(range(1, 11)):
        raise ValueError("patent claims 1-10 were not extracted")

    rows = requirement_rows()
    unknown_status = sorted(
        {
            str(row.get("status"))
            for row in rows
            if row.get("status") not in STATUS_LABELS
        }
    )
    if unknown_status:
        raise ValueError("unknown requirement status: " + ", ".join(unknown_status))
    missing_evidence: list[str] = []
    evidence_paths: dict[str, Path] = {}
    for row in rows:
        for evidence in (
            list(row["implementation_evidence"])
            + list(row["experimental_evidence"])
        ):
            source_path = evidence.split("::", maxsplit=1)[0]
            resolved_evidence = PROJECT_ROOT / source_path
            evidence_paths[source_path] = resolved_evidence
            if not resolved_evidence.is_file():
                missing_evidence.append(f"{row['id']}:{evidence}")
    if missing_evidence:
        raise FileNotFoundError(
            "requirement evidence is missing: " + ", ".join(missing_evidence)
        )
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in STATUS_LABELS
    }
    matrix = {
        "schema_version": 1,
        "valid": True,
        "formal_paper_metric": False,
        "purpose": "patent_alignment_and_paper_scope_control",
        "patent": {
            **_binding(patent_document),
            "claim_count": len(claims),
            "claim_numbers": sorted(claims),
            "claim_text_sha256": {
                str(number): hashlib.sha256(text.encode("utf-8")).hexdigest()
                for number, text in claims.items()
            },
        },
        "summary": {
            "requirement_count": len(rows),
            "status_counts": counts,
            "complete_patent_replication": False,
            "architecture_method_section_ready_with_boundaries": True,
            "formal_qwen_three_seed_performance_ready": False,
            "hardware_validation_ready": False,
            "selected_development_interpretation": outcome["test"][
                "conclusion"
            ],
        },
        "requirements": rows,
        "paper_claim_policy": {
            "allowed_now": [
                "实现了专利式三模态融合栈并将其接入原生YOLO11s-seg Segment头。",
                "结构干预证明4x4观测可因果改变P3/P4/P5头前特征。",
                "缺失观测时回退路径对无效sidecar严格不变。",
                "当前rule-sidecar开发实验没有显示可观的汇总任务增益。",
                (
                    "实体4x4阵列的软件接入、逐格校准与时间同步契约已实现，"
                    "但真实硬件记录数仍为0。"
                ),
                (
                    "标准化分层诊断中，v5/v6在无框重叠子集的掩膜"
                    "mR50:95相对RGB分别提高约0.52/0.85个百分点；"
                    "该结果属于事后单测试集开发证据。"
                ),
            ],
            "not_allowed_now": [
                "真实物理传感器测量或硬件鲁棒性已经得到验证。",
                "已经复现完整RGB-D医疗废物专利系统。",
                "当前结果达到原专利98.8%、93.3%、95.7%或15.2 ms。",
                "单种子rule-sidecar结果属于预注册正式Qwen结论。",
                "现有11例小目标或20例高重叠代理足以支持稳健的小目标或遮挡性能结论。",
            ],
        },
        "remaining_external_evidence": [
            {
                "priority": 1,
                "item": "完成预注册整图Qwen sidecar三种子训练与固定测试汇总",
                "current_blocker": "DASHSCOPE_API_KEY环境变量不可用",
            },
            {
                "priority": 2,
                "item": "采集同步4x4实体传感器数据并记录校准、时间同步和单位",
                "current_blocker": "需要外部硬件与采集流程",
            },
            {
                "priority": 3,
                "item": "补充RGB-D输入和深度消融",
                "current_blocker": "现有原图没有深度通道",
            },
            {
                "priority": 4,
                "item": "扩充小目标样本并人工标注遮挡程度或amodal目标",
                "current_blocker": "需要新增数据、人工标注规则和双人复核",
            },
        ],
        "bindings": {name: _binding(path) for name, path in paths.items()},
        "evidence_bindings": {
            name: _binding(path)
            for name, path in sorted(evidence_paths.items())
        },
    }
    _atomic_json(output, matrix)

    status_labels = STATUS_LABELS
    markdown = [
        "# 专利要求证据矩阵",
        "",
        f"- 原始权利要求：`{len(claims)}` 项；矩阵拆分：`{len(rows)}` 项。",
        "- 完整专利复现：`false`；方法结构章节：可在边界声明下撰写；正式Qwen三种子结论：`false`。",
        "- v6结论：融合栈存在可验证的头前因果路径，但当前rule sidecar汇总任务增益可忽略。",
        "",
        "| 编号 | 要求 | 状态 | 证据边界 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['id']} | {row['requirement']} | "
            f"{status_labels[row['status']]} | {row['boundary']} |"
        )
    markdown.extend(
        [
            "",
            "## 可写入论文",
            "",
            "- 已实现专利式三模态融合栈，并将融合后的P3/P4/P5直接送入原生Segment头。",
            "- 结构干预表明外部4x4观测能改变三个尺度的头前特征；availability=0时回退严格不变。",
            "- 当前观测应称为图像条件soft sensor、Qwen辅助先验或sidecar，不能称为硬件实测。",
            "- 实体4x4阵列的软件接入、逐格校准和时间同步契约已实现，但真实记录仍为0。",
            "- 当前v6是五轮单种子rule开发证据，且任务级传感器效应可忽略。",
            "- 标准化事后分层可描述无框重叠子集结果；小目标和高重叠子集仅作探索性报告。",
            "",
            "## 仍需外部证据",
            "",
            "1. 提供API环境变量后执行预注册整图Qwen三种子正式实验。",
            "2. 采集同步实体4x4传感器并记录单位、校准和时间同步。",
            "3. 补充RGB-D深度输入及深度消融。",
            "4. 扩充小目标样本，并人工标注遮挡程度或amodal目标以替代框重叠代理。",
            "",
            "原始DOCX中的98.8%、93.3%、95.7%和15.2 ms没有可复核实验产物，"
            "不得作为当前论文结果引用。",
            "",
        ]
    )
    markdown_path = output.with_suffix(".md")
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(markdown), encoding="utf-8")
    temporary.replace(markdown_path)
    return matrix


def main() -> None:
    args = parse_args()
    result = build_matrix(args.output, args.report_run)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": result["valid"],
                "requirement_count": result["summary"]["requirement_count"],
                "status_counts": result["summary"]["status_counts"],
                "complete_patent_replication": result["summary"][
                    "complete_patent_replication"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
