"""Audit the result-independent manuscript sections and their citations."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_SOURCES = {
    "manuscript": REPORTS_ROOT / "MANUSCRIPT_STABLE_SECTIONS.md",
    "bibliography": REPORTS_ROOT / "reference_candidates.bib",
    "method_spec": REPORTS_ROOT / "METHOD_SPEC.md",
    "dataset_card": REPORTS_ROOT / "DATASET_CARD.md",
    "statistical_protocol": REPORTS_ROOT / "STATISTICAL_PROTOCOL.md",
    "patent_alignment": REPORTS_ROOT / "PATENT_ALIGNMENT.md",
    "release_checklist": REPORTS_ROOT / "ARTIFACT_RELEASE_CHECKLIST.md",
    "image_recovery_outcome": REPORTS_ROOT / "IMAGE_READ_RECOVERY_OUTCOME.json",
    "source_repair_transition": (
        REPORTS_ROOT / "SOURCE_IMAGE_REPAIR_TRANSITION.json"
    ),
    "formal_interruption_addendum": (
        REPORTS_ROOT / "FORMAL_QUEUE_INTERRUPTION_ADDENDUM.json"
    ),
    "formal_interruption_outcome": (
        REPORTS_ROOT / "FORMAL_QUEUE_INTERRUPTION_OUTCOME.json"
    ),
    "yolo_dependency_incident": (
        REPORTS_ROOT / "YOLO_SEG_DEPENDENCY_INCIDENT.json"
    ),
    "yolo_dependency_incident_addendum": (
        REPORTS_ROOT / "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.json"
    ),
    "yolo_dependency_outcome": (
        REPORTS_ROOT / "YOLO_SEG_DEPENDENCY_OUTCOME.json"
    ),
    "integrated_scene_protocol_addendum": (
        REPORTS_ROOT
        / "yolo-seg"
        / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
    ),
    "stratified_scene_protocol_addendum": (
        REPORTS_ROOT
        / "yolo-seg"
        / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
    ),
    "scene_strata_reporting_incident": (
        REPORTS_ROOT
        / "yolo-seg"
        / "SCENE_STRATA_REPORTING_INCIDENT.json"
    ),
    "scene_strata_reporting_addendum": (
        REPORTS_ROOT
        / "yolo-seg"
        / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
    ),
    "scene_strata_reporting_outcome": (
        REPORTS_ROOT
        / "yolo-seg"
        / "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json"
    ),
    "sensor_retraining_exit_incident": (
        REPORTS_ROOT / "SENSOR_RETRAINING_EXIT_INCIDENT.json"
    ),
    "sensor_retraining_exit_addendum": (
        REPORTS_ROOT / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json"
    ),
    "sensor_retraining_exit_encoding_addendum": (
        REPORTS_ROOT / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.json"
    ),
    "sensor_retraining_log_encoding_repair": (
        REPORTS_ROOT / "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json"
    ),
    "sensor_retraining_exit_outcome": (
        REPORTS_ROOT / "SENSOR_RETRAINING_EXIT_OUTCOME.json"
    ),
    "v3_config": (
        PROJECT_ROOT / "configs" / "ablations" / "full_cross_modal_quality.json"
    ),
    "yolo_run_manifest": REPORTS_ROOT / "yolo-seg" / "RUN_MANIFEST.json",
}
DEFAULT_OUTPUT = REPORTS_ROOT / "MANUSCRIPT_STABLE_SECTIONS_AUDIT.json"
REQUIRED_HEADINGS = (
    "## 1 研究背景与相关工作",
    "## 2 数据与任务定义",
    "## 3 图像条件软传感器",
    "## 4 多模态可靠性融合方法",
    "## 5 实验设置",
    "## 6 统计分析",
    "## 7 系统范围与专利对齐边界",
)
FORBIDDEN_HEADINGS = (
    "## 摘要",
    "## 结果",
    "## 结论",
    "## Abstract",
    "## Results",
    "## Conclusion",
)
REQUIRED_BOUNDARIES = (
    "图像条件软传感器",
    "不是同步实体传感器测量",
    "对象级主模型不预测边界框或掩码",
    "融合特征没有注入 YOLO Detect/Segment head",
    "输入是 RGB 而非 RGB-D",
    "内部固定留出基准",
    "未完成独立隐私审查",
    "写入来源和精确时刻未知",
    "不等同于专利实施例的医疗废物类别体系",
)
REQUIRED_METHOD_CONTRACTS = (
    "正式配置的仿射矩阵是单位矩阵",
    "三尺度先验初始化为 `0.3/0.5/0.2`",
    "文本查询温度为 `0.07`",
    "dropout 为 `0.3`",
    "对数方差限制在 `[-3,2]`",
    (
        "missing、noise、bias 和 stuck 触发概率依次为 "
        "`0.10`、`0.50`、`0.25` 和 `0.15`"
    ),
    "学习率在 5 轮线性 warmup 后按 epoch 余弦衰减至零",
    "`optimizer=auto`",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit stable manuscript prose and citation keys"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _citation_keys(text: str) -> list[str]:
    return sorted(set(re.findall(r"@([A-Za-z0-9_:-]+)", text)))


def _bibliography_keys(text: str) -> set[str]:
    return set(
        re.findall(
            r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,",
            text,
            flags=re.IGNORECASE,
        )
    )


def audit_manuscript_stable_sections(
    *,
    sources: Mapping[str, Path],
) -> dict[str, Any]:
    resolved = {name: _resolve(path) for name, path in sources.items()}
    missing_sources = [
        name for name, path in resolved.items() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "manuscript audit sources are missing: "
            + ", ".join(sorted(missing_sources))
        )
    manuscript = resolved["manuscript"].read_text(encoding="utf-8-sig")
    bibliography = resolved["bibliography"].read_text(
        encoding="utf-8-sig"
    )
    citations = _citation_keys(manuscript)
    bibliography_keys = _bibliography_keys(bibliography)
    normalized_manuscript = re.sub(r"\s+", "", manuscript)
    missing_citations = sorted(set(citations) - bibliography_keys)
    missing_headings = [
        heading for heading in REQUIRED_HEADINGS if heading not in manuscript
    ]
    present_forbidden_headings = [
        heading for heading in FORBIDDEN_HEADINGS if heading in manuscript
    ]
    missing_boundaries = [
        boundary
        for boundary in REQUIRED_BOUNDARIES
        if re.sub(r"\s+", "", boundary) not in normalized_manuscript
    ]
    missing_method_contracts = [
        contract
        for contract in REQUIRED_METHOD_CONTRACTS
        if re.sub(r"\s+", "", contract) not in normalized_manuscript
    ]
    percentage_literals = sorted(
        set(re.findall(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?%", manuscript))
    )
    errors: list[str] = []
    if missing_headings:
        errors.append(
            "required manuscript headings are missing: "
            + ", ".join(missing_headings)
        )
    if present_forbidden_headings:
        errors.append(
            "result-dependent headings are present: "
            + ", ".join(present_forbidden_headings)
        )
    if missing_boundaries:
        errors.append(
            "required claim boundaries are missing: "
            + ", ".join(missing_boundaries)
        )
    if missing_method_contracts:
        errors.append(
            "required method contracts are missing: "
            + ", ".join(missing_method_contracts)
        )
    if missing_citations:
        errors.append(
            "citation keys are missing from the bibliography: "
            + ", ".join(missing_citations)
        )
    if percentage_literals:
        errors.append(
            "stable manuscript contains percentage literals: "
            + ", ".join(percentage_literals)
        )
    if len(citations) < 10:
        errors.append("stable manuscript has fewer than 10 citation keys")
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "result_independent_manuscript_sections_and_claim_boundaries",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "summary": {
            "required_heading_count": len(REQUIRED_HEADINGS),
            "citation_count": len(citations),
            "missing_citation_count": len(missing_citations),
            "required_boundary_count": len(REQUIRED_BOUNDARIES),
            "required_method_contract_count": len(
                REQUIRED_METHOD_CONTRACTS
            ),
            "percentage_literal_count": len(percentage_literals),
        },
        "citations": citations,
        "missing_citations": missing_citations,
        "missing_headings": missing_headings,
        "present_forbidden_headings": present_forbidden_headings,
        "missing_boundaries": missing_boundaries,
        "missing_method_contracts": missing_method_contracts,
        "percentage_literals": percentage_literals,
        "source_bindings": {
            name: _binding(path) for name, path in sorted(resolved.items())
        },
        "errors": errors,
        "boundaries": [
            (
                "This audit checks structure, source bindings, citation keys, "
                "and required claim boundaries; it does not peer-review prose."
            ),
            (
                "Title, abstract, result interpretation, and conclusion remain "
                "outside this result-independent manuscript artifact."
            ),
            (
                "Dynamic result numbers must be inserted from the strictly "
                "audited paper-table bundle."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# 论文稳定章节审计",
            "",
            f"生成时间：`{report['generated_at']}`。",
            "",
            (
                f"必需章节 {summary['required_heading_count']} 个，引用键 "
                f"{summary['citation_count']} 个，缺失引用 "
                f"{summary['missing_citation_count']} 个，百分数字面量 "
                f"{summary['percentage_literal_count']} 个，方法契约 "
                f"{summary['required_method_contract_count']} 项。"
            ),
            (
                "软传感器、对象级分类、两阶段系统、RGB-D 缺口和内部固定基准"
                "边界，以及隐私审查待完成的数据治理边界和训练完整性事件均已"
                "锁定。"
            ),
            "",
            "本审计不替代同行评审；标题、摘要、结果和结论仍需在最终表组完成后编写。",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    report = audit_manuscript_stable_sections(sources=DEFAULT_SOURCES)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "valid": report["valid"],
                **report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
