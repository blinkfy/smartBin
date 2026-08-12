"""Build the v7 manuscript result inclusion and exclusion register."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evaluate_uncertainty import _source_binding
from mmfc.training import PROJECT_ROOT, resolve_project_path


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORTS / "MANUSCRIPT_RESULT_GOVERNANCE_V7.json"
INPUTS = {
    "final_claim_evidence": REPORTS / "FINAL_CLAIM_EVIDENCE.json",
    "paper_readiness": REPORTS / "PAPER_READINESS_AUDIT.json",
    "paper_readiness_v6_addendum": (
        REPORTS / "PAPER_READINESS_V6_ADDENDUM.json"
    ),
    "motivation_decision_packet": (
        REPORTS / "MOTIVATION_DECISION_PACKET.json"
    ),
    "patent_requirement_matrix": (
        REPORTS / "PATENT_REQUIREMENT_MATRIX.json"
    ),
    "qwen_distilled_extension": (
        REPORTS / "OBJECT_QWEN_DISTILLED_EXTENSION_INDEX.json"
    ),
    "qwen_distilled_addendum": (
        REPORTS / "OBJECT_QWEN_DISTILLED_JOINT_ADDENDUM.json"
    ),
    "uncertainty_temperature_addendum": (
        REPORTS / "LATENT_UNCERTAINTY_TEMPERATURE_ADDENDUM.json"
    ),
    "standardized_strata_addendum": (
        REPORTS
        / "standardized-instance-strata"
        / "MANUSCRIPT_ADDENDUM.json"
    ),
    "formal_qwen_v6_protocol": (
        REPORTS
        / "joint-multimodal-seg-patent-qwen"
        / "QWEN_FORMAL_EXTENSION_PROTOCOL.json"
    ),
    "submission_index": REPORTS / "SUBMISSION_INDEX.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v7 manuscript result governance register"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_project_path(path)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build() -> dict[str, Any]:
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "manuscript governance inputs missing: " + ", ".join(missing)
        )
    values = {name: _read(path) for name, path in INPUTS.items()}
    motivation = values["motivation_decision_packet"]
    patent = values["patent_requirement_matrix"]
    qwen = values["qwen_distilled_extension"]
    uncertainty = values["uncertainty_temperature_addendum"]
    strata = values["standardized_strata_addendum"]
    formal_qwen = values["formal_qwen_v6_protocol"]
    readiness_v6 = values["paper_readiness_v6_addendum"]
    if (
        motivation.get("state") != "awaiting_user_confirmation"
        or motivation.get("recommended_option") != "A"
        or motivation.get("confirmed_option") is not None
        or patent.get("valid") is not True
        or patent.get("summary", {}).get("complete_patent_replication")
        is not False
        or qwen.get("valid") is not True
        or qwen.get("formal_paper_metric") is not False
        or uncertainty.get("valid") is not True
        or uncertainty.get("formal_paper_metric") is not False
        or strata.get("valid") is not True
        or formal_qwen.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
        or readiness_v6.get("valid") is not True
    ):
        raise ValueError("manuscript governance source states are inconsistent")

    patent_counts = patent["summary"]["status_counts"]
    return {
        "schema_version": 1,
        "version": 7,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_scope": {
            "recommended_option": "A",
            "recommended_name": (
                "低质量图像条件软传感器下的可靠多模态垃圾分类"
            ),
            "user_confirmed": False,
            "final_title_locked": False,
            "final_abstract_locked": False,
            "draft_evidence_ready": True,
            "submission_ready": False,
            "reason_submission_not_ready": [
                "paper motivation has not been confirmed",
                "no new external blind validation batch",
                "formal direct full-image Qwen three-seed run is pending",
            ],
        },
        "primary_result_inclusion": [
            {
                "id": "P1",
                "result": (
                    "v3 reliable multimodal object classification and "
                    "three-seed reliability ablation"
                ),
                "role": "primary",
                "evidence": [
                    "FINAL_CLAIM_EVIDENCE.json",
                    (
                        "confirmatory-ablations/v3-no-reliability-workers0/"
                        "PAPER_ADDENDUM.json"
                    ),
                ],
                "allowed_claim": (
                    "Reliability-aware fusion improves robustness and "
                    "same-checkpoint behavior under registered software "
                    "sensor degradation."
                ),
                "boundary": (
                    "The sensor is image-conditioned and the fixed internal "
                    "test is not an external blind cohort."
                ),
            },
            {
                "id": "P2",
                "result": "formal object-level retrained ablations",
                "role": "primary_supporting",
                "evidence": ["ablations/ablation_summary.json"],
                "allowed_claim": (
                    "The reported modality and loss comparisons follow "
                    "independent retraining under the frozen protocol."
                ),
                "boundary": (
                    "Legacy full-model ablations are not component-wise v3 "
                    "cross-modal-quality ablations."
                ),
            },
            {
                "id": "P3",
                "result": "target-domain YOLO11s-seg performance",
                "role": "system_baseline",
                "evidence": ["yolo-seg/evaluation.json"],
                "allowed_claim": (
                    "The independent target-domain segmenter provides box "
                    "and instance-mask localization for the system."
                ),
                "boundary": (
                    "It does not by itself demonstrate multimodal fusion."
                ),
            },
        ],
        "supplementary_result_inclusion": [
            {
                "id": "S1",
                "result": (
                    "v5/v6 native Segment-head reliability and patent-stack "
                    "mechanism diagnostics"
                ),
                "role": "development_extension",
                "allowed_claim": (
                    "Multimodal conditioning reaches the native Segment "
                    "head and the reliability gate reacts to corruption."
                ),
                "forbidden_claim": (
                    "Single-seed rule-sidecar development establishes "
                    "formal segmentation superiority."
                ),
            },
            {
                "id": "S2",
                "result": "object-Qwen-distilled RGB scene sidecar",
                "role": "transparent_negative_ablation",
                "allowed_claim": (
                    "The RGB student reconstructs the scene grid beyond the "
                    "mean baseline and is compatible with the v6 stack."
                ),
                "forbidden_claim": (
                    "The distilled sidecar improves detection or segmentation."
                ),
                "accuracy_gain_detected": False,
            },
            {
                "id": "S3",
                "result": "validation-fit latent variance temperature",
                "role": "uncertainty_supplement",
                "allowed_claim": (
                    "Post-hoc validation scaling improves latent Gaussian "
                    "NLL and standardized-residual coverage while preserving "
                    "variance-error ranking."
                ),
                "forbidden_claim": (
                    "Latent calibration establishes physical sensor "
                    "uncertainty calibration."
                ),
                "test_nll_change_interval": uncertainty["test_metrics"][
                    "nll_cluster_bootstrap_interval"
                ],
            },
            {
                "id": "S4",
                "result": "COCO-scale and overlap-proxy diagnostics",
                "role": "sample_limited_posthoc",
                "allowed_claim": (
                    "Report descriptive internal recall by visible-mask "
                    "scale and box-overlap proxy."
                ),
                "forbidden_claim": (
                    "The 11 small and 20 high-overlap targets establish "
                    "robust small-object or occlusion performance."
                ),
                "support": strata["support"],
            },
        ],
        "mandatory_exclusions": [
            {
                "id": "X1",
                "claim": "98.8% accuracy or original patent performance values",
                "reason": "no reproducible source artifacts",
            },
            {
                "id": "X2",
                "claim": "physical 4x4 sensor measurements",
                "reason": "current observations are image-conditioned",
            },
            {
                "id": "X3",
                "claim": "RGB-D or depth-camera validation",
                "reason": "the backbone input remains three-channel RGB",
            },
            {
                "id": "X4",
                "claim": "formal direct full-image Qwen v6 results",
                "reason": "preregistered protocol is awaiting credentials/data",
            },
            {
                "id": "X5",
                "claim": "medical-waste taxonomy",
                "reason": "the implemented scope is 19-class general waste",
            },
            {
                "id": "X6",
                "claim": "external deployment generalization",
                "reason": "no independent blind cohort or hardware trial",
            },
        ],
        "paper_table_plan": [
            {
                "table": "Table 1",
                "content": "dataset composition, splits, Qwen coverage",
                "role": "main",
            },
            {
                "table": "Table 2",
                "content": "object-level main models and calibration metrics",
                "role": "main",
            },
            {
                "table": "Table 3",
                "content": "reliability intervention and degradation results",
                "role": "main",
            },
            {
                "table": "Table 4",
                "content": "YOLO-seg and joint-head development results",
                "role": "main_with_development_label",
            },
            {
                "table": "Table S1",
                "content": "independent retraining ablations",
                "role": "supplement",
            },
            {
                "table": "Table S2",
                "content": "latent uncertainty raw versus calibrated",
                "role": "supplement",
            },
            {
                "table": "Table S3",
                "content": "Qwen-distilled matched/fallback/shuffled negative result",
                "role": "supplement",
            },
            {
                "table": "Table S4",
                "content": "scale and overlap-proxy diagnostics",
                "role": "supplement",
            },
        ],
        "patent_alignment_snapshot": {
            "implemented": int(patent_counts["implemented"]),
            "partially_implemented": int(
                patent_counts["partially_implemented"]
            ),
            "not_implemented": int(patent_counts["not_implemented"]),
            "intentional_scope_change": int(
                patent_counts["intentional_scope_change"]
            ),
            "unsupported_performance_claim": int(
                patent_counts["unsupported_performance_claim"]
            ),
            "complete_patent_replication": False,
        },
        "remaining_gates": [
            {
                "priority": 1,
                "gate": "motivation_confirmation",
                "owner": "user",
                "state": "awaiting_A_B_or_C",
                "recommended": "A",
            },
            {
                "priority": 2,
                "gate": "formal_direct_full_image_qwen",
                "owner": "external_credentials",
                "state": "preregistered_not_started",
            },
            {
                "priority": 3,
                "gate": "external_blind_validation",
                "owner": "new_data_collection",
                "state": "not_available",
            },
            {
                "priority": 4,
                "gate": "hardware_or_rgbd_patent_completion",
                "owner": "hardware_and_data",
                "state": "not_implemented",
            },
            {
                "priority": 5,
                "gate": "small_and_occlusion_support",
                "owner": "new_annotations_or_data",
                "state": "sample_limited",
            },
        ],
        "bindings": {
            **{name: _source_binding(path) for name, path in INPUTS.items()},
            "builder_source": _source_binding(Path(__file__)),
        },
    }


def write(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    report = build()
    _atomic_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    patent = report["patent_alignment_snapshot"]
    lines = [
        "# 论文结果纳入与排除清单 v7",
        "",
        "## 当前判定",
        "",
        "- 论文证据已足够撰写草稿，但尚未达到投稿完成状态。",
        "- 推荐主线仍为 A：低质量图像条件软传感器下的可靠多模态垃圾分类。",
        "- 用户尚未确认 A/B/C，因此最终标题、摘要和贡献顺序不锁定。",
        "",
        "## 主文结果",
        "",
    ]
    for item in report["primary_result_inclusion"]:
        lines.append(
            f"- `{item['id']}` {item['result']}：{item['allowed_claim']} "
            f"边界：{item['boundary']}"
        )
    lines.extend(["", "## 补充材料", ""])
    for item in report["supplementary_result_inclusion"]:
        lines.append(
            f"- `{item['id']}` {item['result']}：可写 "
            f"{item['allowed_claim']} 禁止写 {item['forbidden_claim']}"
        )
    lines.extend(["", "## 必须排除", ""])
    for item in report["mandatory_exclusions"]:
        lines.append(
            f"- `{item['id']}` {item['claim']}：{item['reason']}。"
        )
    lines.extend(
        [
            "",
            "## 专利对齐快照",
            "",
            (
                f"- 已实现 {patent['implemented']}，部分实现 "
                f"{patent['partially_implemented']}，未实现 "
                f"{patent['not_implemented']}，主动范围调整 "
                f"{patent['intentional_scope_change']}，不支持的性能主张 "
                f"{patent['unsupported_performance_claim']}。"
            ),
            "- 当前不是完整专利复现。",
            "",
            "## 剩余门",
            "",
        ]
    )
    for gate in report["remaining_gates"]:
        lines.append(
            f"- P{gate['priority']} `{gate['gate']}`：{gate['state']}。"
        )
    lines.append("")
    _atomic_text(output.with_suffix(".md"), "\n".join(lines))
    return report


def main() -> None:
    report = write(parse_args().output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "draft_evidence_ready": report["paper_scope"][
                    "draft_evidence_ready"
                ],
                "submission_ready": report["paper_scope"][
                    "submission_ready"
                ],
                "recommended_option": report["paper_scope"][
                    "recommended_option"
                ],
                "remaining_gate_count": len(report["remaining_gates"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
