"""Build an additive paper-readiness update for post-audit v6 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORTS / "PAPER_READINESS_V6_ADDENDUM.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "qwen-material" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the additive v6 paper-readiness update"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _read(path: Path) -> dict[str, Any]:
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


def build_addendum(
    output: Path = DEFAULT_OUTPUT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    output = _resolve(output)
    checkpoint_path = _resolve(checkpoint_path)
    paths = {
        "historical_readiness_audit": REPORTS / "PAPER_READINESS_AUDIT.json",
        "historical_evidence_audit": REPORTS / "FINAL_EVIDENCE_AUDIT.json",
        "object_test_evaluation": REPORTS / "test" / "evaluation.json",
        "uncertainty_floor_audit": REPORTS / "UNCERTAINTY_FLOOR_AUDIT.json",
        "uncertainty_comparison": (
            REPORTS / "UNCERTAINTY_DIAGNOSTICS_COMPARISON.json"
        ),
        "patent_stack_index": REPORTS / "PATENT_STACK_EXTENSION_INDEX.json",
        "standardized_strata_audit": (
            REPORTS / "standardized-instance-strata" / "AUDIT.json"
        ),
        "motivation_packet": REPORTS / "MOTIVATION_DECISION_PACKET.json",
        "qwen_extension_preflight": (
            REPORTS
            / "joint-multimodal-seg-patent-qwen"
            / "QWEN_FORMAL_EXTENSION_PREFLIGHT.json"
        ),
        "object_best_checkpoint": checkpoint_path,
        "builder_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "paper-readiness addendum inputs missing: " + ", ".join(missing)
        )

    readiness = _read(paths["historical_readiness_audit"])
    evidence = _read(paths["historical_evidence_audit"])
    object_test = _read(paths["object_test_evaluation"])
    floor = _read(paths["uncertainty_floor_audit"])
    uncertainty = _read(paths["uncertainty_comparison"])
    patent = _read(paths["patent_stack_index"])
    strata = _read(paths["standardized_strata_audit"])
    motivation = _read(paths["motivation_packet"])
    qwen = _read(paths["qwen_extension_preflight"])
    historical_warnings = set(readiness.get("warnings", []))
    required_historical_warnings = {
        "Target-domain YOLO-seg remains separate from the MMFC feature head.",
        (
            "Relative-size and bbox-overlap strata are not human "
            "small-target or occlusion labels."
        ),
    }
    metrics = object_test.get("metrics", {})
    checkpoint = object_test.get("checkpoint", {})
    if (
        readiness.get("valid") is not True
        or readiness.get("internal_paper_evidence_ready") is not True
        or readiness.get("full_patent_reproduction_ready") is not False
        or evidence.get("valid") is not True
        or not required_historical_warnings.issubset(historical_warnings)
        or checkpoint.get("epoch") != 97
        or checkpoint.get("size_bytes") != checkpoint_path.stat().st_size
        or metrics.get("sample_count") != 826
        or floor.get("valid") is not True
        or uncertainty.get("valid") is not True
        or patent.get("valid") is not True
        or patent.get("formal_paper_metric") is not False
        or strata.get("valid") is not True
        or strata.get("formal_paper_metric") is not False
        or motivation.get("state") != "awaiting_user_confirmation"
        or motivation.get("confirmed_option") is not None
        or qwen.get("state") != "awaiting_api_credentials"
        or qwen.get("api_credentials_available") is not False
    ):
        raise ValueError("paper-readiness v6 addendum contract is invalid")

    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "evidence_role": "additive_post_audit_paper_readiness_update",
        "historical_audits_modified": False,
        "readiness": {
            "internal_object_classification_evidence": "complete",
            "reliable_multimodal_object_method": "complete_internal_evidence",
            "joint_native_segment_head": "development_mechanism_evidence",
            "standardized_instance_strata": "posthoc_internal_diagnostic",
            "formal_full_image_qwen_v6_adaptation": (
                "awaiting_api_credentials"
            ),
            "full_patent_reproduction": False,
            "external_deployment_claim": False,
            "full_paper_motivation_confirmed": False,
        },
        "object_checkpoint_recovery": {
            "found": True,
            "selected_epoch": int(checkpoint["epoch"]),
            "selection_val_accuracy": float(
                checkpoint["selection_val_accuracy"]
            ),
            "checkpoint_binding": _binding(checkpoint_path),
            "test_metrics": {
                key: metrics[key]
                for key in (
                    "sample_count",
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                    "top3_accuracy",
                    "negative_log_likelihood",
                    "multiclass_brier_score",
                    "ece",
                )
            },
            "test_split_external_blind": False,
        },
        "post_audit_updates": [
            {
                "historical_statement": (
                    "Target-domain YOLO-seg remains separate from the MMFC "
                    "feature head."
                ),
                "current_status": "partially_superseded",
                "update": (
                    "v5/v6 now inject reliable multimodal conditioning into "
                    "the native YOLO11s-seg Segment-head path and pass "
                    "structural/intervention audits."
                ),
                "boundary": (
                    "The current joint-head result is single-seed "
                    "development/mechanism evidence and does not replace "
                    "the preregistered Qwen three-seed gate."
                ),
            },
            {
                "historical_statement": (
                    "Relative-size and bbox-overlap strata are not human "
                    "small-target or occlusion labels."
                ),
                "current_status": "partially_superseded",
                "update": (
                    "Visible-mask scale now uses standard COCO thresholds; "
                    "bbox overlap remains a potential crowding proxy."
                ),
                "boundary": (
                    "Only 11 small and 20 high-overlap targets are available; "
                    "there are still no human occlusion or amodal labels."
                ),
            },
        ],
        "uncertainty_status": {
            "aggregate_floor_collapse_mitigated": True,
            "v3_variance_to_mse_ratio": uncertainty["comparison"]["models"][
                "v3"
            ]["predicted_variance_to_mse_ratio"],
            "v3_sample_mean_spearman": uncertainty["comparison"]["models"][
                "v3"
            ]["sample_mean_spearman"],
            "absolute_scale_still_conservative": True,
            "physical_sensor_calibration_established": False,
        },
        "active_gates": [
            {
                "gate": "paper_motivation",
                "state": motivation["state"],
                "recommended_option": motivation["recommended_option"],
                "confirmed_option": motivation["confirmed_option"],
            },
            {
                "gate": "formal_full_image_qwen_v6_adaptation",
                "state": qwen["state"],
                "planned_seeds": qwen["planned_seeds"],
                "workers": qwen["training_workers"],
                "batch_size": qwen["training_batch_size"],
            },
            {
                "gate": "external_physical_validation",
                "state": "not_available",
                "requires": [
                    "synchronized_hardware_4x4_sensor_array",
                    "rgb_d_or_depth_input_if_patent_scope_is_retained",
                    "new_external_blind_test_set",
                ],
            },
        ],
        "claim_policy": {
            "allowed": [
                (
                    "The object-classification checkpoint exists and has an "
                    "826-sample internal test evaluation."
                ),
                (
                    "The v5/v6 joint-head extension provides audited "
                    "development and mechanism evidence."
                ),
                (
                    "COCO visible-mask scale strata are available as a "
                    "post-hoc internal diagnostic."
                ),
            ],
            "not_allowed": [
                "The internal test set is an external blind benchmark.",
                "The project fully reproduces the RGB-D hardware patent.",
                "The bbox-overlap proxy is a human occlusion annotation.",
                "Formal full-image Qwen v6 three-seed results are complete.",
            ],
        },
        "bindings": {
            name: _binding(path)
            for name, path in paths.items()
            if name != "object_best_checkpoint"
        },
    }
    _atomic_text(
        output,
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )

    test_metrics = result["object_checkpoint_recovery"]["test_metrics"]
    markdown = [
        "# v6后续论文就绪性增量附录",
        "",
        "- 本附录不修改历史 `PAPER_READINESS_AUDIT.json` 或 `FINAL_EVIDENCE_AUDIT.json`。",
        "- 内部对象分类证据：完成；联合原生Segment头：开发/机制证据；完整专利复现：否。",
        "- 正式整图Qwen v6三种子适配：等待API环境凭据；论文主线：等待用户确认A/B/C。",
        "",
        "## 对象检查点",
        "",
        (
            f"- `best.pt` 已找到，选择轮次为{checkpoint['epoch']}，"
            f"验证准确率为{checkpoint['selection_val_accuracy']:.4%}。"
        ),
        (
            f"- 826条内部测试：Accuracy {test_metrics['accuracy']:.2%}，"
            f"Macro-F1 {test_metrics['macro_f1']:.2%}，"
            f"Weighted-F1 {test_metrics['weighted_f1']:.2%}，"
            f"ECE {test_metrics['ece']:.2%}。"
        ),
        "- 该测试集已用于多轮内部比较，不是外部盲测。",
        "",
        "## 历史边界更新",
        "",
        (
            "- v5/v6现已把可靠多模态调制接入原生YOLO11s-seg Segment头，"
            "但仍是单种子开发与机制证据。"
        ),
        (
            "- 已按COCO可见掩膜面积阈值构建标准尺度分层；框重叠仍只是拥挤代理，"
            "小目标/高重叠仅11/20例。"
        ),
        (
            "- 方差下界贴底已由-3下界与方差先验缓解；潜空间方差相对排序改善，"
            "绝对量值仍明显偏保守，不能称为物理传感器标定。"
        ),
        "",
        "## 当前关口",
        "",
        "1. 用户确认论文动机A/B/C后，才能固定题目、摘要、贡献顺序和结论。",
        "2. 提供API环境变量后，执行已冻结的整图Qwen v6三种子适配协议。",
        "3. 若保留完整专利复现主张，仍需RGB-D、同步实体4x4传感器和新外部盲测。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return result


def main() -> None:
    args = parse_args()
    result = build_addendum(args.output, args.checkpoint)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "object_checkpoint_found": result[
                    "object_checkpoint_recovery"
                ]["found"],
                "readiness": result["readiness"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
