"""Write the seed-42 class-balance result report from locked artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from mmfc.training import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a class-balance candidate report"
    )
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--reference-val", type=Path, required=True)
    parser.add_argument("--candidate-val", type=Path, required=True)
    parser.add_argument("--reference-test", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument("--paired-comparison", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def _pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _metric_table(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[str]:
    specs = (
        ("accuracy", "Accuracy", "pct"),
        ("macro_precision", "Macro Precision", "pct"),
        ("macro_recall", "Macro Recall", "pct"),
        ("macro_f1", "Macro F1", "pct"),
        ("weighted_f1", "Weighted F1", "pct"),
        ("negative_log_likelihood", "NLL", "raw"),
        ("multiclass_brier_score", "Brier", "raw"),
        ("ece", "ECE", "pct"),
    )
    lines = [
        "| 指标 | v3 | 平衡候选 | 候选 - v3 |",
        "|---|---:|---:|---:|",
    ]
    for key, label, kind in specs:
        left = float(reference[key])
        right = float(candidate[key])
        if kind == "pct":
            lines.append(
                f"| {label} | {_pct(left)} | {_pct(right)} | {_pp(right-left)} |"
            )
        else:
            lines.append(
                f"| {label} | {left:.4f} | {right:.4f} | {right-left:+.4f} |"
            )
    return lines


def build_report(
    *,
    training_summary: Mapping[str, Any],
    reference_val: Mapping[str, Any],
    candidate_val: Mapping[str, Any],
    reference_test: Mapping[str, Any],
    candidate_test: Mapping[str, Any],
    paired: Mapping[str, Any],
    decision: Mapping[str, Any],
    checkpoint_audit: Mapping[str, Any],
    leakage_audit: Mapping[str, Any] | None = None,
) -> str:
    history = training_summary.get("history", [])
    if not history:
        raise ValueError("training summary has no history")
    final = history[-1]
    reference_val_metrics = reference_val["metrics"]
    candidate_val_metrics = candidate_val["metrics"]
    reference_test_metrics = reference_test["metrics"]
    candidate_test_metrics = candidate_test["metrics"]
    differences = paired["paired_bootstrap"]["differences"]
    outcome = paired["outcome_changes"]
    mcnemar = paired["mcnemar_exact"]
    passed = bool(decision["passed"])
    decision_text = (
        "通过首种子继续门槛，按锁定协议扩展种子 43/44。"
        if passed
        else "未通过首种子继续门槛，按锁定协议停止该候选。"
    )
    macro_precision_delta = (
        float(candidate_test_metrics["macro_precision"])
        - float(reference_test_metrics["macro_precision"])
    )
    macro_recall_delta = (
        float(candidate_test_metrics["macro_recall"])
        - float(reference_test_metrics["macro_recall"])
    )

    lines = [
        "# v3 类别平衡候选正式实验",
        "",
        "生成日期：2026-07-27",
        "",
        "## 实验身份",
        "",
        "- 底座：`full_cross_modal_quality`（v3）；",
        "- 唯一行为改动：sqrt-inverse 类别权重仅用于 CE，Focal 不加权；",
        "- 最佳检查点选择：验证 Macro F1；",
        f"- 训练种子：42，完成轮数：{training_summary['completed_epochs']}；",
        (
            f"- 最佳 epoch：{training_summary['best_checkpoint_epoch']}，"
            f"验证 Macro F1 {_pct(training_summary['best_selection_score'])}，"
            f"同期 Accuracy {_pct(training_summary['best_checkpoint_val_accuracy'])}；"
        ),
        (
            f"- 最后一轮训练/验证 Accuracy："
            f"{_pct(float(final['train_accuracy']))}/"
            f"{_pct(float(final['val_accuracy']))}，验证 Macro F1 "
            f"{_pct(float(final['val_macro_f1']))}；"
        ),
        f"- 配置指纹：`{training_summary['config_fingerprint']}`；",
        (
            f"- 检查点审计：`valid={str(bool(checkpoint_audit['valid'])).lower()}`。"
        ),
        "",
        "## 验证集",
        "",
        *_metric_table(reference_val_metrics, candidate_val_metrics),
        "",
        "验证集用于各自检查点选择。v3 按 Accuracy 选择，平衡候选按预注册的 "
        "Macro F1 选择，因此这里比较的是两个协议最终选中的检查点，而不是事后为 "
        "v3 重新挑选 Macro F1 轮次。",
        "",
        "## 固定留出测试",
        "",
        *_metric_table(reference_test_metrics, candidate_test_metrics),
        "",
        (
            "平衡候选的 Macro Recall "
            f"{_pp(macro_recall_delta)}，但 Macro Precision "
            f"{_pp(macro_precision_delta)}，最终 Macro F1 未改善。该结果更接近"
            "决策边界向少数类召回移动，而不是整体分类质量提升。"
        ),
        "",
        "## 配对统计",
        "",
        (
            f"候选新纠正 {outcome['candidate_only_correct']} 条，v3 独有正确 "
            f"{outcome['reference_only_correct']} 条；精确 McNemar "
            f"`p={float(mcnemar['p_value']):.6g}`。"
        ),
        "",
        (
            "源图聚类配对 bootstrap 的 Accuracy 差值 95% 区间为 "
            f"[{_pp(float(differences['accuracy']['lower']))},"
            f"{_pp(float(differences['accuracy']['upper']))}]，Macro F1 为 "
            f"[{_pp(float(differences['macro_f1']['lower']))},"
            f"{_pp(float(differences['macro_f1']['upper']))}]。"
        ),
        "",
        "区间若跨 0，只能表述为点估计满足或不满足预注册门槛，不能声称统计显著提升。",
        "",
        "## 预注册继续判定",
        "",
        "| 条件 | v3 | 候选 | 差值 | 结果 |",
        "|---|---:|---:|---:|---|",
    ]
    criterion_labels = {
        "validation_macro_f1_non_decrease": "验证 Macro F1 不下降",
        "test_macro_f1_non_decrease": "测试 Macro F1 不下降",
        "test_accuracy_drop_within_limit": "测试 Accuracy 下降不超过 1 pp",
    }
    for key, item in decision["criteria"].items():
        lines.append(
            "| {label} | {reference} | {candidate} | {delta} | {result} |".format(
                label=criterion_labels[key],
                reference=_pct(float(item["reference"])),
                candidate=_pct(float(item["candidate"])),
                delta=_pp(float(item["delta"])),
                result="通过" if item["passed"] else "未通过",
            )
        )
    lines.extend(
        [
            "",
            f"结论：**{decision_text}**",
            "",
            "该判定只控制是否继续训练种子，不把同一固定测试集变成新的外部确认集。",
            "多种子结果即使稳定，也仍需遵守 `EVALUATION_GOVERNANCE.md` 的测试访问",
            "边界。",
        ]
    )

    reference_per_class = reference_test.get("per_class")
    candidate_per_class = candidate_test.get("per_class")
    if isinstance(reference_per_class, list) and isinstance(candidate_per_class, list):
        reference_by_label = {
            str(item["label"]): item for item in reference_per_class
        }
        class_deltas = sorted(
            (
                (
                    str(item["label"]),
                    int(item["support"]),
                    float(item["f1"])
                    - float(reference_by_label[str(item["label"])]["f1"]),
                )
                for item in candidate_per_class
                if str(item["label"]) in reference_by_label
            ),
            key=lambda item: item[2],
        )
        if len(class_deltas) >= 6:
            losses = "、".join(
                f"`{label}` {_pp(delta)}（support={support}）"
                for label, support, delta in class_deltas[:3]
            )
            gains = "、".join(
                f"`{label}` {_pp(delta)}（support={support}）"
                for label, support, delta in reversed(class_deltas[-3:])
            )
            lines.extend(
                [
                    "",
                    "## 逐类观察",
                    "",
                    f"点估计增幅最大的三类为 {gains}；降幅最大的三类为 {losses}。",
                    "其中多类 support 很小，这些差值仅用于描述错误迁移，不作为逐类"
                    "显著提升或后续调权依据。",
                ]
            )

    if leakage_audit is not None:
        sensitivity = (
            leakage_audit.get("prediction_sensitivity", {}).get("balanced")
        )
        if isinstance(sensitivity, Mapping):
            all_metrics = sensitivity["all"]
            excluded_metrics = sensitivity["after_high_confidence_exclusion"]
            lines.extend(
                [
                    "",
                    "## 近重复剔除敏感性",
                    "",
                    (
                        f"保守剔除 {sensitivity['excluded_source_count']} 张高置信近重复"
                        f"源图涉及的 {sensitivity['excluded_sample_count']} 个测试对象后，"
                        f"Accuracy 从 {_pct(float(all_metrics['accuracy']))} 变为 "
                        f"{_pct(float(excluded_metrics['accuracy']))}，Macro F1 从 "
                        f"{_pct(float(all_metrics['macro_f1']))} 变为 "
                        f"{_pct(float(excluded_metrics['macro_f1']))}。负结果结论不变。"
                    ),
                ]
            )
        duplicate_sensitivity = leakage_audit.get(
            "exact_object_duplicate_prediction_sensitivity", {}
        ).get("balanced")
        if isinstance(duplicate_sensitivity, Mapping):
            all_metrics = duplicate_sensitivity["all"]
            deduplicated_metrics = duplicate_sensitivity[
                "after_exact_object_duplicate_exclusion"
            ]
            lines.extend(
                [
                    "",
                    "## 完全重复对象敏感性",
                    "",
                    (
                        f"按确定性 sample_id 规则移除 "
                        f"{duplicate_sensitivity['excluded_sample_count']} 条"
                        "完全重复测试对象后，Accuracy 从 "
                        f"{_pct(float(all_metrics['accuracy']))} 变为 "
                        f"{_pct(float(deduplicated_metrics['accuracy']))}，"
                        "Macro F1 从 "
                        f"{_pct(float(all_metrics['macro_f1']))} 变为 "
                        f"{_pct(float(deduplicated_metrics['macro_f1']))}。"
                        "负结果结论不变。"
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `RUN_MANIFEST.json`：配置、数据、源码和初始权重哈希；",
            "- `CHECKPOINT_AUDIT.json`：best/last 角色、轮次、指纹和 schema；",
            "- `val/evaluation.json` 与 `test/evaluation.json`；",
            "- `PAIRED_COMPARISON.json`；",
            "- `CONTINUATION_DECISION.json`；",
            "- `training_curves.png`；",
            "- `../class_performance_v2b_v3_balanced.png`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = {
        name: resolve_project_path(getattr(args, name))
        for name in (
            "training_summary",
            "reference_val",
            "candidate_val",
            "reference_test",
            "candidate_test",
            "paired_comparison",
            "decision",
            "checkpoint_audit",
        )
    }
    leakage_audit_path = (
        resolve_project_path(args.leakage_audit)
        if args.leakage_audit is not None
        else None
    )
    report = build_report(
        training_summary=_load(paths["training_summary"]),
        reference_val=_load(paths["reference_val"]),
        candidate_val=_load(paths["candidate_val"]),
        reference_test=_load(paths["reference_test"]),
        candidate_test=_load(paths["candidate_test"]),
        paired=_load(paths["paired_comparison"]),
        decision=_load(paths["decision"]),
        checkpoint_audit=_load(paths["checkpoint_audit"]),
        leakage_audit=(
            _load(leakage_audit_path) if leakage_audit_path is not None else None
        ),
    )
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    inputs = {key: str(value) for key, value in paths.items()}
    if leakage_audit_path is not None:
        inputs["leakage_audit"] = str(leakage_audit_path)
    print(
        json.dumps(
            {"output": str(output), "inputs": inputs},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
