"""Build the post-queue evidence bundle for the confirmed Plan A manuscript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from audit_recent_robust_baselines import (
    AMENDED_METHOD_TRAINING_SEEDS,
    PRIMARY_PAIRED_CONDITIONS,
)
from evaluate_unseen_sensor_conditions import CONDITIONS
from mmfc.model import PROJECT_ROOT
from prepare_recent_robust_baselines import METHODS, SEEDS


REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_COMPONENT_OUTCOME = (
    REPORT_ROOT
    / "confirmatory-ablations"
    / "v3-loss-components"
    / "OUTCOME.json"
)
DEFAULT_UNSEEN_OUTCOME = (
    REPORT_ROOT / "unseen-sensor-conditions-v1" / "OUTCOME.json"
)
DEFAULT_BASELINE_OUTCOME = (
    REPORT_ROOT / "recent-robust-baselines" / "OUTCOME.json"
)
DEFAULT_OUTPUT_DIR = REPORT_ROOT / "plan-a-confirmatory-outcome"
COMPONENT_VARIANTS = (
    "v3_no_grid_reconstruction",
    "v3_no_ranking",
)
COMPONENT_LABELS = {
    "v3_no_grid_reconstruction": "无网格重建监督",
    "v3_no_ranking": "无可靠性排序监督",
}
METHOD_LABELS = {
    "qmf": "QMF适配",
    "tmc": "TMC适配",
}
CONDITION_LABELS = {
    "clean": "干净观测",
    "noise_030_out_of_range": "范围外噪声0.30",
    "bias_030_out_of_range": "范围外偏移0.30",
    "impulse_025": "脉冲故障25%",
    "gamma_2": "非线性响应gamma=2",
    "spatial_permutation": "空间位置打乱",
    "foreign_sample": "异物样本替换",
    "stuck_midpoint": "中值卡死",
    "clean_false_missing": "正常观测误报缺失",
    "missing_false_available": "缺失观测误报可用",
    "missing_correct": "缺失且正确标记",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Plan A confirmatory manuscript addendum"
    )
    parser.add_argument(
        "--component-outcome",
        type=Path,
        default=DEFAULT_COMPONENT_OUTCOME,
    )
    parser.add_argument(
        "--unseen-outcome",
        type=Path,
        default=DEFAULT_UNSEEN_OUTCOME,
    )
    parser.add_argument(
        "--baseline-outcome",
        type=Path,
        default=DEFAULT_BASELINE_OUTCOME,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _optional_finite(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _finite(value, label)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != len(SEEDS):
        raise ValueError("three matched training seeds are required")
    return mean(values), stdev(values)


def _variable_mean_std(
    values: Sequence[float],
) -> tuple[float, float | None]:
    if not values:
        raise ValueError("at least one training seed is required")
    return mean(values), stdev(values) if len(values) > 1 else None


def _baseline_method_seeds(
    baseline: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    raw = baseline.get("method_training_seeds")
    if not isinstance(raw, Mapping) or set(raw) != set(METHODS):
        raise ValueError("recent-baseline method/seed mapping is missing")
    result = {
        method: tuple(raw[method])
        for method in METHODS
        if isinstance(raw.get(method), list)
    }
    if result != AMENDED_METHOD_TRAINING_SEEDS:
        raise ValueError("recent-baseline amended seed scope mismatch")
    return result


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_sources(
    component: Mapping[str, Any],
    unseen: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> None:
    if (
        component.get("valid") is not True
        or component.get("state") != "complete"
        or set(component.get("variants", {})) != set(COMPONENT_VARIANTS)
    ):
        raise ValueError("component-ablation outcome is not complete and valid")
    if (
        unseen.get("valid") is not True
        or unseen.get("state") != "complete"
        or unseen.get("model_seeds") != list(SEEDS)
        or set(unseen.get("conditions", {})) != set(CONDITIONS)
    ):
        raise ValueError("unseen-condition outcome is not complete and valid")
    unseen_rows = unseen.get("summary")
    if (
        not isinstance(unseen_rows, list)
        or {row.get("condition") for row in unseen_rows} != set(CONDITIONS)
    ):
        raise ValueError("unseen-condition summary identity mismatch")
    baseline_method_seeds = _baseline_method_seeds(baseline)
    if (
        baseline.get("valid") is not True
        or baseline.get("state") != "complete"
        or baseline.get("methods") != list(METHODS)
        or baseline.get("training_seeds") != list(SEEDS)
        or baseline.get("run_count") != sum(
            len(values) for values in baseline_method_seeds.values()
        )
        or baseline.get("primary_paired_conditions")
        != list(PRIMARY_PAIRED_CONDITIONS)
    ):
        raise ValueError("recent-baseline outcome is not complete and valid")
    baseline_rows = baseline.get("summary")
    expected_pairs = {
        (method, condition)
        for method in METHODS
        for condition in CONDITIONS
    }
    if (
        not isinstance(baseline_rows, list)
        or {
            (row.get("method"), row.get("condition"))
            for row in baseline_rows
        }
        != expected_pairs
    ):
        raise ValueError("recent-baseline summary identity mismatch")


def _component_points(
    item: Mapping[str, Any],
    variant: str,
    section: str,
    metric: str,
) -> list[float]:
    runs = item.get("runs")
    expected = {str(seed) for seed in SEEDS}
    if not isinstance(runs, Mapping) or set(runs) != expected:
        raise ValueError(f"{variant} component seed identity mismatch")
    return [
        _finite(
            runs[str(seed)].get(section, {}).get(metric),
            f"{variant} seed{seed} {section} {metric}",
        )
        for seed in SEEDS
    ]


def _require_matching_mean(
    reported: object,
    computed: float,
    label: str,
) -> None:
    value = _finite(reported, label)
    if not math.isclose(value, computed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} does not match the three seed runs")


def _component_rows(component: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    variants = component["variants"]
    for variant in COMPONENT_VARIANTS:
        item = variants[variant]
        directions = item["full_minus_ablation_seed_directions"]
        ablation_accuracy = _component_points(
            item, variant, "test_metrics", "accuracy"
        )
        ablation_macro_f1 = _component_points(
            item, variant, "test_metrics", "macro_f1"
        )
        effect_accuracy = _component_points(
            item, variant, "full_minus_ablation", "accuracy"
        )
        effect_macro_f1 = _component_points(
            item, variant, "full_minus_ablation", "macro_f1"
        )
        accuracy_mean, accuracy_std = _mean_std(ablation_accuracy)
        macro_f1_mean, macro_f1_std = _mean_std(ablation_macro_f1)
        effect_accuracy_mean, effect_accuracy_std = _mean_std(effect_accuracy)
        effect_macro_f1_mean, effect_macro_f1_std = _mean_std(effect_macro_f1)
        _require_matching_mean(
            item["ablation_test_mean"]["accuracy"],
            accuracy_mean,
            f"{variant} aggregate accuracy",
        )
        _require_matching_mean(
            item["ablation_test_mean"]["macro_f1"],
            macro_f1_mean,
            f"{variant} aggregate macro_f1",
        )
        _require_matching_mean(
            item["full_minus_ablation_mean"]["accuracy"],
            effect_accuracy_mean,
            f"{variant} aggregate accuracy effect",
        )
        _require_matching_mean(
            item["full_minus_ablation_mean"]["macro_f1"],
            effect_macro_f1_mean,
            f"{variant} aggregate macro_f1 effect",
        )
        accuracy_positive = sum(value > 0.0 for value in effect_accuracy)
        macro_f1_positive = sum(value > 0.0 for value in effect_macro_f1)
        if (
            directions.get("accuracy_positive") != accuracy_positive
            or directions.get("macro_f1_positive") != macro_f1_positive
        ):
            raise ValueError(f"{variant} component direction count mismatch")
        rows.append(
            {
                "variant": variant,
                "label_zh": COMPONENT_LABELS[variant],
                "ablation_accuracy": accuracy_mean,
                "ablation_accuracy_sample_std": accuracy_std,
                "ablation_macro_f1": macro_f1_mean,
                "ablation_macro_f1_sample_std": macro_f1_std,
                "full_minus_ablation_accuracy_pp": 100.0
                * effect_accuracy_mean,
                "full_minus_ablation_accuracy_pp_sample_std": 100.0
                * effect_accuracy_std,
                "full_minus_ablation_macro_f1_pp": 100.0
                * effect_macro_f1_mean,
                "full_minus_ablation_macro_f1_pp_sample_std": 100.0
                * effect_macro_f1_std,
                "accuracy_positive_seed_count": accuracy_positive,
                "macro_f1_positive_seed_count": macro_f1_positive,
            }
        )
    return rows


def _unseen_rows(unseen: Mapping[str, Any]) -> list[dict[str, Any]]:
    indexed = {row["condition"]: row for row in unseen["summary"]}
    rows = []
    for condition, contract in CONDITIONS.items():
        item = indexed[condition]
        rows.append(
            {
                "condition": condition,
                "label_zh": CONDITION_LABELS[condition],
                "category": contract["category"],
                "accuracy_mean": _finite(
                    item["accuracy_mean"], f"{condition} accuracy"
                ),
                "accuracy_sample_std": _finite(
                    item["accuracy_sample_std"], f"{condition} accuracy std"
                ),
                "macro_f1_mean": _finite(
                    item["macro_f1_mean"], f"{condition} macro_f1"
                ),
                "macro_f1_sample_std": _finite(
                    item["macro_f1_sample_std"],
                    f"{condition} macro_f1 std",
                ),
                "condition_minus_clean_accuracy_pp_mean": _finite(
                    item["condition_minus_clean_accuracy_pp_mean"],
                    f"{condition} accuracy effect",
                ),
                "condition_minus_clean_macro_f1_pp_mean": _finite(
                    item["condition_minus_clean_macro_f1_pp_mean"],
                    f"{condition} macro_f1 effect",
                ),
                "reported_availability_mean": _finite(
                    item["reported_availability_mean"],
                    f"{condition} availability",
                ),
                "true_quality_mean": _finite(
                    item["true_quality_mean"], f"{condition} quality"
                ),
                "predicted_reliability_mean": _finite(
                    item["predicted_reliability_mean"],
                    f"{condition} reliability",
                ),
            }
        )
    return rows


def _baseline_rows(baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    method_seeds = _baseline_method_seeds(baseline)
    indexed = {
        (row["method"], row["condition"]): row
        for row in baseline["summary"]
    }
    rows = []
    for method in METHODS:
        for condition in CONDITIONS:
            item = indexed[(method, condition)]
            rows.append(
                {
                    "method": method,
                    "method_label_zh": METHOD_LABELS[method],
                    "condition": condition,
                    "condition_label_zh": CONDITION_LABELS[condition],
                    "category": item["category"],
                    "training_seeds": list(method_seeds[method]),
                    "seed_count": len(method_seeds[method]),
                    "evidence_level": item["evidence_level"],
                    "accuracy_mean": _finite(
                        item["accuracy_mean"],
                        f"{method}/{condition} accuracy",
                    ),
                    "accuracy_sample_std": _optional_finite(
                        item["accuracy_sample_std"],
                        f"{method}/{condition} accuracy std",
                    ),
                    "macro_f1_mean": _finite(
                        item["macro_f1_mean"],
                        f"{method}/{condition} macro_f1",
                    ),
                    "macro_f1_sample_std": _optional_finite(
                        item["macro_f1_sample_std"],
                        f"{method}/{condition} macro_f1 std",
                    ),
                    "sensor_branch_weight_mean": _finite(
                        item["sensor_branch_weight_mean"],
                        f"{method}/{condition} sensor weight",
                    ),
                }
            )
    return rows


def _paired_rows(baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    method_seeds = _baseline_method_seeds(baseline)
    comparisons = baseline.get("paired_candidate_minus_v3", {})
    rows = []
    for method in METHODS:
        for condition in PRIMARY_PAIRED_CONDITIONS:
            accuracy_points = []
            macro_f1_points = []
            per_seed = []
            for seed in method_seeds[method]:
                key = f"{method}_seed{seed}_{condition}"
                comparison = comparisons.get(key)
                if not isinstance(comparison, Mapping):
                    raise ValueError(f"missing paired comparison: {key}")
                if (
                    comparison.get("repeats") != 2000
                    or comparison.get("cluster_count") != 570
                    or comparison.get("sample_count") != 826
                ):
                    raise ValueError(f"paired comparison contract mismatch: {key}")
                differences = comparison.get("differences", {})
                accuracy = differences.get("accuracy", {})
                macro_f1 = differences.get("macro_f1", {})
                accuracy_point = _finite(
                    accuracy.get("point"), f"{key} accuracy point"
                )
                macro_f1_point = _finite(
                    macro_f1.get("point"), f"{key} macro_f1 point"
                )
                accuracy_points.append(accuracy_point)
                macro_f1_points.append(macro_f1_point)
                per_seed.append(
                    {
                        "seed": seed,
                        "accuracy_pp": 100.0 * accuracy_point,
                        "accuracy_ci_lower_pp": 100.0
                        * _finite(
                            accuracy.get("lower"),
                            f"{key} accuracy lower",
                        ),
                        "accuracy_ci_upper_pp": 100.0
                        * _finite(
                            accuracy.get("upper"),
                            f"{key} accuracy upper",
                        ),
                        "macro_f1_pp": 100.0 * macro_f1_point,
                        "macro_f1_ci_lower_pp": 100.0
                        * _finite(
                            macro_f1.get("lower"),
                            f"{key} macro_f1 lower",
                        ),
                        "macro_f1_ci_upper_pp": 100.0
                        * _finite(
                            macro_f1.get("upper"),
                            f"{key} macro_f1 upper",
                        ),
                    }
                )
            accuracy_mean, accuracy_std = _variable_mean_std(
                accuracy_points
            )
            macro_f1_mean, macro_f1_std = _variable_mean_std(
                macro_f1_points
            )
            rows.append(
                {
                    "method": method,
                    "method_label_zh": METHOD_LABELS[method],
                    "condition": condition,
                    "condition_label_zh": CONDITION_LABELS[condition],
                    "training_seeds": list(method_seeds[method]),
                    "seed_count": len(method_seeds[method]),
                    "evidence_level": (
                        "confirmatory_multiseed"
                        if len(method_seeds[method]) > 1
                        else "reference_single_seed"
                    ),
                    "candidate_minus_v3_accuracy_pp_mean": 100.0
                    * accuracy_mean,
                    "candidate_minus_v3_accuracy_pp_sample_std": (
                        100.0 * accuracy_std
                        if accuracy_std is not None
                        else None
                    ),
                    "candidate_minus_v3_macro_f1_pp_mean": 100.0
                    * macro_f1_mean,
                    "candidate_minus_v3_macro_f1_pp_sample_std": (
                        100.0 * macro_f1_std
                        if macro_f1_std is not None
                        else None
                    ),
                    "accuracy_candidate_positive_seed_count": sum(
                        point > 0.0 for point in accuracy_points
                    ),
                    "macro_f1_candidate_positive_seed_count": sum(
                        point > 0.0 for point in macro_f1_points
                    ),
                    "per_seed_cluster_bootstrap": per_seed,
                }
            )
    return rows


def compile_confirmatory_bundle(
    component: Mapping[str, Any],
    unseen: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all queued outcomes and derive paper-facing neutral summaries."""

    _validate_sources(component, unseen, baseline)
    component_rows = _component_rows(component)
    unseen_rows = _unseen_rows(unseen)
    baseline_rows = _baseline_rows(baseline)
    paired_rows = _paired_rows(baseline)
    non_reference = [
        row for row in unseen_rows if row["condition"] != "clean"
    ]
    worst_unseen = min(
        non_reference,
        key=lambda row: row["condition_minus_clean_macro_f1_pp_mean"],
    )
    clean_paired = [
        row for row in paired_rows if row["condition"] == "clean"
    ]
    return {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "plan_a_post_queue_confirmatory_manuscript_evidence",
        "training_seeds": list(SEEDS),
        "baseline_method_training_seeds": {
            method: list(values)
            for method, values in _baseline_method_seeds(baseline).items()
        },
        "component_ablations": component_rows,
        "unseen_sensor_conditions": unseen_rows,
        "recent_robust_baselines": baseline_rows,
        "paired_candidate_minus_v3": paired_rows,
        "decision_summary": {
            "component_macro_f1_positive_seed_counts": {
                row["variant"]: row["macro_f1_positive_seed_count"]
                for row in component_rows
            },
            "worst_unseen_macro_f1_condition": worst_unseen["condition"],
            "worst_unseen_macro_f1_effect_pp": worst_unseen[
                "condition_minus_clean_macro_f1_pp_mean"
            ],
            "clean_candidate_minus_v3_macro_f1_pp": {
                row["method"]: row[
                    "candidate_minus_v3_macro_f1_pp_mean"
                ]
                for row in clean_paired
            },
            "clean_candidate_macro_f1_positive_seed_counts": {
                row["method"]: row[
                    "macro_f1_candidate_positive_seed_count"
                ]
                for row in clean_paired
            },
            "clean_candidate_seed_counts": {
                row["method"]: row["seed_count"] for row in clean_paired
            },
        },
        "claim_policy": {
            "allowed": [
                (
                    "报告两个辅助监督组件的三种子独立重训均值、样本标准差和"
                    "full-v3减消融方向计数。"
                ),
                (
                    "报告11种确定性软件条件下的三种子均值、样本标准差"
                    "及相对clean变化。"
                ),
                (
                    "报告QMF同输入适配的三种子均值/样本标准差，以及TMC "
                    "seed42的单种子参考点估计和源图簇配对区间；TMC不报告"
                    "样本标准差或重复性结论。"
                ),
            ],
            "not_allowed": [
                "把内部固定测试集结果称为外部盲测或部署泛化。",
                "把软件注入故障称为真实硬件故障验证。",
                "把候选类别文本原型称为独立采集的逐样本文本模态。",
                "把QMF/TMC适配称为原作者数据、编码器和训练流水线的逐字复现。",
                "从三种子均值直接构造未经预注册的跨种子显著性结论。",
                "把TMC单种子结果写成多次运行、均值±标准差或重复性证据。",
            ],
        },
        "errors": [],
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _pp(value: float) -> str:
    return f"{value:+.2f}"


def _pct_with_sd(value: float, sample_std: float | None) -> str:
    if sample_std is None:
        return f"{_pct(value)} (n=1; SD=N/A)"
    return f"{_pct(value)} ± {100.0 * sample_std:.2f}"


def _pp_with_sd(value: float, sample_std: float | None) -> str:
    if sample_std is None:
        return f"{_pp(value)} (n=1; SD=N/A)"
    return f"{_pp(value)} ± {sample_std:.2f}"


def _latex_pct_with_sd(value: float, sample_std: float | None) -> str:
    if sample_std is None:
        return f"{_pct(value)} (n=1)"
    return f"{_pct(value)}$\\pm${100.0 * sample_std:.2f}"


def _markdown(bundle: Mapping[str, Any]) -> str:
    component_rows = bundle["component_ablations"]
    unseen_rows = bundle["unseen_sensor_conditions"]
    baseline_rows = [
        row
        for row in bundle["recent_robust_baselines"]
        if row["condition"] in PRIMARY_PAIRED_CONDITIONS
    ]
    paired_rows = bundle["paired_candidate_minus_v3"]
    lines = [
        "# 方案A确认性实验结案与论文增量",
        "",
        "- 组件与未见条件队列完成；直接基线按用户指令形成透明范围修订。",
        "- 所有结果限定于复用的内部固定测试集和图像条件soft sidecar。",
        "- 表中“候选-v3”正值表示QMF/TMC适配高于完整v3，负值表示完整v3更高。",
        "- QMF为n=3；TMC仅seed42、n=1，SD不适用，不能称为重复实验。",
        "",
        "## 辅助监督组件",
        "",
        "| 变体 | Accuracy (%, 均值±SD) | Macro F1 (%, 均值±SD) | full-v3减变体 Accuracy (pp, 均值±SD) | full-v3减变体 Macro F1 (pp, 均值±SD) | Acc/F1正向种子 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {row['label_zh']} | {_pct(row['ablation_accuracy'])} ± "
            f"{100.0 * row['ablation_accuracy_sample_std']:.2f} | "
            f"{_pct(row['ablation_macro_f1'])} ± "
            f"{100.0 * row['ablation_macro_f1_sample_std']:.2f} | "
            f"{_pp(row['full_minus_ablation_accuracy_pp'])} ± "
            f"{row['full_minus_ablation_accuracy_pp_sample_std']:.2f} | "
            f"{_pp(row['full_minus_ablation_macro_f1_pp'])} ± "
            f"{row['full_minus_ablation_macro_f1_pp_sample_std']:.2f} | "
            f"{row['accuracy_positive_seed_count']}/3 / "
            f"{row['macro_f1_positive_seed_count']}/3 |"
        )
        for row in component_rows
    )
    lines.extend(
        [
            "",
            "## 未见传感器与availability条件",
            "",
            "| 条件 | Accuracy (%) | Macro F1 (%) | Macro F1相对clean (pp) | 预测可靠性 | 真实质量 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        (
            f"| {row['label_zh']} | "
            f"{_pct(row['accuracy_mean'])} ± "
            f"{100.0 * row['accuracy_sample_std']:.2f} | "
            f"{_pct(row['macro_f1_mean'])} ± "
            f"{100.0 * row['macro_f1_sample_std']:.2f} | "
            f"{_pp(row['condition_minus_clean_macro_f1_pp_mean'])} | "
            f"{row['predicted_reliability_mean']:.3f} | "
            f"{row['true_quality_mean']:.3f} |"
        )
        for row in unseen_rows
    )
    lines.extend(
        [
            "",
            "## 近期直接基线",
            "",
            "| 方法 | 条件 | Accuracy (%) | Macro F1 (%) | 传感器分支权重 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        (
            f"| {row['method_label_zh']} | {row['condition_label_zh']} | "
            f"{_pct_with_sd(row['accuracy_mean'], row['accuracy_sample_std'])} | "
            f"{_pct_with_sd(row['macro_f1_mean'], row['macro_f1_sample_std'])} | "
            f"{row['sensor_branch_weight_mean']:.3f} |"
        )
        for row in baseline_rows
    )
    lines.extend(
        [
            "",
            "## 与完整v3的配对方向",
            "",
            "| 方法 | 条件 | 候选-v3 Accuracy (pp) | 候选-v3 Macro F1 (pp) | Macro F1候选正向种子 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        (
            f"| {row['method_label_zh']} | {row['condition_label_zh']} | "
            f"{_pp_with_sd(row['candidate_minus_v3_accuracy_pp_mean'], row['candidate_minus_v3_accuracy_pp_sample_std'])} | "
            f"{_pp_with_sd(row['candidate_minus_v3_macro_f1_pp_mean'], row['candidate_minus_v3_macro_f1_pp_sample_std'])} | "
            f"{row['macro_f1_candidate_positive_seed_count']}/"
            f"{row['seed_count']} |"
        )
        for row in paired_rows
    )
    lines.extend(
        [
            "",
            "## 论文表述边界",
            "",
            *[
                f"- 可写：{item}"
                for item in bundle["claim_policy"]["allowed"]
            ],
            *[
                f"- 禁止：{item}"
                for item in bundle["claim_policy"]["not_allowed"]
            ],
            "",
        ]
    )
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _latex(bundle: Mapping[str, Any]) -> str:
    component_rows = bundle["component_ablations"]
    unseen_rows = bundle["unseen_sensor_conditions"]
    baseline_rows = [
        row
        for row in bundle["recent_robust_baselines"]
        if row["condition"] == "clean"
    ]
    paired_rows = [
        row
        for row in bundle["paired_candidate_minus_v3"]
        if row["condition"] == "clean"
    ]
    worst = bundle["decision_summary"]
    component_directions = "；".join(
        (
            f"{row['label_zh']}为{row['accuracy_positive_seed_count']}/3和"
            f"{row['macro_f1_positive_seed_count']}/3"
        )
        for row in component_rows
    )
    lines = [
        "% Auto-generated from frozen Plan A confirmatory outcomes.",
        r"\subsection{组件级归因、未见退化与直接基线}",
        "",
        (
            "在既有三种子复合可靠性消融之外，本文进一步分别移除网格重建监督和"
            "可靠性排序监督，并保持数据、退化增强、编码器、优化器和100轮日程"
            "匹配。表\\ref{tab:plan-a-components}中的差异定义为完整v3减去对应"
            "独立重训变体，方向计数用于描述训练随机性，不替代跨种子显著性检验。"
        ),
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{两个辅助监督组件的三种子独立重训结果（均值$\pm$样本标准差）。}",
        r"\label{tab:plan-a-components}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"变体 & Accuracy & Macro F1 & $\Delta$Acc (pp) & $\Delta$F1 (pp) \\",
        r"\midrule",
    ]
    lines.extend(
        (
            f"{_latex_escape(row['label_zh'])} & "
            f"{_pct(row['ablation_accuracy'])}$\\pm$"
            f"{100.0 * row['ablation_accuracy_sample_std']:.2f} & "
            f"{_pct(row['ablation_macro_f1'])}$\\pm$"
            f"{100.0 * row['ablation_macro_f1_sample_std']:.2f} & "
            f"{_pp(row['full_minus_ablation_accuracy_pp'])}$\\pm$"
            f"{row['full_minus_ablation_accuracy_pp_sample_std']:.2f} & "
            f"{_pp(row['full_minus_ablation_macro_f1_pp'])}$\\pm$"
            f"{row['full_minus_ablation_macro_f1_pp_sample_std']:.2f} \\\\"
        )
        for row in component_rows
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            (
                "按完整v3减消融计算，Accuracy和Macro F1的正向种子数分别为："
                f"{component_directions}。"
            ),
            "",
            (
                "为检查训练注入器之外的响应，本文在固定内部测试上增加范围外强度、"
                "脉冲、非线性、空间置换、异物替换、中值卡死及availability误报。"
                "这11种条件是确定性软件干预，不是实体传感器故障。三种子中Macro "
                "F1相对clean下降最大的条件为"
                f"\\texttt{{{_latex_escape(worst['worst_unseen_macro_f1_condition'])}}}，"
                f"平均变化为{_pp(worst['worst_unseen_macro_f1_effect_pp'])}个百分点。"
            ),
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{完整v3在未见软件条件下的三种子结果。}",
            r"\label{tab:plan-a-unseen}",
            r"\scriptsize",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"条件 & Accuracy & Macro F1 & F1相对clean (pp) \\",
            r"\midrule",
        ]
    )
    lines.extend(
        (
            f"{_latex_escape(row['label_zh'])} & "
            f"{_pct(row['accuracy_mean'])}$\\pm$"
            f"{100.0 * row['accuracy_sample_std']:.2f} & "
            f"{_pct(row['macro_f1_mean'])}$\\pm$"
            f"{100.0 * row['macro_f1_sample_std']:.2f} & "
            f"{_pp(row['condition_minus_clean_macro_f1_pp_mean'])} \\\\"
        )
        for row in unseen_rows
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            (
                "直接方法对照采用QMF的质量感知动态决策融合"
                "\\cite{zhang2023quality}与TMC的Dirichlet证据组合"
                "\\cite{han2021trusted}。两者均复用本项目的视觉编码器、候选类别"
                "文本原型、4$\\times$4 sidecar和划分，仅替换融合及目标函数；"
                "因此属于同输入公式适配，不是原作者数据与编码器的逐字复现。"
                "QMF报告三个种子的均值与样本标准差；按计算成本范围修订，TMC"
                "只保留完整seed42作为单种子参考（$n=1$），不计算样本标准差，"
                "也不作重复性声明。"
            ),
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{近期直接基线在clean条件下的混合种子范围结果。}",
            r"\label{tab:plan-a-baselines}",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"方法 & Accuracy & Macro F1 & 候选-v3 Acc (pp) & 候选-v3 F1 (pp) \\",
            r"\midrule",
        ]
    )
    paired_index = {row["method"]: row for row in paired_rows}
    for row in baseline_rows:
        paired = paired_index[row["method"]]
        lines.append(
            f"{_latex_escape(row['method_label_zh'])} & "
            f"{_latex_pct_with_sd(row['accuracy_mean'], row['accuracy_sample_std'])} & "
            f"{_latex_pct_with_sd(row['macro_f1_mean'], row['macro_f1_sample_std'])} & "
            f"{_pp(paired['candidate_minus_v3_accuracy_pp_mean'])} & "
            f"{_pp(paired['candidate_minus_v3_macro_f1_pp_mean'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            (
                "所有比较继续受同一证据边界约束：固定测试集已在开发中复用，"
                "候选文本不是独立逐样本观测，软件故障不能外推为硬件鲁棒性，"
                "QMF三种子均值不构成未经注册的总体显著性检验；TMC单种子"
                "结果仅作参考，不能解释为多次运行。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_outputs(
    component_path: Path = DEFAULT_COMPONENT_OUTCOME,
    unseen_path: Path = DEFAULT_UNSEEN_OUTCOME,
    baseline_path: Path = DEFAULT_BASELINE_OUTCOME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    force: bool = False,
) -> dict[str, Any]:
    component_path = _resolve(component_path)
    unseen_path = _resolve(unseen_path)
    baseline_path = _resolve(baseline_path)
    output_dir = _resolve(output_dir)
    inputs = {
        "component_outcome": component_path,
        "unseen_outcome": unseen_path,
        "baseline_outcome": baseline_path,
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Plan A confirmatory inputs missing: " + ", ".join(missing)
        )
    output_paths = {
        "json": output_dir / "OUTCOME.json",
        "markdown": output_dir / "OUTCOME.md",
        "component_csv": output_dir / "component_ablations.csv",
        "unseen_csv": output_dir / "unseen_conditions.csv",
        "baseline_csv": output_dir / "recent_baselines.csv",
        "paired_csv": output_dir / "paired_candidate_minus_v3.csv",
        "latex": output_dir / "MANUSCRIPT_ADDENDUM.tex",
    }
    occupied = [path for path in output_paths.values() if path.exists()]
    if occupied and not force:
        raise FileExistsError(
            "refusing to overwrite Plan A outcome: "
            + ", ".join(str(path) for path in occupied)
        )
    bundle = compile_confirmatory_bundle(
        _read_json(component_path),
        _read_json(unseen_path),
        _read_json(baseline_path),
    )
    bundle["source_bindings"] = {
        **{name: _binding(path) for name, path in inputs.items()},
        "builder_source": _binding(Path(__file__)),
        "qmf_tmc_bibliography": _binding(
            PROJECT_ROOT
            / "paper_rewriting_output"
            / "final_paper"
            / "references.bib"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output_paths["json"],
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(output_paths["markdown"], _markdown(bundle))
    _write_csv(output_paths["component_csv"], bundle["component_ablations"])
    _write_csv(output_paths["unseen_csv"], bundle["unseen_sensor_conditions"])
    _write_csv(output_paths["baseline_csv"], bundle["recent_robust_baselines"])
    flat_paired = []
    for row in bundle["paired_candidate_minus_v3"]:
        flat_paired.append(
            {
                key: value
                for key, value in row.items()
                if key != "per_seed_cluster_bootstrap"
            }
        )
    _write_csv(output_paths["paired_csv"], flat_paired)
    _atomic_text(output_paths["latex"], _latex(bundle))
    return bundle


def main() -> None:
    args = parse_args()
    bundle = build_outputs(
        args.component_outcome,
        args.unseen_outcome,
        args.baseline_outcome,
        args.output_dir,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "state": bundle["state"],
                "valid": bundle["valid"],
                "training_seeds": bundle["training_seeds"],
                "component_count": len(bundle["component_ablations"]),
                "unseen_condition_count": len(
                    bundle["unseen_sensor_conditions"]
                ),
                "baseline_row_count": len(
                    bundle["recent_robust_baselines"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
