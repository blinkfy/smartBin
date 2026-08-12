"""Reproducible ablation configurations and paper-ready summaries."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AblationDefinition:
    name: str
    group: str
    description: str
    model: Mapping[str, Any]
    loss: Mapping[str, Any]
    degradation_enabled: bool | None
    degradation_overrides: Mapping[str, Any] | None = None


ABLATION_DEFINITIONS: dict[str, AblationDefinition] = {
    "sensor_only": AblationDefinition(
        name="sensor_only",
        group="modality",
        description="Observed 4x4 sensor features only",
        model={
            "enabled_modalities": ["sensor"],
            "fusion_mode": "mean",
            "film_enabled": False,
            "sensor_reliability_enabled": False,
        },
        loss={"reconstruction_weight": 0.0, "calibration_weight": 0.0},
        degradation_enabled=False,
    ),
    "vision_only": AblationDefinition(
        name="vision_only",
        group="modality",
        description="YOLO multi-scale visual features only",
        model={
            "enabled_modalities": ["visual"],
            "fusion_mode": "mean",
            "film_enabled": False,
            "sensor_reliability_enabled": False,
        },
        loss={"reconstruction_weight": 0.0, "calibration_weight": 0.0},
        degradation_enabled=False,
    ),
    "vision_text": AblationDefinition(
        name="vision_text",
        group="modality",
        description="Visual features plus CLIP class-description context",
        model={
            "enabled_modalities": ["visual", "text"],
            "fusion_mode": "mean",
            "film_enabled": False,
            "sensor_reliability_enabled": False,
        },
        loss={"reconstruction_weight": 0.0, "calibration_weight": 0.0},
        degradation_enabled=False,
    ),
    "vision_sensor": AblationDefinition(
        name="vision_sensor",
        group="modality",
        description="Visual features plus observed 4x4 sensor features",
        model={
            "enabled_modalities": ["visual", "sensor"],
            "fusion_mode": "mean",
            "film_enabled": False,
            "sensor_reliability_enabled": False,
        },
        loss={"reconstruction_weight": 0.0, "calibration_weight": 0.0},
        degradation_enabled=False,
    ),
    "text_sensor": AblationDefinition(
        name="text_sensor",
        group="modality",
        description=(
            "Observed 4x4 sensor features plus sensor-conditioned CLIP "
            "class-description context"
        ),
        model={
            "enabled_modalities": ["text", "sensor"],
            "fusion_mode": "mean",
            "film_enabled": False,
            "sensor_reliability_enabled": False,
        },
        loss={"reconstruction_weight": 0.0, "calibration_weight": 0.0},
        degradation_enabled=False,
    ),
    "trimodal_mean": AblationDefinition(
        name="trimodal_mean",
        group="modality",
        description="Visual, text, and sensor adapters with equal-mean fusion",
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "mean",
            "film_enabled": False,
            "sensor_reliability_enabled": False,
        },
        loss={"reconstruction_weight": 0.0, "calibration_weight": 0.0},
        degradation_enabled=False,
    ),
    "trimodal_attention_no_reliability": AblationDefinition(
        name="trimodal_attention_no_reliability",
        group="mechanism",
        description=(
            "FiLM-attention trimodal fusion without sensor prediction or "
            "reliability gating"
        ),
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "attention",
            "film_enabled": True,
            "sensor_reliability_enabled": False,
        },
        loss={"reconstruction_weight": 0.0, "calibration_weight": 0.0},
        degradation_enabled=None,
    ),
    "full": AblationDefinition(
        name="full",
        group="reference",
        description="Complete reliability-gated FiLM-attention model",
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "attention",
            "film_enabled": True,
            "sensor_reliability_enabled": True,
        },
        loss={},
        degradation_enabled=None,
    ),
    "full_quality_aware": AblationDefinition(
        name="full_quality_aware",
        group="enhancement",
        description=(
            "Full model with raw-grid quality correction and broader "
            "degradation training"
        ),
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "attention",
            "film_enabled": True,
            "sensor_reliability_enabled": True,
            "sensor_grid_quality_enabled": True,
            "sensor_grid_quality_hidden_dim": 16,
            "sensor_log_variance_min": -3.0,
        },
        loss={
            "reconstruction_weight": 0.1,
            "calibration_weight": 0.3,
        },
        degradation_enabled=True,
        degradation_overrides={
            "missing_probability": 0.10,
            "noise_probability": 0.50,
            "noise_std_range": [0.02, 0.15],
            "bias_probability": 0.25,
            "bias_strength": 0.15,
            "stuck_probability": 0.15,
            "quality_scale": 0.25,
        },
    ),
    "full_quality_aware_prior": AblationDefinition(
        name="full_quality_aware_prior",
        group="enhancement",
        description=(
            "Quality-aware full model with variance prior and clean "
            "calibration anchor"
        ),
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "attention",
            "film_enabled": True,
            "sensor_reliability_enabled": True,
            "sensor_grid_quality_enabled": True,
            "sensor_grid_quality_hidden_dim": 16,
            "sensor_log_variance_min": -3.0,
        },
        loss={
            "reconstruction_weight": 0.1,
            "calibration_weight": 0.3,
            "variance_prior_weight": 0.1,
            "variance_prior_target": -2.0,
            "calibration_clean_weight": 3.0,
        },
        degradation_enabled=True,
        degradation_overrides={
            "missing_probability": 0.10,
            "noise_probability": 0.50,
            "noise_std_range": [0.02, 0.15],
            "bias_probability": 0.25,
            "bias_strength": 0.15,
            "stuck_probability": 0.15,
            "quality_scale": 0.25,
        },
    ),
    "full_cross_modal_quality": AblationDefinition(
        name="full_cross_modal_quality",
        group="enhancement",
        description=(
            "Quality-aware model with visual-text raw-grid prediction and "
            "pairwise reliability ranking"
        ),
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "attention",
            "film_enabled": True,
            "sensor_reliability_enabled": True,
            "sensor_grid_quality_enabled": True,
            "sensor_grid_quality_hidden_dim": 16,
            "sensor_cross_modal_quality_enabled": True,
            "sensor_grid_decoder_hidden_dim": 64,
            "sensor_cross_modal_quality_hidden_dim": 32,
            "sensor_log_variance_min": -3.0,
        },
        loss={
            "reconstruction_weight": 0.1,
            "calibration_weight": 0.3,
            "variance_prior_weight": 0.1,
            "variance_prior_target": -2.0,
            "calibration_clean_weight": 3.0,
            "sensor_grid_reconstruction_weight": 0.5,
            "calibration_ranking_weight": 0.05,
            "calibration_ranking_margin": 0.05,
        },
        degradation_enabled=True,
        degradation_overrides={
            "missing_probability": 0.10,
            "noise_probability": 0.50,
            "noise_std_range": [0.02, 0.15],
            "bias_probability": 0.25,
            "bias_strength": 0.15,
            "stuck_probability": 0.15,
            "quality_scale": 0.25,
        },
    ),
    "full_no_degradation": AblationDefinition(
        name="full_no_degradation",
        group="mechanism",
        description="Full model without software sensor degradation in training",
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "attention",
            "film_enabled": True,
            "sensor_reliability_enabled": True,
        },
        loss={},
        degradation_enabled=False,
    ),
    "full_no_reconstruction": AblationDefinition(
        name="full_no_reconstruction",
        group="mechanism",
        description="Full model without sensor latent reconstruction loss",
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "attention",
            "film_enabled": True,
            "sensor_reliability_enabled": True,
        },
        loss={"reconstruction_weight": 0.0},
        degradation_enabled=None,
    ),
    "full_no_calibration": AblationDefinition(
        name="full_no_calibration",
        group="mechanism",
        description="Full model without sensor reliability calibration loss",
        model={
            "enabled_modalities": ["visual", "text", "sensor"],
            "fusion_mode": "attention",
            "film_enabled": True,
            "sensor_reliability_enabled": True,
        },
        loss={"calibration_weight": 0.0},
        degradation_enabled=None,
    ),
}


def ablation_names() -> tuple[str, ...]:
    return tuple(ABLATION_DEFINITIONS)


def build_ablation_config(
    base_config: Mapping[str, Any],
    experiment_name: str,
    *,
    checkpoint_root: str | Path = "checkpoints/ablations",
    epochs: int | None = None,
    seed: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    save_checkpoints: bool | None = None,
) -> dict[str, Any]:
    """Create one concrete standalone config from the baseline config."""

    if experiment_name not in ABLATION_DEFINITIONS:
        raise ValueError(f"unknown ablation experiment: {experiment_name}")
    definition = ABLATION_DEFINITIONS[experiment_name]
    config = copy.deepcopy(dict(base_config))
    config.pop("_config_path", None)
    config.setdefault("model", {}).update(copy.deepcopy(dict(definition.model)))
    config.setdefault("loss", {}).update(copy.deepcopy(dict(definition.loss)))
    if definition.degradation_enabled is not None:
        robustness = config.setdefault("robustness", {})
        degradation = robustness.setdefault("sensor_degradation", {})
        degradation["enabled"] = definition.degradation_enabled
    if definition.degradation_overrides:
        robustness = config.setdefault("robustness", {})
        degradation = robustness.setdefault("sensor_degradation", {})
        degradation.update(
            copy.deepcopy(dict(definition.degradation_overrides))
        )

    training = config.setdefault("training", {})
    checkpoint_dir = Path(checkpoint_root) / experiment_name
    training["checkpoint_dir"] = checkpoint_dir.as_posix()
    if epochs is not None:
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        training["epochs"] = int(epochs)
    if seed is not None:
        training["seed"] = int(seed)
    for option_name, value in (
        ("max_train_samples", max_train_samples),
        ("max_val_samples", max_val_samples),
    ):
        if value is not None:
            if value <= 0:
                raise ValueError(f"{option_name} must be greater than 0")
            training[option_name] = int(value)
    if save_checkpoints is not None:
        training["save_checkpoints"] = bool(save_checkpoints)
    config["experiment"] = {
        "name": definition.name,
        "group": definition.group,
        "description": definition.description,
        "formal_retraining_required": True,
        "reference_checkpoint": (
            "checkpoints/qwen-material/best.pt"
            if experiment_name == "full"
            else None
        ),
    }
    return config


def write_ablation_configs(
    base_config: Mapping[str, Any],
    output_dir: Path,
    *,
    experiments: Sequence[str] | None = None,
    checkpoint_root: str | Path = "checkpoints/ablations",
    epochs: int | None = None,
    seed: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    save_checkpoints: bool | None = None,
) -> dict[str, str]:
    """Write deterministic UTF-8 configs for the requested experiment suite."""

    names = list(experiments or ablation_names())
    if not names:
        raise ValueError("at least one ablation experiment is required")
    if len(set(names)) != len(names):
        raise ValueError("ablation experiment names must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for name in names:
        config = build_ablation_config(
            base_config,
            name,
            checkpoint_root=checkpoint_root,
            epochs=epochs,
            seed=seed,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            save_checkpoints=save_checkpoints,
        )
        path = output_dir / f"{name}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        artifacts[name] = str(path.resolve())
    return artifacts


def load_ablation_reports(
    reports_dir: Path,
    experiments: Sequence[str] | None = None,
    *,
    allow_missing: bool = False,
) -> dict[str, dict[str, Any]]:
    names = list(experiments or ablation_names())
    reports: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name in names:
        path = reports_dir / name / "evaluation.json"
        if not path.is_file():
            missing.append(name)
            continue
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
    if missing and not allow_missing:
        raise FileNotFoundError(
            "missing ablation evaluation reports: " + ", ".join(missing)
        )
    if "full" not in reports:
        raise ValueError("ablation reports must include the full reference")
    return reports


def holm_adjust_p_values(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original comparison order."""

    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be between 0 and 1")
    count = len(values)
    if count == 0:
        return []
    order = sorted(range(count), key=values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def summarize_ablation_comparisons(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten paired full-vs-ablation statistics for paper tables."""

    if not comparisons:
        raise ValueError("at least one ablation comparison is required")
    names = list(comparisons)
    p_values = [
        float(comparisons[name]["mcnemar_exact"]["p_value"])
        for name in names
    ]
    adjusted = holm_adjust_p_values(p_values)
    rows: list[dict[str, Any]] = []
    for name, adjusted_p in zip(names, adjusted):
        comparison = comparisons[name]
        reference_name = str(comparison["reference"]["name"])
        candidate_name = str(comparison["candidate"]["name"])
        if reference_name != "full":
            raise ValueError(
                f"{name} comparison reference must be full, got {reference_name}"
            )
        if candidate_name != name:
            raise ValueError(
                f"{name} comparison candidate name is {candidate_name}"
            )
        differences = comparison["paired_bootstrap"]["differences"]
        accuracy = differences["accuracy"]
        macro_f1 = differences["macro_f1"]
        outcomes = comparison["outcome_changes"]
        rows.append(
            {
                "experiment": name,
                "accuracy_delta_pp": 100.0 * float(accuracy["point"]),
                "accuracy_ci_lower_pp": 100.0 * float(accuracy["lower"]),
                "accuracy_ci_upper_pp": 100.0 * float(accuracy["upper"]),
                "macro_f1_delta_pp": 100.0 * float(macro_f1["point"]),
                "macro_f1_ci_lower_pp": 100.0 * float(macro_f1["lower"]),
                "macro_f1_ci_upper_pp": 100.0 * float(macro_f1["upper"]),
                "candidate_only_correct": int(
                    outcomes["candidate_only_correct"]
                ),
                "full_only_correct": int(
                    outcomes["reference_only_correct"]
                ),
                "mcnemar_p_value": float(
                    comparison["mcnemar_exact"]["p_value"]
                ),
                "mcnemar_holm_p_value": float(adjusted_p),
            }
        )
    return rows


def _ablation_comparison_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 消融配对统计",
        "",
        "- 参考模型：`full`；差值定义为消融变体减去 full，单位为百分点。",
        "- 区间：按 `source_image_id` 聚类的 95% 配对 percentile bootstrap。",
        "- 检验：Accuracy 使用双侧精确 McNemar；Holm 列校正本表全部比较。",
        "",
        "| 实验 | Accuracy Δ [95% CI] | Macro F1 Δ [95% CI] | "
        "变体/Full 独有正确 | McNemar p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {accuracy_delta_pp:+.2f} "
            "[{accuracy_ci_lower_pp:+.2f},{accuracy_ci_upper_pp:+.2f}] | "
            "{macro_f1_delta_pp:+.2f} "
            "[{macro_f1_ci_lower_pp:+.2f},{macro_f1_ci_upper_pp:+.2f}] | "
            "{candidate_only_correct}/{full_only_correct} | "
            "{mcnemar_p_value:.4g} | {mcnemar_holm_p_value:.4g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "每个变体与 full 均为单个训练种子下的独立重训。区间量化固定测试样本的"
            "配对不确定性，不包含训练种子方差；跨模型差异不能替代同检查点前向"
            "干预的机制因果证据。",
            "",
        ]
    )
    return "\n".join(lines)


def write_ablation_comparison_artifacts(
    comparisons: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    """Write CSV, JSON, and Markdown paired ablation statistics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize_ablation_comparisons(comparisons)
    csv_path = output_dir / "ablation_pairwise_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "ablation_pairwise_statistics.json"
    json_path.write_text(
        json.dumps(
            {
                "reference": "full",
                "multiple_comparison_correction": "Holm",
                "comparison_count": len(rows),
                "summary": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "ABLATION_STATISTICS.md"
    markdown_path.write_text(
        _ablation_comparison_markdown(rows),
        encoding="utf-8",
    )
    return {
        "statistics_csv": str(csv_path.resolve()),
        "statistics_json": str(json_path.resolve()),
        "statistics_markdown": str(markdown_path.resolve()),
    }


def summarize_ablation_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten evaluation reports and compute deltas from the full model."""

    if "full" not in reports:
        raise ValueError("ablation reports must include the full reference")
    reference = reports["full"]["metrics"]
    rows: list[dict[str, Any]] = []
    for name, report in reports.items():
        if name not in ABLATION_DEFINITIONS:
            raise ValueError(f"unknown ablation experiment: {name}")
        definition = ABLATION_DEFINITIONS[name]
        metrics = report["metrics"]
        performance = report.get("performance", {})
        checkpoint = report.get("checkpoint", {})
        accuracy = float(metrics["accuracy"])
        macro_f1 = float(metrics["macro_f1"])
        rows.append(
            {
                "experiment": name,
                "group": definition.group,
                "description": definition.description,
                "sample_count": int(metrics["sample_count"]),
                "accuracy": accuracy,
                "accuracy_delta_pp": 100.0
                * (accuracy - float(reference["accuracy"])),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": macro_f1,
                "macro_f1_delta_pp": 100.0
                * (macro_f1 - float(reference["macro_f1"])),
                "weighted_f1": float(metrics["weighted_f1"]),
                "ece": float(metrics["ece"]),
                "negative_log_likelihood": float(
                    metrics["negative_log_likelihood"]
                ),
                "multiclass_brier_score": float(
                    metrics["multiclass_brier_score"]
                ),
                "active_parameter_count": int(
                    performance.get(
                        "active_parameter_count",
                        performance.get("parameter_count", 0),
                    )
                ),
                "active_trainable_parameter_count": int(
                    performance.get(
                        "active_trainable_parameter_count",
                        performance.get("trainable_parameter_count", 0),
                    )
                ),
                "amortized_forward_ms_per_sample": float(
                    performance.get("amortized_forward_ms_per_sample", 0.0)
                ),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_path": checkpoint.get("path"),
            }
        )
    return rows


def _write_ablation_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["experiment"]) for row in rows]
    indices = np.arange(len(rows))
    width = 0.36
    figure, axes = plt.subplots(
        2, 1, figsize=(max(10.0, len(rows) * 1.2), 8.5)
    )
    accuracy = np.asarray(
        [100.0 * float(row["accuracy"]) for row in rows]
    )
    macro_f1 = np.asarray(
        [100.0 * float(row["macro_f1"]) for row in rows]
    )
    axes[0].bar(indices - width / 2, accuracy, width, label="Accuracy")
    axes[0].bar(indices + width / 2, macro_f1, width, label="Macro F1")
    lower = max(0.0, float(min(accuracy.min(), macro_f1.min())) - 3.0)
    upper = min(100.0, float(max(accuracy.max(), macro_f1.max())) + 3.0)
    axes[0].set_ylim(lower, upper)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_xticks(indices)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left")
    axes[0].set_title("Multimodal ablation comparison")

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].bar(
        indices - width / 2,
        [float(row["accuracy_delta_pp"]) for row in rows],
        width,
        label="Accuracy delta",
    )
    axes[1].bar(
        indices + width / 2,
        [float(row["macro_f1_delta_pp"]) for row in rows],
        width,
        label="Macro F1 delta",
    )
    axes[1].set_ylabel("Delta vs full (pp)")
    axes[1].set_xticks(indices, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _ablation_markdown(
    rows: list[dict[str, Any]], protocol: str
) -> str:
    protocol_note = (
        "每个实验均应使用对应配置独立训练并以最佳验证集检查点测试。"
        if protocol == "formal_retrained"
        else "所有变体共享完整模型检查点，仅作前向干预诊断，不能替代独立重训消融。"
    )
    lines = [
        "# 多模态消融实验",
        "",
        f"- 协议：`{protocol}`。",
        f"- 说明：{protocol_note}",
        "- 差值定义：变体指标减去 full 指标，单位为百分点。",
        "",
        "| 实验 | Accuracy | 相对 full | Macro F1 | 相对 full | Active Params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {accuracy:.2%} | {accuracy_delta_pp:+.2f} pp | "
            "{macro_f1:.2%} | {macro_f1_delta_pp:+.2f} pp | "
            "{active_parameter_count:,} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            protocol_note,
            "模态贡献结论应同时参考 Accuracy、Macro F1、类别级指标和重复训练方差，"
            "不能仅依据一次检查点或单一总准确率。",
            "",
        ]
    )
    return "\n".join(lines)


def write_ablation_artifacts(
    reports: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    *,
    protocol: str,
) -> dict[str, str]:
    if protocol not in {"formal_retrained", "posthoc_intervention"}:
        raise ValueError(
            "protocol must be formal_retrained or posthoc_intervention"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize_ablation_reports(reports)

    csv_path = output_dir / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "ablation_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "protocol": protocol,
                "reference": "full",
                "experiment_order": list(reports),
                "summary": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "ABLATION_RESULTS.md"
    markdown_path.write_text(
        _ablation_markdown(rows, protocol),
        encoding="utf-8",
    )
    plot_path = output_dir / "ablation_scores.png"
    _write_ablation_plot(plot_path, rows)
    return {
        "summary_csv": str(csv_path.resolve()),
        "summary_json": str(json_path.resolve()),
        "summary_markdown": str(markdown_path.resolve()),
        "scores_png": str(plot_path.resolve()),
    }
