"""Sensor degradation scenarios and paper-ready robustness summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def default_sensor_scenarios() -> dict[str, dict[str, Any] | None]:
    """Return ordered deterministic evaluation scenarios."""

    def options(**overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "enabled": True,
            "missing_probability": 0.0,
            "noise_probability": 0.0,
            "noise_std": 0.0,
            "bias_probability": 0.0,
            "bias_strength": 0.0,
            "stuck_probability": 0.0,
            "quality_scale": 0.25,
        }
        values.update(overrides)
        return values

    return {
        "clean": None,
        "missing_25": options(missing_probability=0.25),
        "missing_50": options(missing_probability=0.50),
        "missing_100": options(missing_probability=1.00),
        "noise_003": options(noise_probability=1.0, noise_std=0.03),
        "noise_008": options(noise_probability=1.0, noise_std=0.08),
        "noise_015": options(noise_probability=1.0, noise_std=0.15),
        "bias_008": options(bias_probability=1.0, bias_strength=0.08),
        "bias_015": options(bias_probability=1.0, bias_strength=0.15),
        "stuck": options(stuck_probability=1.0),
        "stuck_zero": options(stuck_probability=1.0, stuck_value=0.0),
        "stuck_half": options(stuck_probability=1.0, stuck_value=0.5),
        "stuck_one": options(stuck_probability=1.0, stuck_value=1.0),
        "mixed": options(
            missing_probability=0.20,
            noise_probability=0.50,
            noise_std=0.08,
            bias_probability=0.30,
            bias_strength=0.12,
            stuck_probability=0.20,
        ),
    }


def _diagnostic_scalar(result: Mapping[str, Any], name: str) -> float:
    value = result.get("diagnostics", {}).get(name, 0.0)
    array = np.asarray(value, dtype=np.float64)
    return float(array.reshape(-1)[0]) if array.size else 0.0


def _diagnostic_vector(
    result: Mapping[str, Any], name: str, size: int
) -> list[float]:
    value = result.get("diagnostics", {}).get(name, [0.0] * size)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        return [0.0] * size
    return array.tolist()


def summarize_robustness(
    results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten scenario results and compute drops relative to clean."""

    if "clean" not in results:
        raise ValueError("robustness results must include a clean scenario")
    clean_metrics = results["clean"]["metrics"]
    rows: list[dict[str, Any]] = []
    for scenario, result in results.items():
        metrics = result["metrics"]
        condition = result.get("sensor_condition", {})
        calibration = result.get("sensor_reliability_calibration", {})
        modality = _diagnostic_vector(result, "modality_weights", 3)
        rows.append(
            {
                "scenario": scenario,
                "accuracy": float(metrics["accuracy"]),
                "accuracy_drop_pp": 100.0
                * (float(clean_metrics["accuracy"]) - float(metrics["accuracy"])),
                "macro_f1": float(metrics["macro_f1"]),
                "macro_f1_drop_pp": 100.0
                * (float(clean_metrics["macro_f1"]) - float(metrics["macro_f1"])),
                "weighted_f1": float(metrics["weighted_f1"]),
                "ece": float(metrics["ece"]),
                "availability_mean": float(condition.get("availability_mean", 1.0)),
                "mean_absolute_grid_change": float(
                    condition.get("mean_absolute_grid_change", 0.0)
                ),
                "quality_target_mean": float(
                    condition.get("quality_target_mean", 1.0)
                ),
                "sensor_reliability": _diagnostic_scalar(
                    result, "sensor_reliability"
                ),
                "reliability_target_mean": float(
                    calibration.get("target_mean", 1.0)
                ),
                "reliability_mae": float(calibration.get("mae", 0.0)),
                "reliability_rmse": float(calibration.get("rmse", 0.0)),
                "reliability_mean_error": float(
                    calibration.get("mean_error", 0.0)
                ),
                "reliability_pearson": calibration.get(
                    "pearson_correlation"
                ),
                "reliability_spearman": calibration.get(
                    "spearman_correlation"
                ),
                "reliability_pairwise_accuracy": calibration.get(
                    "pairwise_ordering_accuracy"
                ),
                "reliability_pair_count": int(
                    calibration.get("pair_count", 0)
                ),
                "sensor_consistency": _diagnostic_scalar(
                    result, "sensor_consistency"
                ),
                "sensor_discrepancy": _diagnostic_scalar(
                    result, "sensor_discrepancy"
                ),
                "sensor_uncertainty": _diagnostic_scalar(
                    result, "sensor_uncertainty"
                ),
                "visual_weight": modality[0],
                "text_weight": modality[1],
                "sensor_weight": modality[2],
            }
        )
    return rows


def aggregate_robustness_runs(
    seed_results: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate per-seed robustness summaries with sample standard deviations."""

    if not seed_results:
        raise ValueError("at least one seed result is required")

    per_seed = {
        int(seed): summarize_robustness(results)
        for seed, results in seed_results.items()
    }
    first_rows = next(iter(per_seed.values()))
    scenario_order = [str(row["scenario"]) for row in first_rows]
    row_maps = {
        seed: {str(row["scenario"]): row for row in rows}
        for seed, rows in per_seed.items()
    }
    expected = set(scenario_order)
    for seed, rows in row_maps.items():
        if set(rows) != expected:
            raise ValueError(
                f"seed {seed} has different robustness scenarios"
            )

    metric_names = [name for name in first_rows[0] if name != "scenario"]
    seeds = list(per_seed)
    aggregated: list[dict[str, Any]] = []
    for scenario in scenario_order:
        row: dict[str, Any] = {
            "scenario": scenario,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric_name in metric_names:
            values: list[float] = []
            for seed in seeds:
                value = row_maps[seed][scenario].get(metric_name)
                if value is None:
                    continue
                numeric = float(value)
                if np.isfinite(numeric):
                    values.append(numeric)
            row[f"{metric_name}_mean"] = (
                float(np.mean(values)) if values else None
            )
            if len(values) > 1:
                row[f"{metric_name}_std"] = (
                    0.0
                    if all(value == values[0] for value in values[1:])
                    else float(np.std(values, ddof=1))
                )
            else:
                row[f"{metric_name}_std"] = 0.0 if values else None
            if metric_name in {
                "reliability_pearson",
                "reliability_spearman",
                "reliability_pairwise_accuracy",
            }:
                row[f"{metric_name}_valid_seed_count"] = len(values)
        aggregated.append(row)
    return aggregated


def _write_robustness_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["scenario"]) for row in rows]
    indices = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(max(10.0, len(rows) * 1.05), 8.5))

    axes[0].plot(
        indices,
        [100.0 * row["accuracy"] for row in rows],
        marker="o",
        linewidth=2,
        label="Accuracy",
    )
    axes[0].plot(
        indices,
        [100.0 * row["macro_f1"] for row in rows],
        marker="s",
        linewidth=2,
        label="Macro F1",
    )
    classification_scores = [
        100.0 * float(row[metric])
        for row in rows
        for metric in ("accuracy", "macro_f1")
    ]
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(
        max(0.0, min(classification_scores) - 2.0),
        min(100.0, max(classification_scores) + 2.0),
    )
    axes[0].set_xticks(indices)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left")
    axes[0].set_title("Classification robustness under sensor degradation")

    axes[1].plot(
        indices,
        [row["sensor_reliability"] for row in rows],
        marker="o",
        linewidth=2,
        label="Predicted reliability",
    )
    axes[1].plot(
        indices,
        [row["availability_mean"] for row in rows],
        marker="s",
        linewidth=2,
        label="Availability",
    )
    axes[1].plot(
        indices,
        [row["quality_target_mean"] for row in rows],
        marker="^",
        linewidth=2,
        label="Quality target",
    )
    axes[1].set_ylabel("Mean value")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(indices, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="lower left")

    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _write_multiseed_robustness_plot(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["scenario"]) for row in rows]
    indices = np.arange(len(rows))
    figure, axes = plt.subplots(
        2, 1, figsize=(max(10.0, len(rows) * 1.05), 8.5)
    )

    def values(name: str, scale: float = 1.0) -> np.ndarray:
        return np.asarray(
            [
                np.nan if row.get(name) is None else scale * float(row[name])
                for row in rows
            ],
            dtype=np.float64,
        )

    axes[0].errorbar(
        indices,
        values("accuracy_mean", 100.0),
        yerr=values("accuracy_std", 100.0),
        marker="o",
        linewidth=2,
        capsize=4,
        label="Accuracy",
    )
    axes[0].errorbar(
        indices,
        values("macro_f1_mean", 100.0),
        yerr=values("macro_f1_std", 100.0),
        marker="s",
        linewidth=2,
        capsize=4,
        label="Macro F1",
    )
    classification_scores = np.concatenate(
        (
            values("accuracy_mean", 100.0)
            - values("accuracy_std", 100.0),
            values("accuracy_mean", 100.0)
            + values("accuracy_std", 100.0),
            values("macro_f1_mean", 100.0)
            - values("macro_f1_std", 100.0),
            values("macro_f1_mean", 100.0)
            + values("macro_f1_std", 100.0),
        )
    )
    finite_scores = classification_scores[
        np.isfinite(classification_scores)
    ]
    axes[0].set_ylabel("Score mean +/- std (%)")
    axes[0].set_ylim(
        max(0.0, float(np.min(finite_scores)) - 2.0),
        min(100.0, float(np.max(finite_scores)) + 2.0),
    )
    axes[0].set_xticks(indices)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left")
    axes[0].set_title(
        "Classification robustness under sensor degradation"
    )

    for mean_name, std_name, marker, label in (
        (
            "sensor_reliability_mean",
            "sensor_reliability_std",
            "o",
            "Predicted reliability",
        ),
        (
            "availability_mean_mean",
            "availability_mean_std",
            "s",
            "Availability",
        ),
        (
            "quality_target_mean_mean",
            "quality_target_mean_std",
            "^",
            "Quality target",
        ),
    ):
        axes[1].errorbar(
            indices,
            values(mean_name),
            yerr=values(std_name),
            marker=marker,
            linewidth=2,
            capsize=4,
            label=label,
        )
    axes[1].set_ylabel("Mean value +/- std")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(indices, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="lower left")

    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_robustness_artifacts(
    results: Mapping[str, Mapping[str, Any]], output_dir: Path
) -> dict[str, str]:
    """Write compact cross-scenario summaries and a publication-ready plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize_robustness(results)
    csv_path = output_dir / "robustness_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "robustness_summary.json"
    json_path.write_text(
        json.dumps(
            {"scenario_order": list(results), "summary": rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_path = output_dir / "robustness_curves.png"
    _write_robustness_plot(plot_path, rows)
    return {
        "summary_csv": str(csv_path.resolve()),
        "summary_json": str(json_path.resolve()),
        "curves_png": str(plot_path.resolve()),
    }


def write_multiseed_robustness_artifacts(
    seed_results: Mapping[int, Mapping[str, Mapping[str, Any]]],
    output_dir: Path,
) -> dict[str, str]:
    """Write cross-seed mean/std tables, raw summaries, and error-bar plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate_robustness_runs(seed_results)
    csv_path = output_dir / "robustness_multiseed_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_seed = {
        str(seed): summarize_robustness(results)
        for seed, results in seed_results.items()
    }
    json_path = output_dir / "robustness_multiseed_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "seeds": [int(seed) for seed in seed_results],
                "seed_count": len(seed_results),
                "scenario_order": [
                    str(row["scenario"]) for row in rows
                ],
                "summary": rows,
                "per_seed": per_seed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_path = output_dir / "robustness_multiseed_curves.png"
    _write_multiseed_robustness_plot(plot_path, rows)
    return {
        "summary_csv": str(csv_path.resolve()),
        "summary_json": str(json_path.resolve()),
        "curves_png": str(plot_path.resolve()),
    }
