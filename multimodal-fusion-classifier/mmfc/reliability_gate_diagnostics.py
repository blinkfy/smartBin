"""Direct diagnostics for the learned sensor-reliability gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_GATE_SCENARIOS = (
    "clean",
    "missing_100",
    "noise_003",
    "noise_008",
    "noise_015",
    "bias_008",
    "bias_015",
    "stuck",
    "mixed",
)

DISPLAY_LABELS = {
    "clean": "Clean",
    "missing_100": "Missing",
    "noise_003": "Noise\nlow",
    "noise_008": "Noise\nmedium",
    "noise_015": "Noise\nhigh",
    "bias_008": "Bias\nlow",
    "bias_015": "Bias\nhigh",
    "stuck": "Stuck",
    "mixed": "Mixed",
}


def _read_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction file not found: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"{path}:{line_number} has no sample_id")
            if sample_id in records:
                raise ValueError(f"{path}:{line_number} repeats {sample_id}")
            reliability = float(record["sensor_reliability"])
            if not np.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
                raise ValueError(
                    f"{path}:{line_number} has invalid sensor_reliability"
                )
            records[sample_id] = record
    if not records:
        raise ValueError(f"prediction file is empty: {path}")
    return records


def load_gate_predictions(
    root: Path,
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str] = DEFAULT_GATE_SCENARIOS,
) -> dict[int, dict[str, dict[str, dict[str, Any]]]]:
    """Load and align per-sample predictions from robustness reports."""

    loaded: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for seed in seeds:
        seed_records: dict[str, dict[str, dict[str, Any]]] = {}
        expected_ids: set[str] | None = None
        for scenario in scenarios:
            records = _read_prediction_map(
                root
                / "seeds"
                / f"seed-{int(seed)}"
                / scenario
                / "predictions.jsonl"
            )
            sample_ids = set(records)
            if expected_ids is None:
                expected_ids = sample_ids
            elif sample_ids != expected_ids:
                raise ValueError(
                    f"seed {seed} scenario {scenario} has misaligned samples"
                )
            seed_records[scenario] = records
        loaded[int(seed)] = seed_records
    if not loaded:
        raise ValueError("at least one perturbation seed is required")
    return loaded


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _average_ranks(np.asarray(left, dtype=np.float64).reshape(-1))
    right_ranks = _average_ranks(np.asarray(right, dtype=np.float64).reshape(-1))
    if np.std(left_ranks) <= 1e-12 or np.std(right_ranks) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def _scenario_vector(
    records: Mapping[str, Mapping[str, Any]], sample_ids: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [float(records[sample_id]["sensor_reliability"]) for sample_id in sample_ids],
        dtype=np.float64,
    )


def compute_gate_diagnostics(
    predictions: Mapping[
        int, Mapping[str, Mapping[str, Mapping[str, Any]]]
    ],
    *,
    scenarios: Sequence[str] = DEFAULT_GATE_SCENARIOS,
) -> dict[str, Any]:
    """Compute gate distributions, fault detection, and severity ordering."""

    seeds = sorted(int(seed) for seed in predictions)
    if "clean" not in scenarios:
        raise ValueError("gate diagnostics require the clean scenario")
    sample_ids = sorted(predictions[seeds[0]]["clean"])
    per_seed_vectors: dict[int, dict[str, np.ndarray]] = {}
    per_seed_quality: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        per_seed_vectors[seed] = {
            scenario: _scenario_vector(
                predictions[seed][scenario], sample_ids
            )
            for scenario in scenarios
        }
        per_seed_quality[seed] = {
            scenario: np.asarray(
                [
                    float(
                        predictions[seed][scenario][sample_id].get(
                            "sensor_quality_target",
                            1.0 if scenario == "clean" else 0.0,
                        )
                    )
                    for sample_id in sample_ids
                ],
                dtype=np.float64,
            )
            for scenario in scenarios
        }

    distribution_summary: list[dict[str, Any]] = []
    distribution_values: dict[str, list[float]] = {}
    for scenario in scenarios:
        stacked = np.stack(
            [per_seed_vectors[seed][scenario] for seed in seeds], axis=0
        )
        sample_medians = np.median(stacked, axis=0)
        distribution_values[scenario] = sample_medians.tolist()
        distribution_summary.append(
            {
                "scenario": scenario,
                "sample_count": int(sample_medians.size),
                "mean": float(sample_medians.mean()),
                "std": float(sample_medians.std(ddof=1)),
                "q1": float(np.quantile(sample_medians, 0.25)),
                "median": float(np.median(sample_medians)),
                "q3": float(np.quantile(sample_medians, 0.75)),
            }
        )

    detection: list[dict[str, Any]] = []
    fault_scenarios = [scenario for scenario in scenarios if scenario != "clean"]
    for scenario in fault_scenarios:
        nominal_aurocs: list[float] = []
        nominal_auprcs: list[float] = []
        effective_aurocs: list[float] = []
        effective_auprcs: list[float] = []
        effective_fractions: list[float] = []
        for seed in seeds:
            clean = per_seed_vectors[seed]["clean"]
            fault = per_seed_vectors[seed][scenario]
            targets = np.concatenate(
                (np.ones(clean.size), np.zeros(fault.size))
            )
            scores = np.concatenate((clean, fault))
            nominal_aurocs.append(float(roc_auc_score(targets, scores)))
            nominal_auprcs.append(float(average_precision_score(targets, scores)))
            effective_mask = per_seed_quality[seed][scenario] < 1.0 - 1e-6
            if not bool(effective_mask.any()):
                raise ValueError(
                    f"scenario {scenario} seed {seed} has no effective faults"
                )
            effective_fault = fault[effective_mask]
            effective_targets = np.concatenate(
                (np.ones(clean.size), np.zeros(effective_fault.size))
            )
            effective_scores = np.concatenate((clean, effective_fault))
            effective_weights = np.concatenate(
                (
                    np.ones(clean.size, dtype=np.float64),
                    np.full(
                        effective_fault.size,
                        clean.size / effective_fault.size,
                        dtype=np.float64,
                    ),
                )
            )
            effective_aurocs.append(
                float(roc_auc_score(effective_targets, effective_scores))
            )
            effective_auprcs.append(
                float(
                    average_precision_score(
                        effective_targets,
                        effective_scores,
                        sample_weight=effective_weights,
                    )
                )
            )
            effective_fractions.append(float(effective_mask.mean()))
        detection.append(
            {
                "scenario": scenario,
                "auroc": _mean_std(effective_aurocs),
                "auprc": _mean_std(effective_auprcs),
                "nominal_auroc": _mean_std(nominal_aurocs),
                "nominal_auprc": _mean_std(nominal_auprcs),
                "effective_fault_fraction": _mean_std(effective_fractions),
                "per_seed": {
                    str(seed): {
                        "auroc": auroc,
                        "auprc": auprc,
                        "nominal_auroc": nominal_auroc,
                        "nominal_auprc": nominal_auprc,
                        "effective_fault_fraction": effective_fraction,
                    }
                    for (
                        seed,
                        auroc,
                        auprc,
                        nominal_auroc,
                        nominal_auprc,
                        effective_fraction,
                    ) in zip(
                        seeds,
                        effective_aurocs,
                        effective_auprcs,
                        nominal_aurocs,
                        nominal_auprcs,
                        effective_fractions,
                    )
                },
            }
        )

    def macro_detection(selected: Sequence[str]) -> dict[str, float]:
        rows = [row for row in detection if row["scenario"] in selected]
        return {
            "auroc": float(np.mean([row["auroc"]["mean"] for row in rows])),
            "auprc": float(np.mean([row["auprc"]["mean"] for row in rows])),
            "scenario_count": len(rows),
        }

    severity_families = {
        "noise": (
            ("clean", 0.0),
            ("noise_003", 0.03),
            ("noise_008", 0.08),
            ("noise_015", 0.15),
        ),
        "bias": (
            ("clean", 0.0),
            ("bias_008", 0.08),
            ("bias_015", 0.15),
        ),
    }
    severity_correlations: dict[str, Any] = {}
    for family, levels in severity_families.items():
        if any(scenario not in scenarios for scenario, _ in levels):
            continue
        correlations: list[float] = []
        for seed in seeds:
            scores = np.concatenate(
                [per_seed_vectors[seed][scenario] for scenario, _ in levels]
            )
            severity = np.concatenate(
                [
                    np.full(len(sample_ids), value, dtype=np.float64)
                    for _, value in levels
                ]
            )
            correlations.append(_spearman(scores, severity))
        severity_correlations[family] = {
            **_mean_std(correlations),
            "per_seed": {
                str(seed): value for seed, value in zip(seeds, correlations)
            },
            "levels": [
                {"scenario": scenario, "severity": severity}
                for scenario, severity in levels
            ],
        }

    available_faults = [
        scenario for scenario in fault_scenarios if scenario != "missing_100"
    ]
    return {
        "valid": True,
        "seeds": seeds,
        "seed_count": len(seeds),
        "sample_count": len(sample_ids),
        "scenario_order": list(scenarios),
        "distribution_summary": distribution_summary,
        "distribution_values": distribution_values,
        "fault_detection": detection,
        "macro_detection_all_faults": macro_detection(fault_scenarios),
        "macro_detection_available_faults": macro_detection(available_faults),
        "severity_correlations": severity_correlations,
        "protocol": {
            "positive_class": "clean sensor observation",
            "negative_class": (
                "effective controlled fault with quality_target < 1 - 1e-6"
            ),
            "nominal_detection_also_reported": True,
            "auprc_baseline": 0.5,
            "auprc_weighting": (
                "fault records are reweighted to equal total clean weight"
            ),
            "distribution_aggregation": (
                "per-sample median reliability across perturbation seeds"
            ),
            "metric_aggregation": (
                "metric per perturbation seed, then mean and sample standard deviation"
            ),
        },
    }


def write_gate_figure(report: Mapping[str, Any], path: Path) -> None:
    """Write an editable two-panel SVG diagnostic figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(report["scenario_order"])
    values = [report["distribution_values"][scenario] for scenario in scenarios]
    detection = report["fault_detection"]
    colors = {
        "clean": "#4f7d65",
        "missing_100": "#777777",
        "noise_003": "#5b7fa3",
        "noise_008": "#5b7fa3",
        "noise_015": "#5b7fa3",
        "bias_008": "#a4514d",
        "bias_015": "#a4514d",
        "stuck": "#766c8b",
        "mixed": "#333333",
    }
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "svg.hashsalt": "mmfc-gate-diagnostic-2026",
        }
    )
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.0),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.25},
    )

    parts = left.violinplot(
        values,
        positions=np.arange(len(scenarios)),
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.22,
    )
    for body, scenario in zip(parts["bodies"], scenarios):
        body.set_facecolor(colors[scenario])
        body.set_edgecolor("#30343a")
        body.set_linewidth(0.65)
        body.set_alpha(0.72)
    box = left.boxplot(
        values,
        positions=np.arange(len(scenarios)),
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.2},
        boxprops={"facecolor": "#30343a", "linewidth": 0.6},
        whiskerprops={"color": "#30343a", "linewidth": 0.7},
        capprops={"color": "#30343a", "linewidth": 0.7},
    )
    for median in box["medians"]:
        median.set_zorder(5)
    left.set_xticks(
        np.arange(len(scenarios)),
        [DISPLAY_LABELS[scenario] for scenario in scenarios],
    )
    left.set_ylim(-0.03, 1.03)
    left.set_ylabel("Reliability score $r$")
    left.set_title("(a) Gate response under controlled sensor faults", loc="left")
    left.grid(axis="y", color="#d9dce1", linewidth=0.55)
    left.set_axisbelow(True)

    fault_names = [row["scenario"] for row in detection]
    indices = np.arange(len(fault_names))
    width = 0.36
    auroc = [row["auroc"]["mean"] for row in detection]
    auroc_std = [row["auroc"]["std"] for row in detection]
    auprc = [row["auprc"]["mean"] for row in detection]
    auprc_std = [row["auprc"]["std"] for row in detection]
    right.bar(
        indices - width / 2,
        auroc,
        width,
        yerr=auroc_std,
        capsize=2,
        color="#4f7d65",
        label="AUROC",
    )
    right.bar(
        indices + width / 2,
        auprc,
        width,
        yerr=auprc_std,
        capsize=2,
        color="#6d6783",
        label="AUPRC",
    )
    right.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8)
    right.set_xticks(
        indices,
        [DISPLAY_LABELS[scenario] for scenario in fault_names],
    )
    right.set_ylim(0.45, 1.02)
    right.set_ylabel("Clean-versus-fault discrimination")
    right.set_title("(b) Direct fault-detection performance", loc="left")
    right.grid(axis="y", color="#d9dce1", linewidth=0.55)
    right.set_axisbelow(True)
    right.legend(frameon=False, ncol=2, loc="upper right")
    for axis in (left, right):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.065, right=0.99, top=0.91, bottom=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format=path.suffix.lstrip("."),
        facecolor="white",
        metadata={"Date": None, "Creator": "Matplotlib"},
    )
    plt.close(figure)


def write_gate_artifacts(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(report)
    (output_dir / "GATE_DIAGNOSTICS.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = list(report["distribution_summary"])
    with (output_dir / "gate_distribution_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    detection_rows = []
    for row in report["fault_detection"]:
        detection_rows.append(
            {
                "scenario": row["scenario"],
                "auroc_mean": row["auroc"]["mean"],
                "auroc_std": row["auroc"]["std"],
                "auprc_mean": row["auprc"]["mean"],
                "auprc_std": row["auprc"]["std"],
                "nominal_auroc_mean": row["nominal_auroc"]["mean"],
                "nominal_auroc_std": row["nominal_auroc"]["std"],
                "nominal_auprc_mean": row["nominal_auprc"]["mean"],
                "nominal_auprc_std": row["nominal_auprc"]["std"],
                "effective_fault_fraction_mean": row[
                    "effective_fault_fraction"
                ]["mean"],
            }
        )
    with (output_dir / "gate_fault_detection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detection_rows[0]))
        writer.writeheader()
        writer.writerows(detection_rows)
    write_gate_figure(report, output_dir / "reliability_gate_diagnostic.svg")
