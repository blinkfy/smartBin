"""Plot the locked v2b/v3 robustness and gate comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from mmfc.data import write_json
from mmfc.training import resolve_project_path


REQUIRED_COLUMNS = (
    "scenario",
    "v2b_accuracy",
    "v3_accuracy",
    "v2b_macro_f1",
    "v3_macro_f1",
    "no_gate_macro_f1",
    "v3_minus_v2b_macro_f1_pp",
    "gate_minus_no_gate_macro_f1_pp",
    "v2b_reliability_mae",
    "v3_reliability_mae",
    "v2b_spearman",
    "v3_spearman",
    "v2b_pairwise_accuracy",
    "v3_pairwise_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot robustness comparison across fixed scenarios"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _optional_float(value: str) -> float:
    return float(value) if value.strip() else float("nan")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("comparison CSV has no header")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"comparison CSV is missing columns: {missing}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {"scenario": raw["scenario"]}
            for name in REQUIRED_COLUMNS[1:]:
                row[name] = _optional_float(raw[name])
            rows.append(row)
    if not rows:
        raise ValueError("comparison CSV has no scenarios")
    return rows


def _short_label(name: str) -> str:
    replacements = {
        "missing_25": "miss 25%",
        "missing_50": "miss 50%",
        "missing_100": "miss 100%",
        "noise_003": "noise .03",
        "noise_008": "noise .08",
        "noise_015": "noise .15",
        "bias_008": "bias .08",
    }
    return replacements.get(name, name)


def plot_rows(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = np.arange(len(rows))
    labels = [_short_label(str(row["scenario"])) for row in rows]
    blue = "#1967A3"
    red = "#D1495B"
    green = "#2A9D8F"
    gold = "#E9A23B"
    violet = "#7A5195"
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), sharex=True)

    score_axis = axes[0, 0]
    score_axis.plot(
        indices,
        100.0 * np.asarray([row["v2b_macro_f1"] for row in rows]),
        marker="o",
        color=blue,
        label="v2b",
    )
    score_axis.plot(
        indices,
        100.0 * np.asarray([row["v3_macro_f1"] for row in rows]),
        marker="o",
        color=red,
        label="v3 gate",
    )
    score_axis.plot(
        indices,
        100.0 * np.asarray([row["no_gate_macro_f1"] for row in rows]),
        marker="o",
        color=green,
        linestyle="--",
        label="v3 no gate",
    )
    score_axis.set_ylabel("Macro F1 (%)")
    score_axis.set_title("Classification under sensor degradation")
    score_axis.legend(loc="best", fontsize=8)

    delta_axis = axes[0, 1]
    width = 0.36
    delta_axis.axhline(0.0, color="#202124", linewidth=1.0)
    delta_axis.bar(
        indices - width / 2,
        [row["v3_minus_v2b_macro_f1_pp"] for row in rows],
        width,
        color=gold,
        label="v3 - v2b",
    )
    delta_axis.bar(
        indices + width / 2,
        [row["gate_minus_no_gate_macro_f1_pp"] for row in rows],
        width,
        color=violet,
        label="gate - no gate",
    )
    delta_axis.set_ylabel("Macro F1 delta (pp)")
    delta_axis.set_title("Incremental contribution")
    delta_axis.legend(loc="best", fontsize=8)

    mae_axis = axes[1, 0]
    mae_axis.plot(
        indices,
        [row["v2b_reliability_mae"] for row in rows],
        marker="o",
        color=blue,
        label="v2b",
    )
    mae_axis.plot(
        indices,
        [row["v3_reliability_mae"] for row in rows],
        marker="o",
        color=red,
        label="v3",
    )
    mae_axis.set_ylabel("Reliability MAE (lower is better)")
    mae_axis.set_title("Reliability magnitude error")
    mae_axis.legend(loc="best", fontsize=8)

    rank_axis = axes[1, 1]
    rank_specs = (
        ("v2b_spearman", "v2b Spearman", blue, "-"),
        ("v3_spearman", "v3 Spearman", red, "-"),
        ("v2b_pairwise_accuracy", "v2b pairwise", green, "--"),
        ("v3_pairwise_accuracy", "v3 pairwise", violet, "--"),
    )
    for key, label, color, style in rank_specs:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        rank_axis.plot(
            indices,
            values,
            marker="o",
            color=color,
            linestyle=style,
            label=label,
        )
    rank_axis.axhline(0.0, color="#202124", linewidth=0.8)
    rank_axis.set_ylabel("Rank/order score")
    rank_axis.set_title("Continuous reliability ordering")
    rank_axis.legend(loc="best", fontsize=7, ncol=2)

    for axis in axes.reshape(-1):
        axis.grid(axis="y", alpha=0.2)
        axis.set_xticks(indices, labels, rotation=32, ha="right")
    mae_axis.set_xlabel("Evaluation scenario")
    rank_axis.set_xlabel("Evaluation scenario")
    figure.suptitle(
        "Cross-modal reliability robustness: five degradation seeds",
        fontsize=14,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    v3_positive = [
        row["v3_minus_v2b_macro_f1_pp"]
        for row in rows
        if math.isfinite(row["v3_minus_v2b_macro_f1_pp"])
        and row["v3_minus_v2b_macro_f1_pp"] > 0.0
    ]
    gate_positive = [
        row["gate_minus_no_gate_macro_f1_pp"]
        for row in rows
        if math.isfinite(row["gate_minus_no_gate_macro_f1_pp"])
        and row["gate_minus_no_gate_macro_f1_pp"] > 0.0
    ]
    return {
        "scenario_count": len(rows),
        "v3_macro_f1_improved_scenarios": len(v3_positive),
        "gate_macro_f1_improved_scenarios": len(gate_positive),
        "output": str(output),
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output = resolve_project_path(args.output)
    rows = load_rows(input_path)
    report = plot_rows(rows, output)
    report["input"] = str(input_path)
    report_path = output.with_suffix(".json")
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
