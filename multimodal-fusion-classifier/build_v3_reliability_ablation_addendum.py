"""Build a concise paper-facing addendum from the audited ablation outcome."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability"
)
DEFAULT_OUTCOME = REPORT_ROOT / "OUTCOME.json"
DEFAULT_OUTPUT = REPORT_ROOT / "PAPER_ADDENDUM.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v3 reliability-ablation paper addendum"
    )
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _effect_row(
    aggregate: Mapping[str, Any],
    metric: str,
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    row = aggregate[metric]
    mean = float(row["mean"])
    benefit = -mean if lower_is_better else mean
    return {
        "metric": metric,
        "full_minus_ablation_mean": mean,
        "sample_std": float(row["sample_std"]),
        "full_benefit_mean": benefit,
        "favorable_seed_count": (
            int(row["negative_seed_count"])
            if lower_is_better
            else int(row["positive_seed_count"])
        ),
        "training_seed_count": 3,
    }


def build_addendum(
    outcome_path: Path = DEFAULT_OUTCOME,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    outcome_path = _resolve(outcome_path)
    output_path = _resolve(output_path)
    outcome = _read_json(outcome_path)
    if (
        outcome.get("state") != "complete"
        or outcome.get("valid") is not True
        or outcome.get("errors") != []
    ):
        raise ValueError("audited ablation outcome is not complete and valid")
    matched = outcome["matched_test_comparison"]
    aggregate = matched["aggregate"]
    effects = [
        _effect_row(aggregate, "accuracy"),
        _effect_row(aggregate, "macro_f1"),
        _effect_row(aggregate, "negative_log_likelihood", lower_is_better=True),
        _effect_row(aggregate, "multiclass_brier_score", lower_is_better=True),
        _effect_row(aggregate, "ece", lower_is_better=True),
    ]
    robustness_rows = outcome["robustness_comparison"]["rows"]
    addendum = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "paper_ready_v3_reliability_subsystem_ablation",
        "effect_direction": (
            "Positive full_benefit_mean favors full v3 for every reported metric."
        ),
        "clean_test_matched_seed_effects": effects,
        "robustness_seed42_effects": robustness_rows,
        "recommended_claim": (
            "Report the three matched-training-seed clean-test effects as the "
            "primary component evidence, and the seed-42 degradation results as "
            "supporting descriptive robustness evidence."
        ),
        "claim_boundaries": [
            "The intervention removes the entire reliability subsystem, not only a gate.",
            "The internal test set has been inspected and is not an external blind set.",
            "Qwen sidecars are image-conditioned soft-sensor observations.",
            "Five degradation seeds are perturbation repeats, not training replicates.",
        ],
        "source": {
            "path": outcome_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(outcome_path),
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n",
    )
    with output_path.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    accuracy = next(row for row in effects if row["metric"] == "accuracy")
    macro_f1 = next(row for row in effects if row["metric"] == "macro_f1")
    ece = next(row for row in effects if row["metric"] == "ece")
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# v3 reliability-subsystem ablation: paper addendum",
                "",
                "| Clean test metric | Full-v3 benefit mean | Sample SD | Favorable seeds |",
                "|---|---:|---:|---:|",
                *[
                    (
                        f"| {row['metric']} | {row['full_benefit_mean']:+.6f} | "
                        f"{row['sample_std']:.6f} | "
                        f"{row['favorable_seed_count']}/3 |"
                    )
                    for row in effects
                ],
                "",
                (
                    "Plain-language result: full v3 minus the no-reliability "
                    f"ablation averaged {accuracy['full_benefit_mean'] * 100:+.2f} "
                    "percentage points in Accuracy and "
                    f"{macro_f1['full_benefit_mean'] * 100:+.2f} points in "
                    f"Macro F1; its ECE benefit was {ece['full_benefit_mean'] * 100:+.2f} "
                    "points."
                ),
                "",
                "Use the three matched training seeds for the primary claim. "
                "Treat the seed-42 degradation analysis as descriptive support.",
                "",
            ]
        ),
    )
    return addendum


def main() -> None:
    args = parse_args()
    result = build_addendum(args.outcome, args.output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "output": _resolve(args.output).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
