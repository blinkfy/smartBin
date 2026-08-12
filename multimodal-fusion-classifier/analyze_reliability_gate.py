"""Summarize direct gate behavior from fixed-checkpoint robustness reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmfc.reliability_gate_diagnostics import (
    DEFAULT_GATE_SCENARIOS,
    compute_gate_diagnostics,
    load_gate_predictions,
    write_gate_artifacts,
)
from mmfc.training import PROJECT_ROOT, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether reliability r directly detects sensor faults"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "reliability-gate-diagnostic"
            / "raw"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "reliability-gate-diagnostic"
            / "summary"
        ),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030]
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=list(DEFAULT_GATE_SCENARIOS)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_gate_predictions(
        resolve_project_path(args.input),
        seeds=args.seeds,
        scenarios=args.scenarios,
    )
    report = compute_gate_diagnostics(
        predictions,
        scenarios=args.scenarios,
    )
    output = resolve_project_path(args.output)
    write_gate_artifacts(report, output)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "sample_count": report["sample_count"],
                "seed_count": report["seed_count"],
                "macro_all": report["macro_detection_all_faults"],
                "macro_available": report[
                    "macro_detection_available_faults"
                ],
                "severity_correlations": report["severity_correlations"],
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

