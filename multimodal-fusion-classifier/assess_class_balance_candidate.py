"""Apply the pre-registered class-balance continuation rule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mmfc.data import write_json
from mmfc.training import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess whether a class-balanced candidate merits more seeds"
    )
    parser.add_argument("--reference-val", type=Path, required=True)
    parser.add_argument("--candidate-val", type=Path, required=True)
    parser.add_argument("--reference-test", type=Path, required=True)
    parser.add_argument("--candidate-test", type=Path, required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--max-test-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"evaluation document has no metrics object: {path}")

    required = ("accuracy", "macro_f1")
    result: dict[str, float] = {}
    for name in required:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"metrics.{name} must be finite in {path}")
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"metrics.{name} must be in [0, 1] in {path}")
        result[name] = numeric
    return result


def _criterion(
    *,
    reference: float,
    candidate: float,
    minimum_delta: float,
) -> dict[str, Any]:
    delta = candidate - reference
    return {
        "reference": reference,
        "candidate": candidate,
        "delta": delta,
        "delta_percentage_points": 100.0 * delta,
        "minimum_delta": minimum_delta,
        "passed": delta >= minimum_delta - 1e-12,
    }


def assess_candidate(
    *,
    reference_val: Mapping[str, float],
    candidate_val: Mapping[str, float],
    reference_test: Mapping[str, float],
    candidate_test: Mapping[str, float],
    reference_name: str,
    candidate_name: str,
    max_test_accuracy_drop: float = 0.01,
) -> dict[str, Any]:
    """Return the locked point-estimate decision for one training seed."""

    if not 0.0 <= max_test_accuracy_drop <= 1.0:
        raise ValueError("max_test_accuracy_drop must be in [0, 1]")

    criteria = {
        "validation_macro_f1_non_decrease": _criterion(
            reference=float(reference_val["macro_f1"]),
            candidate=float(candidate_val["macro_f1"]),
            minimum_delta=0.0,
        ),
        "test_macro_f1_non_decrease": _criterion(
            reference=float(reference_test["macro_f1"]),
            candidate=float(candidate_test["macro_f1"]),
            minimum_delta=0.0,
        ),
        "test_accuracy_drop_within_limit": _criterion(
            reference=float(reference_test["accuracy"]),
            candidate=float(candidate_test["accuracy"]),
            minimum_delta=-max_test_accuracy_drop,
        ),
    }
    passed = all(item["passed"] for item in criteria.values())
    return {
        "valid": True,
        "reference_name": reference_name,
        "candidate_name": candidate_name,
        "max_test_accuracy_drop": max_test_accuracy_drop,
        "criteria": criteria,
        "passed": passed,
        "decision": (
            "extend_training_seeds" if passed else "stop_negative_result"
        ),
        "decision_scope": (
            "Point-estimate pre-registration gate only; paired intervals remain "
            "supporting statistical evidence."
        ),
    }


def main() -> None:
    args = parse_args()
    paths = {
        "reference_val": resolve_project_path(args.reference_val),
        "candidate_val": resolve_project_path(args.candidate_val),
        "reference_test": resolve_project_path(args.reference_test),
        "candidate_test": resolve_project_path(args.candidate_test),
    }
    report = assess_candidate(
        reference_val=_load_metrics(paths["reference_val"]),
        candidate_val=_load_metrics(paths["candidate_val"]),
        reference_test=_load_metrics(paths["reference_test"]),
        candidate_test=_load_metrics(paths["candidate_test"]),
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
        max_test_accuracy_drop=args.max_test_accuracy_drop,
    )
    report["inputs"] = {name: str(path) for name, path in paths.items()}
    output = resolve_project_path(args.output)
    write_json(output, report)
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
