"""Build a paper-facing addendum from the completed v3 seed replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
BUILDER_SOURCE = Path(__file__).resolve()
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "full_cross_modal_quality"
    / "seed-replication"
)
DEFAULT_OUTCOME = REPORT_ROOT / "OUTCOME.json"
DEFAULT_OUTPUT = REPORT_ROOT / "PAPER_ADDENDUM.json"
DISPLAY_METRICS = (
    ("accuracy", "Accuracy"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("negative_log_likelihood", "NLL"),
    ("multiclass_brier_score", "Multiclass Brier"),
    ("ece", "ECE"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paper addendum for the v3 seed replication"
    )
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_source_bindings(
    bindings: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(f"malformed outcome source binding: {name}")
            continue
        raw_path = binding.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"malformed outcome source path: {name}")
            continue
        path = _resolve(Path(raw_path))
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"outcome source escapes project: {name}")
            continue
        if not path.is_file():
            errors.append(f"outcome source is missing: {name}")
            continue
        if path.stat().st_size != binding.get("size_bytes"):
            errors.append(f"outcome source size mismatch: {name}")
        if _sha256(path) != binding.get("sha256"):
            errors.append(f"outcome source hash mismatch: {name}")
    return errors


def _validate_outcome(outcome: Mapping[str, Any]) -> list[str]:
    errors = []
    training = outcome.get("training", {})
    validation = outcome.get("validation_summary", {})
    test = outcome.get("test_summary", {})
    if (
        outcome.get("state") != "complete"
        or outcome.get("valid") is not True
        or outcome.get("errors") != []
        or training.get("seeds") != [42, 43, 44]
        or training.get("new_replicate_seeds") != [43, 44]
        or training.get("epochs_per_run") != 100
        or training.get("resume_used") is not False
        or training.get("best_seed_replacement_performed") is not False
    ):
        errors.append("outcome identity or training invariants mismatch")
    for split_name, summary in (("validation", validation), ("test", test)):
        if (
            summary.get("valid") is not True
            or summary.get("run_count") != 3
            or summary.get("sample_count_per_run") != 826
            or [item.get("name") for item in summary.get("individual", [])]
            != ["seed42", "seed43", "seed44"]
        ):
            errors.append(f"{split_name} summary identity mismatch")
        aggregate = summary.get("aggregate", {})
        for metric, _ in DISPLAY_METRICS:
            statistics = aggregate.get(metric, {})
            if set(statistics) != {"mean", "sample_std", "min", "max"}:
                errors.append(f"{split_name} aggregate is incomplete: {metric}")
    bindings = outcome.get("source_bindings", {})
    if not isinstance(bindings, Mapping) or not bindings:
        errors.append("outcome source bindings are missing")
    else:
        errors.extend(_validate_source_bindings(bindings))
    return errors


def _write_csv(path: Path, addendum: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "metric",
                "mean",
                "sample_std",
                "min",
                "max",
            ),
        )
        writer.writeheader()
        for split in ("validation", "test"):
            for metric, _ in DISPLAY_METRICS:
                writer.writerow(
                    {
                        "split": split,
                        "metric": metric,
                        **addendum["statistics"][split][metric],
                    }
                )


def build_addendum(
    *,
    outcome_path: Path = DEFAULT_OUTCOME,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    outcome_path = _resolve(outcome_path)
    output_path = _resolve(output_path)
    if not outcome_path.is_file():
        raise FileNotFoundError(f"seed replication outcome is missing: {outcome_path}")
    outcome = _read_json(outcome_path)
    errors = _validate_outcome(outcome)
    if errors:
        raise ValueError("; ".join(errors))
    validation = outcome["validation_summary"]
    test = outcome["test_summary"]
    addendum = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "paper_evidence_extension_for_v3_training_seed_stability",
        "training_seeds": [42, 43, 44],
        "new_training_seeds": [43, 44],
        "epochs_per_seed": 100,
        "sample_count_per_split_per_seed": 826,
        "statistics": {
            "validation": validation["aggregate"],
            "test": test["aggregate"],
        },
        "individual": {
            "validation": validation["individual"],
            "test": test["individual"],
        },
        "claims": {
            "permitted": [
                (
                    "The fixed v3 configuration was trained from scratch with "
                    "seeds 42, 43, and 44 and summarized descriptively."
                ),
                (
                    "Training-seed stability is reported as mean, sample "
                    "standard deviation, minimum, and maximum."
                ),
            ],
            "forbidden": [
                "claiming that the fixed internal test is an external blind set",
                "selecting a replacement main model from the best test seed",
                "generalizing three-seed variation to deployment uncertainty",
                "describing Qwen sidecars as hardware sensor measurements",
            ],
        },
        "source_bindings": {
            "builder_source": _binding(BUILDER_SOURCE),
            "outcome": _binding(outcome_path),
            "protocol": _binding(REPORT_ROOT / "PROTOCOL.json"),
            "validation_summary": _binding(
                REPORT_ROOT / "VALIDATION_SEED_SUMMARY.json"
            ),
            "test_summary": _binding(REPORT_ROOT / "TEST_SEED_SUMMARY.json"),
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n",
    )
    csv_path = output_path.with_name("training_seed_replication.csv")
    _write_csv(csv_path, addendum)
    lines = [
        "# v3 training-seed replication paper addendum",
        "",
        "The fixed v3 configuration was trained independently with seeds 42, "
        "43, and 44 for 100 epochs per seed. Seed 42 remains the main model; "
        "the additional runs are confirmatory and were not used for test-based "
        "model replacement.",
        "",
        "| Split | Metric | Mean | Sample SD | Min | Max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        for metric, display_name in DISPLAY_METRICS:
            values = addendum["statistics"][split][metric]
            lines.append(
                f"| {split} | {display_name} | {values['mean']:.4f} | "
                f"{values['sample_std']:.4f} | {values['min']:.4f} | "
                f"{values['max']:.4f} |"
            )
    lines.extend(
        [
            "",
            "These are descriptive three-seed statistics on an already "
            "inspected internal split. They do not constitute external blind "
            "validation or deployment-level uncertainty.",
            "",
            "The Qwen-derived sidecars remain image-conditioned soft sensor "
            "estimates, not measurements from independent physical hardware.",
            "",
        ]
    )
    _atomic_write(output_path.with_suffix(".md"), "\n".join(lines))
    return addendum


def main() -> None:
    args = parse_args()
    result = build_addendum(
        outcome_path=args.outcome,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "training_seeds": result["training_seeds"],
                "output": _resolve(args.output).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
