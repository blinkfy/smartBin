"""Select ensemble weights on validation predictions and evaluate the test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mmfc.ensemble import (
    PredictionTable,
    build_ensemble_result,
    load_prediction_table,
    select_validation_weights,
    selective_risk_profile,
    validate_aligned_tables,
    validation_locked_selective_points,
    weighted_probability_pool,
)
from mmfc.evaluation import (
    grouped_bootstrap_classification_intervals,
    grouped_paired_bootstrap_differences,
    write_evaluation_artifacts,
)
from mmfc.model import PROJECT_ROOT
from mmfc.training import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select probability-ensemble weights on validation predictions "
            "and apply the locked weights to test predictions"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--member",
        nargs=3,
        action="append",
        metavar=("NAME", "VAL_PREDICTIONS", "TEST_PREDICTIONS"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-member", type=str, default=None)
    parser.add_argument("--grid-denominator", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--paired-bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.member) < 2:
        raise ValueError("at least two --member entries are required")
    names = [entry[0] for entry in args.member]
    if len(set(names)) != len(names):
        raise ValueError("ensemble member names must be unique")
    if args.reference_member is not None and args.reference_member not in names:
        raise ValueError("--reference-member is not an ensemble member")
    if args.grid_denominator < 1:
        raise ValueError("--grid-denominator must be positive")
    for field in ("bootstrap_repeats", "paired_bootstrap_repeats"):
        value = int(getattr(args, field))
        if value < 2:
            raise ValueError(f"--{field.replace('_', '-')} must be at least 2")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence must lie in (0,1)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_names(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        value["name"] if isinstance(value, dict) else str(value)
        for value in config["classes"]
    )


def _load_members(
    entries: list[list[str]],
) -> tuple[list[str], list[PredictionTable], list[PredictionTable]]:
    names: list[str] = []
    validation: list[PredictionTable] = []
    test: list[PredictionTable] = []
    for name, validation_path, test_path in entries:
        names.append(str(name))
        validation.append(
            load_prediction_table(resolve_project_path(validation_path))
        )
        test.append(load_prediction_table(resolve_project_path(test_path)))
    validate_aligned_tables(validation)
    validate_aligned_tables(test)
    if validation[0].probabilities.shape[1] != test[0].probabilities.shape[1]:
        raise ValueError("validation and test class counts differ")
    return names, validation, test


def _report_markdown(summary: dict[str, Any]) -> str:
    selected = summary["selection"]["selected"]
    validation = summary["validation_metrics"]
    test = summary["test_metrics"]
    comparison = summary["paired_comparisons"]["reference_member"]
    accuracy = comparison["differences"]["accuracy"]
    macro_f1 = comparison["differences"]["macro_f1"]
    weights = ", ".join(
        f"{name}={weight:.2f}"
        for name, weight in zip(summary["member_names"], selected["weights"])
    )
    selective_rows = []
    for point in summary["selective_classification"]["locked_points"]:
        if not point["available"]:
            continue
        selective_rows.append(
            f"| {100 * point['target_validation_accuracy']:.1f} | "
            f"{point['threshold']:.3f} | "
            f"{100 * point['test']['coverage']:.2f} | "
            f"{100 * point['test']['accuracy']:.2f} |"
        )
    return "\n".join(
        [
            "# Validation-selected probability ensemble",
            "",
            "## Locked protocol",
            "",
            "- Weights were selected only on the validation split.",
            "- The fixed test split was not used for weight selection.",
            f"- Grid denominator: {summary['selection']['grid_denominator']}.",
            f"- Selected weights: `{weights}`.",
            "- Objective: validation Macro F1, then Accuracy, then NLL.",
            "",
            "## Results",
            "",
            "| Split | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | NLL | ECE (%) |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| Validation | {100 * validation['accuracy']:.2f} | "
                f"{100 * validation['macro_f1']:.2f} | "
                f"{100 * validation['weighted_f1']:.2f} | "
                f"{validation['negative_log_likelihood']:.3f} | "
                f"{100 * validation['ece']:.2f} |"
            ),
            (
                f"| Test | {100 * test['accuracy']:.2f} | "
                f"{100 * test['macro_f1']:.2f} | "
                f"{100 * test['weighted_f1']:.2f} | "
                f"{test['negative_log_likelihood']:.3f} | "
                f"{100 * test['ece']:.2f} |"
            ),
            "",
            "## Paired reference comparison",
            "",
            f"Reference member: `{summary['reference_member']}`.",
            "",
            (
                f"- Accuracy difference: {100 * accuracy['point']:+.2f} pp "
                f"[{100 * accuracy['lower']:+.2f}, "
                f"{100 * accuracy['upper']:+.2f}]."
            ),
            (
                f"- Macro F1 difference: {100 * macro_f1['point']:+.2f} pp "
                f"[{100 * macro_f1['lower']:+.2f}, "
                f"{100 * macro_f1['upper']:+.2f}]."
            ),
            "",
            "Intervals use paired source-image-cluster bootstrap resampling.",
            "",
            "## Selective classification",
            "",
            (
                "Test AURC: "
                f"{summary['selective_classification']['test']['aurc']:.5f}."
            ),
            "",
            "| Validation target accuracy (%) | Threshold | Test coverage (%) | Test accepted accuracy (%) |",
            "|---:|---:|---:|---:|",
            *selective_rows,
            "",
            (
                "Thresholds are locked on validation predictions. Rejected "
                "samples remain part of the full-coverage headline metrics."
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    config = load_config(args.config)
    class_names = _class_names(config)
    member_names, validation_tables, test_tables = _load_members(args.member)
    if validation_tables[0].probabilities.shape[1] != len(class_names):
        raise ValueError("prediction class count differs from config")

    selection = select_validation_weights(
        validation_tables,
        class_names,
        denominator=args.grid_denominator,
    )
    weights = selection["selected"]["weights"]
    validation_result, validation_probabilities = build_ensemble_result(
        validation_tables,
        member_names,
        weights,
        class_names,
    )
    test_result, test_probabilities = build_ensemble_result(
        test_tables,
        member_names,
        weights,
        class_names,
    )
    test_result["confidence_intervals"] = (
        grouped_bootstrap_classification_intervals(
            test_tables[0].targets,
            test_probabilities,
            test_tables[0].source_image_ids,
            class_names,
            repeats=args.bootstrap_repeats,
            confidence=args.bootstrap_confidence,
            seed=args.bootstrap_seed,
        )
    )

    reference_name = args.reference_member or member_names[0]
    reference_index = member_names.index(reference_name)
    paired_reference = grouped_paired_bootstrap_differences(
        test_tables[0].targets,
        test_tables[reference_index].probabilities,
        test_probabilities,
        test_tables[0].source_image_ids,
        class_names,
        repeats=args.paired_bootstrap_repeats,
        confidence=args.bootstrap_confidence,
        seed=args.bootstrap_seed + 1,
    )
    uniform_probabilities = weighted_probability_pool(
        test_tables,
        np.full(len(test_tables), 1.0 / len(test_tables)),
    )
    paired_uniform = grouped_paired_bootstrap_differences(
        test_tables[0].targets,
        uniform_probabilities,
        test_probabilities,
        test_tables[0].source_image_ids,
        class_names,
        repeats=args.paired_bootstrap_repeats,
        confidence=args.bootstrap_confidence,
        seed=args.bootstrap_seed + 2,
    )
    selective = {
        "validation": selective_risk_profile(
            validation_tables[0].targets,
            validation_probabilities,
        ),
        "test": selective_risk_profile(
            test_tables[0].targets,
            test_probabilities,
        ),
        "locked_points": validation_locked_selective_points(
            validation_tables[0].targets,
            validation_probabilities,
            test_tables[0].targets,
            test_probabilities,
        ),
        "headline_metrics_use_all_samples": True,
    }

    output = resolve_project_path(args.output)
    validation_artifacts = write_evaluation_artifacts(
        validation_result,
        output / "val",
        class_names=class_names,
        split="validation ensemble",
    )
    test_artifacts = write_evaluation_artifacts(
        test_result,
        output / "test",
        class_names=class_names,
        split="test ensemble",
    )
    bindings = []
    for name, validation_table, test_table in zip(
        member_names,
        validation_tables,
        test_tables,
    ):
        bindings.append(
            {
                "name": name,
                "validation_predictions": {
                    "path": str(validation_table.path),
                    "sha256": _sha256(validation_table.path),
                },
                "test_predictions": {
                    "path": str(test_table.path),
                    "sha256": _sha256(test_table.path),
                },
            }
        )
    summary = {
        "valid": True,
        "method": "validation_selected_weighted_probability_ensemble",
        "member_names": member_names,
        "reference_member": reference_name,
        "selection": selection,
        "validation_metrics": validation_result["metrics"],
        "test_metrics": test_result["metrics"],
        "test_confidence_intervals": test_result["confidence_intervals"],
        "paired_comparisons": {
            "reference_member": paired_reference,
            "uniform_ensemble": paired_uniform,
        },
        "selective_classification": selective,
        "input_bindings": bindings,
        "artifacts": {
            "validation": validation_artifacts,
            "test": test_artifacts,
        },
        "claim_boundary": (
            "This is a post-training three-checkpoint ensemble. It improves "
            "predictive performance at approximately three times single-model "
            "inference cost and does not replace the single-model seed analysis."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "ensemble_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = output / "REPORT.md"
    report_path.write_text(_report_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "selection": selection["selected"],
                "validation_metrics": summary["validation_metrics"],
                "test_metrics": summary["test_metrics"],
                "summary": str(summary_path.resolve()),
                "report": str(report_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
