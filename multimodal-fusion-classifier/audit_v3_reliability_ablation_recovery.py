"""Preflight and close the workers=0 reliability-ablation recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_v3_reliability_ablation import (
    COMPARISON_BOOTSTRAP_SEEDS,
    DEGRADATION_SEEDS,
    METRICS,
    PROJECT_ROOT,
    ROBUSTNESS_SCENARIOS,
    SEEDS,
    _history_rows,
    _prediction_count,
    _validate_evaluation,
    _validate_seed_summary,
    validate_protocol,
)
from prepare_v3_reliability_ablation_recovery import (
    BASE_CONFIG,
    validate_recovery_config,
)


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability-workers0"
)
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability"
    / "PROTOCOL.json"
)
DEFAULT_ADDENDUM = REPORT_ROOT / "RECOVERY_PROTOCOL_ADDENDUM.json"
DEFAULT_OUTPUT = REPORT_ROOT / "OUTCOME.json"
CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "confirmatory-ablations"
    / "v3_no_reliability_workers0"
)
CONFIGS = {
    seed: (
        PROJECT_ROOT
        / "configs"
        / "confirmatory-ablations"
        / "recovery-workers0"
        / f"v3_no_reliability_workers0_seed{seed}.json"
    )
    for seed in SEEDS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the workers=0 reliability-ablation recovery"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
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


def validate_recovery_addendum(
    protocol_path: Path,
    addendum_path: Path,
) -> dict[str, Any]:
    protocol_validation = validate_protocol(protocol_path)
    errors = list(protocol_validation["errors"])
    addendum_path = _resolve(addendum_path)
    if not addendum_path.is_file():
        errors.append("recovery protocol addendum is missing")
        return {
            "valid": False,
            "protocol": protocol_validation,
            "addendum_path": addendum_path,
            "document": {},
            "errors": errors,
        }
    try:
        addendum = _read_json(addendum_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        errors.append(f"recovery addendum is unreadable: {error}")
        addendum = {}
    if (
        addendum.get("schema_version") != 1
        or addendum.get("state") != "frozen_before_workers0_recovery"
        or addendum.get("valid") is not True
        or addendum.get("errors") != []
        or addendum.get("recovery", {}).get("training_seeds") != list(SEEDS)
        or addendum.get("recovery", {}).get("num_workers") != 0
        or addendum.get("recovery", {}).get("resume") is not False
        or addendum.get("recovery", {}).get("fresh_epoch1_runs") is not True
    ):
        errors.append("recovery addendum identity mismatch")
    bindings = addendum.get("source_bindings")
    if not isinstance(bindings, Mapping):
        errors.append("recovery addendum source bindings are missing")
        bindings = {}
    required = {
        "addendum_builder_source",
        "recovery_audit_source",
        "recovery_queue_source",
        "recovery_prepare_source",
        "original_protocol",
        "stall_incident",
        "stall_liveness",
        "partial_checkpoint_audit",
        "base_config",
        "input_pipeline_benchmark",
        "performance_provenance",
    }
    required.update(f"recovery_seed{seed}_config" for seed in SEEDS)
    for name in sorted(required - set(bindings)):
        errors.append(f"recovery source binding missing: {name}")
    for name, raw in bindings.items():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            errors.append(f"malformed recovery source binding: {name}")
            continue
        path = _resolve(Path(raw["path"]))
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"recovery source escapes project: {name}")
            continue
        if not path.is_file():
            errors.append(f"recovery source missing: {name}")
            continue
        if path.stat().st_size != raw.get("size_bytes"):
            errors.append(f"recovery source size mismatch: {name}")
        if _sha256(path) != raw.get("sha256"):
            errors.append(f"recovery source hash mismatch: {name}")
    original_binding = bindings.get("original_protocol", {})
    if isinstance(original_binding, Mapping):
        if _resolve(Path(str(original_binding.get("path", "")))) != (
            protocol_validation["protocol"]
        ):
            errors.append("recovery addendum binds a different original protocol")
    try:
        base = _read_json(BASE_CONFIG)
        for seed, path in CONFIGS.items():
            check = validate_recovery_config(
                base,
                _read_json(path),
                seed=seed,
            )
            errors.extend(
                f"seed {seed} recovery config: {message}"
                for message in check["errors"]
            )
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        errors.append(f"recovery configs are unreadable: {error}")
    return {
        "valid": not errors,
        "protocol": protocol_validation,
        "addendum_path": addendum_path,
        "document": addendum,
        "source_binding_count": len(bindings),
        "errors": errors,
    }


def _validate_training_bundle(
    seed: int,
    *,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    checkpoint_dir = CHECKPOINT_ROOT / f"seed{seed}"
    seed_report = REPORT_ROOT / f"seed{seed}"
    paths = {
        "training_summary": checkpoint_dir / "training_summary.json",
        "history": checkpoint_dir / "history.json",
        "best_checkpoint": checkpoint_dir / "best.pt",
        "last_checkpoint": checkpoint_dir / "last.pt",
        "checkpoint_audit": seed_report / "CHECKPOINT_AUDIT.json",
        "process_exit": seed_report / "TRAINING_PROCESS_EXIT.json",
        "val": seed_report / "val" / "evaluation.json",
        "val_predictions": seed_report / "val" / "predictions.jsonl",
        "test": seed_report / "test" / "evaluation.json",
        "test_predictions": seed_report / "test" / "predictions.jsonl",
        "comparison": seed_report / "MATCHED_TEST_COMPARISON.json",
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        errors.extend(f"seed {seed} recovery artifact missing: {x}" for x in missing)
        return {}, paths
    summary = _read_json(paths["training_summary"])
    try:
        rows = _history_rows(paths["history"])
    except ValueError as error:
        rows = []
        errors.append(f"seed {seed} recovery history invalid: {error}")
    checkpoint_audit = _read_json(paths["checkpoint_audit"])
    process_exit = _read_json(paths["process_exit"])
    if (
        summary.get("completed_epochs") != 100
        or summary.get("resumed_from") is not None
        or summary.get("checkpoints_saved") is not True
        or len(rows) != 100
        or (rows and rows[-1].get("epoch") != 100)
        or checkpoint_audit.get("valid") is not True
        or checkpoint_audit.get("training_summary_completed_epochs") != 100
        or checkpoint_audit.get("history_json_rows") != 100
        or checkpoint_audit.get("last", {}).get("epoch") != 100
        or process_exit.get("seed") != seed
        or process_exit.get("exit_code") != 0
        or process_exit.get("zero_exit") is not True
        or process_exit.get("complete_training_bundle") is not True
        or process_exit.get("num_workers") != 0
        or process_exit.get("workers4_partial_run_reused") is not False
    ):
        errors.append(f"seed {seed} recovery training bundle mismatch")
    best_epoch = checkpoint_audit.get("best", {}).get("epoch")
    if not isinstance(best_epoch, int):
        best_epoch = -1
        errors.append(f"seed {seed} recovery best epoch is invalid")
    validation = _validate_evaluation(
        paths["val"],
        split="val",
        expected_epoch=best_epoch,
        errors=errors,
    )
    test = _validate_evaluation(
        paths["test"],
        split="test",
        expected_epoch=best_epoch,
        errors=errors,
    )
    comparison = _read_json(paths["comparison"])
    paired = comparison.get("paired_bootstrap", {})
    if (
        comparison.get("reference", {}).get("name")
        != f"v3_no_reliability_workers0_seed{seed}"
        or comparison.get("candidate", {}).get("name")
        != f"full_cross_modal_quality_seed{seed}"
        or paired.get("repeats") != 5000
        or paired.get("sample_count") != 826
        or paired.get("cluster_count") != 570
        or paired.get("seed") != COMPARISON_BOOTSTRAP_SEEDS[str(seed)]
    ):
        errors.append(f"seed {seed} recovery comparison identity mismatch")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_checkpoint_sha256": checkpoint_audit.get("best", {}).get("sha256"),
        "last_checkpoint_sha256": checkpoint_audit.get("last", {}).get("sha256"),
        "validation_accuracy": validation.get("metrics", {}).get("accuracy"),
        "validation_macro_f1": validation.get("metrics", {}).get("macro_f1"),
        "test_accuracy": test.get("metrics", {}).get("accuracy"),
        "test_macro_f1": test.get("metrics", {}).get("macro_f1"),
        "process_exit_code": process_exit.get("exit_code"),
    }, paths


def audit_completed_recovery(
    *,
    protocol_path: Path,
    addendum_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    validation = validate_recovery_addendum(protocol_path, addendum_path)
    errors = list(validation["errors"])
    if errors:
        return {
            "schema_version": 1,
            "state": "invalid",
            "valid": False,
            "errors": errors,
        }
    protocol_path = validation["protocol"]["protocol"]
    addendum_path = validation["addendum_path"]
    source_bindings = {
        "original_protocol": _binding(protocol_path),
        "recovery_protocol_addendum": _binding(addendum_path),
    }
    runs = []
    for seed in SEEDS:
        run, paths = _validate_training_bundle(seed, errors=errors)
        if run:
            runs.append(run)
        for label, path in paths.items():
            if path.is_file():
                source_bindings[f"seed{seed}_{label}"] = _binding(path)

    val_summary_path = REPORT_ROOT / "VALIDATION_SEED_SUMMARY.json"
    test_summary_path = REPORT_ROOT / "TEST_SEED_SUMMARY.json"
    matched_path = REPORT_ROOT / "MATCHED_TEST_SEED_SUMMARY.json"
    val_summary = _validate_seed_summary(val_summary_path, errors=errors)
    test_summary = _validate_seed_summary(test_summary_path, errors=errors)
    matched = _read_json(matched_path) if matched_path.is_file() else {}
    if (
        matched.get("valid") is not True
        or matched.get("run_count") != 3
        or [row.get("name") for row in matched.get("individual", [])]
        != ["seed42", "seed43", "seed44"]
    ):
        errors.append("recovery matched-test seed summary mismatch")
    for metric in METRICS:
        aggregate = matched.get("aggregate", {}).get(metric, {})
        if (
            not isinstance(aggregate.get("mean"), (int, float))
            or sum(
                int(aggregate.get(name, -100))
                for name in (
                    "positive_seed_count",
                    "negative_seed_count",
                    "zero_seed_count",
                )
            )
            != 3
        ):
            errors.append(f"recovery matched aggregate invalid: {metric}")

    robustness_summary_path = (
        REPORT_ROOT
        / "seed42"
        / "robustness"
        / "robustness_multiseed_summary.json"
    )
    robustness = (
        _read_json(robustness_summary_path)
        if robustness_summary_path.is_file()
        else {}
    )
    robustness_rows = {
        row.get("scenario"): row for row in robustness.get("summary", [])
    }
    if (
        robustness.get("seeds") != DEGRADATION_SEEDS
        or robustness.get("seed_count") != 5
        or any(
            robustness_rows.get(scenario, {}).get("seed_count") != 5
            for scenario in ROBUSTNESS_SCENARIOS
        )
    ):
        errors.append("recovery seed42 robustness summary mismatch")
    robustness_comparison_path = REPORT_ROOT / "ROBUSTNESS_COMPARISON.json"
    robustness_comparison = (
        _read_json(robustness_comparison_path)
        if robustness_comparison_path.is_file()
        else {}
    )
    if (
        robustness_comparison.get("valid") is not True
        or robustness_comparison.get("training_seed") != 42
        or robustness_comparison.get("degradation_seeds") != DEGRADATION_SEEDS
        or [row.get("scenario") for row in robustness_comparison.get("rows", [])]
        != ROBUSTNESS_SCENARIOS
    ):
        errors.append("recovery robustness comparison mismatch")

    generated = {
        "validation_seed_summary": val_summary_path,
        "validation_seed_summary_csv": val_summary_path.with_suffix(".csv"),
        "test_seed_summary": test_summary_path,
        "test_seed_summary_csv": test_summary_path.with_suffix(".csv"),
        "matched_test_seed_summary": matched_path,
        "matched_test_seed_summary_csv": matched_path.with_suffix(".csv"),
        "robustness_summary": robustness_summary_path,
        "robustness_comparison": robustness_comparison_path,
        "robustness_comparison_csv": robustness_comparison_path.with_suffix(".csv"),
        "robustness_comparison_md": robustness_comparison_path.with_suffix(".md"),
    }
    for label, path in generated.items():
        if not path.is_file():
            errors.append(f"recovery generated artifact missing: {label}")
        else:
            source_bindings[label] = _binding(path)
    if errors:
        return {
            "schema_version": 1,
            "state": "invalid",
            "valid": False,
            "errors": errors,
        }

    incident_path = (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "confirmatory-ablations"
        / "v3-no-reliability"
        / "WORKERS4_STALL_INCIDENT.json"
    )
    source_bindings["workers4_stall_incident"] = _binding(incident_path)
    outcome = {
        "schema_version": 2,
        "state": "complete",
        "valid": True,
        "scope": "matched_seed_v3_entire_reliability_subsystem_ablation",
        "recovery": {
            "workers4_stall_disclosed": True,
            "workers4_partial_run_reused": False,
            "workers0_fresh_epoch1_runs": True,
            "original_protocol_sha256": _sha256(protocol_path),
            "recovery_addendum_sha256": _sha256(addendum_path),
        },
        "training": {
            "seeds": list(SEEDS),
            "epochs_per_run": 100,
            "resume_used": False,
            "num_workers": 0,
            "runs": runs,
        },
        "ablation_validation_summary": val_summary,
        "ablation_test_summary": test_summary,
        "matched_test_comparison": matched,
        "robustness_ablation_summary": robustness,
        "robustness_comparison": robustness_comparison,
        "boundaries": [
            "This removes the entire sensor-reliability subsystem, not only a gate.",
            "The fixed internal test has already been inspected.",
            "Qwen sidecars are image-conditioned soft-sensor observations.",
            "Degradation seeds are perturbation repeats, not training replicates.",
            (
                "The workers=4 partial seed42 run stalled after epoch 69, was "
                "archived, and was not resumed or used in scientific results."
            ),
        ],
        "source_bindings": source_bindings,
        "errors": [],
    }
    output_path = _resolve(output_path)
    _atomic_write(
        output_path,
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
    )
    accuracy = matched["aggregate"]["accuracy"]
    macro_f1 = matched["aggregate"]["macro_f1"]
    ece = matched["aggregate"]["ece"]
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# Matched-seed v3 reliability ablation recovery outcome",
                "",
                "| Metric (full minus ablation) | Mean | Sample SD | Full favorable seeds |",
                "|---|---:|---:|---:|",
                (
                    f"| Accuracy | {accuracy['mean']:+.6f} | "
                    f"{accuracy['sample_std']:.6f} | "
                    f"{accuracy['positive_seed_count']}/3 |"
                ),
                (
                    f"| Macro F1 | {macro_f1['mean']:+.6f} | "
                    f"{macro_f1['sample_std']:.6f} | "
                    f"{macro_f1['positive_seed_count']}/3 |"
                ),
                (
                    f"| ECE | {ece['mean']:+.6f} | {ece['sample_std']:.6f} | "
                    f"{ece['negative_seed_count']}/3 |"
                ),
                "",
                "All three scientific runs started from epoch 1 with workers=0. "
                "The archived workers=4 partial run was not resumed or reused.",
                "",
            ]
        ),
    )
    return outcome


def main() -> None:
    args = parse_args()
    if args.preflight:
        result = validate_recovery_addendum(args.protocol, args.addendum)
        printable = {
            "valid": result["valid"],
            "source_binding_count": result.get("source_binding_count", 0),
            "error_count": len(result["errors"]),
            "protocol": result["protocol"]["protocol"].as_posix(),
            "addendum": result["addendum_path"].as_posix(),
        }
    else:
        result = audit_completed_recovery(
            protocol_path=args.protocol,
            addendum_path=args.addendum,
            output_path=args.output,
        )
        printable = {
            "state": result["state"],
            "valid": result["valid"],
            "error_count": len(result.get("errors", [])),
            "output": _resolve(args.output).as_posix(),
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if result.get("valid") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
