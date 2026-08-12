"""Preflight and close the matched-seed v3 reliability ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from build_v3_reliability_ablation_protocol import (
    COMPARISON_BOOTSTRAP_SEEDS,
    CONFIGS,
    DEGRADATION_SEEDS,
    PROJECT_ROOT,
    REPORT_ROOT,
    ROBUSTNESS_SCENARIOS,
    SEEDS,
    validate_ablation_config,
)


DEFAULT_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_OUTPUT = REPORT_ROOT / "OUTCOME.json"
METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "negative_log_likelihood",
    "multiclass_brier_score",
    "ece",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the matched-seed v3 reliability ablation"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
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


def _path_allowed(name: str, path: Path) -> bool:
    try:
        path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return (
            name == "yolo_initial_weights"
            and path == (PROJECT_ROOT.parent / "yolo11s-seg.pt").resolve()
        )


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    errors: list[str] = []
    if not protocol_path.is_file():
        return {
            "valid": False,
            "protocol": protocol_path,
            "document": {},
            "source_binding_count": 0,
            "errors": ["protocol is missing"],
        }
    try:
        protocol = _read_json(protocol_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {
            "valid": False,
            "protocol": protocol_path,
            "document": {},
            "source_binding_count": 0,
            "errors": [f"protocol is unreadable: {error}"],
        }
    if (
        protocol.get("schema_version") != 1
        or protocol.get("state") != "frozen_before_ablation_training"
        or protocol.get("valid") is not True
        or protocol.get("errors") != []
        or protocol.get("training", {}).get("matched_seeds") != list(SEEDS)
        or protocol.get("training", {}).get("resume") is not False
        or protocol.get("intervention", {}).get("scope")
        != "entire_sensor_reliability_subsystem"
        or protocol.get("robustness_evaluation", {}).get("scenarios")
        != ROBUSTNESS_SCENARIOS
        or protocol.get("robustness_evaluation", {}).get("degradation_seeds")
        != DEGRADATION_SEEDS
    ):
        errors.append("protocol identity or frozen settings mismatch")
    bindings = protocol.get("source_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        errors.append("protocol source bindings are missing")
        bindings = {}
    for name, raw in bindings.items():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            errors.append(f"malformed source binding: {name}")
            continue
        path = _resolve(Path(raw["path"]))
        if not _path_allowed(str(name), path):
            errors.append(f"source binding escapes allowed roots: {name}")
            continue
        if not path.is_file():
            errors.append(f"bound source is missing: {name}")
            continue
        if path.stat().st_size != raw.get("size_bytes"):
            errors.append(f"bound source size mismatch: {name}")
        if _sha256(path) != raw.get("sha256"):
            errors.append(f"bound source hash mismatch: {name}")
    required = {
        "builder_source",
        "audit_source",
        "queue_source",
        "prepare_source",
        "paper_addendum_source",
        "matched_summary_source",
        "robustness_comparison_source",
        "train_entrypoint",
        "evaluate_entrypoint",
        "robustness_entrypoint",
        "prediction_comparison_entrypoint",
        "checkpoint_audit_entrypoint",
        "base_config",
        "performance_provenance",
        "train_manifest",
        "val_manifest",
        "test_manifest",
        "sensor_manifest",
        "full_seed42_robustness",
    }
    for seed in SEEDS:
        required.update(
            {
                f"ablation_seed{seed}_config",
                f"full_seed{seed}_config",
                f"full_seed{seed}_checkpoint",
                f"full_seed{seed}_audit",
                f"full_seed{seed}_val",
                f"full_seed{seed}_val_predictions",
                f"full_seed{seed}_test",
                f"full_seed{seed}_test_predictions",
            }
        )
    for name in sorted(required - set(bindings)):
        errors.append(f"required source binding is missing: {name}")
    base_binding = bindings.get("base_config")
    if isinstance(base_binding, Mapping) and isinstance(base_binding.get("path"), str):
        try:
            base = _read_json(_resolve(Path(base_binding["path"])))
            for seed in SEEDS:
                binding = bindings.get(f"ablation_seed{seed}_config")
                if not isinstance(binding, Mapping):
                    continue
                candidate = _read_json(_resolve(Path(str(binding["path"]))))
                check = validate_ablation_config(base, candidate, seed=seed)
                errors.extend(
                    f"seed {seed} config: {message}" for message in check["errors"]
                )
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
            errors.append(f"bound configs are unreadable: {error}")
    return {
        "valid": not errors,
        "protocol": protocol_path,
        "document": protocol,
        "source_binding_count": len(bindings),
        "errors": errors,
    }


def _history_rows(path: Path) -> list[Mapping[str, Any]]:
    document = _read_json(path)
    rows = document.get("history")
    if not isinstance(rows, list):
        raise ValueError("history.json root must contain a history list")
    return [row for row in rows if isinstance(row, Mapping)]


def _prediction_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def _validate_evaluation(
    path: Path,
    *,
    split: str,
    expected_epoch: int,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing {split} evaluation: {path}")
        return {}
    document = _read_json(path)
    metrics = document.get("metrics", {})
    if (
        metrics.get("sample_count") != 826
        or len(document.get("class_names", [])) != 19
        or document.get("dataset", {}).get("split") != split
        or document.get("checkpoint", {}).get("epoch") != expected_epoch
        or document.get("evaluation_config", {}).get("bootstrap_repeats") != 2000
        or document.get("evaluation_config", {}).get("bootstrap_seed") != 20260727
    ):
        errors.append(f"{split} evaluation identity mismatch: {path}")
    for metric in METRICS:
        if not isinstance(metrics.get(metric), (int, float)):
            errors.append(f"{split} evaluation metric missing: {metric}")
    predictions = path.parent / "predictions.jsonl"
    if not predictions.is_file() or _prediction_count(predictions) != 826:
        errors.append(f"{split} prediction count mismatch: {predictions}")
    return document


def _validate_training_bundle(
    seed: int,
    *,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    checkpoint_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "confirmatory-ablations"
        / "v3_no_reliability"
        / f"seed{seed}"
    )
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
        errors.extend(f"seed {seed} artifact missing: {label}" for label in missing)
        return {}, paths
    summary = _read_json(paths["training_summary"])
    try:
        rows = _history_rows(paths["history"])
    except ValueError as error:
        rows = []
        errors.append(f"seed {seed} history invalid: {error}")
    audit = _read_json(paths["checkpoint_audit"])
    process_exit = _read_json(paths["process_exit"])
    if (
        summary.get("completed_epochs") != 100
        or summary.get("resumed_from") is not None
        or summary.get("checkpoints_saved") is not True
        or len(rows) != 100
        or (rows and rows[-1].get("epoch") != 100)
        or audit.get("valid") is not True
        or audit.get("training_summary_completed_epochs") != 100
        or audit.get("history_json_rows") != 100
        or audit.get("last", {}).get("epoch") != 100
        or process_exit.get("seed") != seed
        or process_exit.get("exit_code") != 0
        or process_exit.get("zero_exit") is not True
        or process_exit.get("complete_training_bundle") is not True
    ):
        errors.append(f"seed {seed} training bundle identity mismatch")
    best_epoch = audit.get("best", {}).get("epoch")
    if not isinstance(best_epoch, int):
        errors.append(f"seed {seed} best checkpoint epoch is invalid")
        best_epoch = -1
    validation = _validate_evaluation(
        paths["val"], split="val", expected_epoch=best_epoch, errors=errors
    )
    test = _validate_evaluation(
        paths["test"], split="test", expected_epoch=best_epoch, errors=errors
    )
    comparison = _read_json(paths["comparison"])
    paired = comparison.get("paired_bootstrap", {})
    if (
        comparison.get("reference", {}).get("name")
        != f"v3_no_reliability_seed{seed}"
        or comparison.get("candidate", {}).get("name")
        != f"full_cross_modal_quality_seed{seed}"
        or paired.get("repeats") != 5000
        or paired.get("sample_count") != 826
        or paired.get("cluster_count") != 570
        or paired.get("seed") != COMPARISON_BOOTSTRAP_SEEDS[str(seed)]
    ):
        errors.append(f"seed {seed} matched comparison identity mismatch")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_checkpoint_sha256": audit.get("best", {}).get("sha256"),
        "validation_accuracy": validation.get("metrics", {}).get("accuracy"),
        "validation_macro_f1": validation.get("metrics", {}).get("macro_f1"),
        "test_accuracy": test.get("metrics", {}).get("accuracy"),
        "test_macro_f1": test.get("metrics", {}).get("macro_f1"),
        "process_exit_code": process_exit.get("exit_code"),
    }, paths


def _validate_seed_summary(
    path: Path,
    *,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"seed summary is missing: {path}")
        return {}
    document = _read_json(path)
    if (
        document.get("valid") is not True
        or document.get("run_count") != 3
        or document.get("sample_count_per_run") != 826
        or [row.get("name") for row in document.get("individual", [])]
        != ["seed42", "seed43", "seed44"]
    ):
        errors.append(f"seed summary identity mismatch: {path}")
    if not path.with_suffix(".csv").is_file():
        errors.append(f"seed summary CSV missing: {path.with_suffix('.csv')}")
    return document


def audit_completed_ablation(
    *,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    validation = validate_protocol(protocol_path)
    errors = list(validation["errors"])
    protocol_path = validation["protocol"]
    if errors:
        return {
            "schema_version": 1,
            "state": "invalid",
            "valid": False,
            "errors": errors,
        }
    source_bindings: dict[str, dict[str, Any]] = {
        "protocol": _binding(protocol_path)
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
        errors.append("matched-test seed summary identity mismatch")
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
            errors.append(f"matched-test aggregate invalid: {metric}")
    if not matched_path.with_suffix(".csv").is_file():
        errors.append("matched-test seed summary CSV missing")

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
        errors.append("seed42 robustness summary identity mismatch")
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
        errors.append("robustness comparison identity mismatch")

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
            errors.append(f"generated artifact missing: {label}")
        else:
            source_bindings[label] = _binding(path)
    if errors:
        return {
            "schema_version": 1,
            "state": "invalid",
            "valid": False,
            "errors": errors,
        }

    outcome = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "matched_seed_v3_entire_reliability_subsystem_ablation",
        "protocol": {
            "path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(protocol_path),
        },
        "training": {
            "seeds": list(SEEDS),
            "epochs_per_run": 100,
            "resume_used": False,
            "num_workers": 4,
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
                "# Matched-seed v3 reliability-subsystem ablation outcome",
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
                "Positive Accuracy/Macro F1 differences favor full v3. Negative "
                "ECE differences favor full v3.",
                "",
                "The clean matched-seed result is primary. Seed-42 degradation "
                "results are supporting descriptive evidence.",
                "",
            ]
        ),
    )
    return outcome


def main() -> None:
    args = parse_args()
    if args.preflight:
        result = validate_protocol(args.protocol)
        printable = {
            "valid": result["valid"],
            "source_binding_count": result["source_binding_count"],
            "error_count": len(result["errors"]),
            "protocol": result["protocol"].as_posix(),
        }
    else:
        result = audit_completed_ablation(
            protocol_path=args.protocol,
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
