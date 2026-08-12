"""Preflight and close the confirmatory v3 training-seed replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from build_v3_seed_replication_protocol import validate_replicate_config


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "full_cross_modal_quality"
    / "seed-replication"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_OUTPUT = REPORT_ROOT / "OUTCOME.json"
REQUIRED_PROTOCOL_BINDINGS = (
    "builder_source",
    "audit_source",
    "queue_source",
    "prepare_source",
    "summary_source",
    "train_entrypoint",
    "evaluate_entrypoint",
    "checkpoint_audit_entrypoint",
    "model_source",
    "loss_source",
    "training_source",
    "data_source",
    "base_config",
    "seed43_config",
    "seed44_config",
    "train_manifest",
    "val_manifest",
    "test_manifest",
    "sensor_manifest",
    "yolo_initial_weights",
    "base_training_summary",
    "base_history",
    "base_best_checkpoint",
    "base_checkpoint_audit",
    "base_val_evaluation",
    "base_test_evaluation",
)
AGGREGATE_METRICS = (
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
        description="Audit the v3 confirmatory training-seed replication"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
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


def _bound_path(
    bindings: Mapping[str, Any],
    name: str,
    errors: list[str],
) -> Path | None:
    binding = bindings.get(name)
    if not isinstance(binding, Mapping):
        errors.append(f"protocol source binding is missing: {name}")
        return None
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"protocol source path is malformed: {name}")
        return None
    return _resolve(Path(raw_path))


def _path_is_allowed(name: str, path: Path) -> bool:
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
    except (OSError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "protocol": protocol_path,
            "document": {},
            "source_binding_count": 0,
            "errors": [f"protocol is unreadable: {error}"],
        }
    if (
        protocol.get("schema_version") != 1
        or protocol.get("status") != "frozen_before_replicate_training"
        or protocol.get("valid") is not True
        or protocol.get("errors") != []
        or protocol.get("replicates", {}).get("seeds") != [43, 44]
        or protocol.get("replicates", {}).get("all_training_seeds")
        != [42, 43, 44]
    ):
        errors.append("protocol identity or seed set mismatch")
    bindings = protocol.get("source_bindings", {})
    if not isinstance(bindings, Mapping):
        bindings = {}
        errors.append("protocol source bindings are missing")
    for name in REQUIRED_PROTOCOL_BINDINGS:
        if name not in bindings:
            errors.append(f"protocol source binding is missing: {name}")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(f"malformed protocol source binding: {name}")
            continue
        raw_path = binding.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"protocol source path is malformed: {name}")
            continue
        path = _resolve(Path(raw_path))
        if not _path_is_allowed(name, path):
            errors.append(f"protocol source escapes project: {name}")
            continue
        if not path.is_file():
            errors.append(f"protocol source is missing: {name}")
            continue
        if path.stat().st_size != binding.get("size_bytes"):
            errors.append(f"protocol source size mismatch: {name}")
        if _sha256(path) != binding.get("sha256"):
            errors.append(f"protocol source hash mismatch: {name}")
    base_path = _bound_path(bindings, "base_config", errors)
    replicate_paths = {
        seed: _bound_path(bindings, f"seed{seed}_config", errors)
        for seed in (43, 44)
    }
    if base_path is not None and all(replicate_paths.values()):
        try:
            base = _read_json(base_path)
            for seed, replicate_path in replicate_paths.items():
                assert replicate_path is not None
                check = validate_replicate_config(
                    base,
                    _read_json(replicate_path),
                    seed=seed,
                )
                errors.extend(
                    f"seed {seed} config: {message}"
                    for message in check["errors"]
                )
        except (OSError, json.JSONDecodeError, KeyError) as error:
            errors.append(f"replicate configs are unreadable: {error}")
    return {
        "valid": not errors,
        "protocol": protocol_path,
        "document": protocol,
        "source_binding_count": len(bindings),
        "errors": errors,
    }


def _validate_evaluation(
    path: Path,
    *,
    label: str,
    split: str,
    expected_epoch: int | None,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"{label} evaluation is missing")
        return {}
    document = _read_json(path)
    metrics = document.get("metrics", {})
    if (
        metrics.get("sample_count") != 826
        or not isinstance(metrics.get("accuracy"), (int, float))
        or not isinstance(metrics.get("macro_f1"), (int, float))
        or len(document.get("class_names", [])) != 19
    ):
        errors.append(f"{label} evaluation identity mismatch")
    if (
        document.get("dataset", {}).get("split") != split
        or document.get("evaluation_config", {}).get("bootstrap_repeats")
        != 2000
        or document.get("checkpoint", {}).get("epoch") != expected_epoch
    ):
        errors.append(f"{label} protocol settings mismatch")
    predictions = path.parent / "predictions.jsonl"
    if not predictions.is_file():
        errors.append(f"{label} predictions are missing")
    else:
        with predictions.open("r", encoding="utf-8-sig") as handle:
            prediction_count = sum(1 for line in handle if line.strip())
        if prediction_count != 826:
            errors.append(f"{label} prediction count mismatch")
    return document


def _summary_matches_evaluations(
    summary: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> bool:
    individuals = summary.get("individual", [])
    if not isinstance(individuals, list):
        return False
    by_name = {
        str(item.get("name")): item
        for item in individuals
        if isinstance(item, Mapping)
    }
    if list(by_name) != ["seed42", "seed43", "seed44"]:
        return False
    for name, evaluation in evaluations.items():
        item = by_name.get(name, {})
        expected_metrics = evaluation.get("metrics", {})
        if item.get("sample_count") != 826:
            return False
        item_metrics = item.get("metrics", {})
        if not isinstance(item_metrics, Mapping):
            return False
        if set(item_metrics) != set(AGGREGATE_METRICS):
            return False
        for metric_name in AGGREGATE_METRICS:
            if expected_metrics.get(metric_name) != item_metrics.get(
                metric_name
            ):
                return False
    return True


def audit_completed_replication(
    *,
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    validation = validate_protocol(protocol_path)
    errors = list(validation["errors"])
    protocol_path = validation["protocol"]
    protocol = validation["document"]
    if errors:
        return {
            "schema_version": 1,
            "state": "invalid",
            "valid": False,
            "errors": errors,
        }
    report_root = protocol_path.parent
    checkpoint_root = (
        PROJECT_ROOT / "checkpoints" / "ablations" / "v3_seed_replicates"
    )
    source_bindings: dict[str, dict[str, Any]] = {
        "protocol": _binding(protocol_path),
    }
    run_rows: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Mapping[str, Any]]] = {
        "validation": {
            "seed42": _read_json(
                PROJECT_ROOT
                / "reports"
                / "qwen-material"
                / "full_cross_modal_quality"
                / "val"
                / "evaluation.json"
            )
        },
        "test": {
            "seed42": _read_json(
                PROJECT_ROOT
                / "reports"
                / "qwen-material"
                / "full_cross_modal_quality"
                / "test"
                / "evaluation.json"
            )
        },
    }
    for seed in (43, 44):
        name = f"full_cross_modal_quality_seed{seed}"
        checkpoint_dir = checkpoint_root / name
        summary_path = checkpoint_dir / "training_summary.json"
        history_path = checkpoint_dir / "history.json"
        audit_path = report_root / f"seed{seed}" / "CHECKPOINT_AUDIT.json"
        exit_path = report_root / f"seed{seed}" / "TRAINING_PROCESS_EXIT.json"
        val_path = report_root / f"seed{seed}" / "val" / "evaluation.json"
        test_path = report_root / f"seed{seed}" / "test" / "evaluation.json"
        required = (
            summary_path,
            history_path,
            checkpoint_dir / "best.pt",
            checkpoint_dir / "last.pt",
            audit_path,
            exit_path,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            errors.extend(
                f"seed {seed} artifact is missing: {path}" for path in missing
            )
            continue
        summary = _read_json(summary_path)
        history = _read_json(history_path)
        checkpoint_audit = _read_json(audit_path)
        process_exit = _read_json(exit_path)
        if (
            summary.get("completed_epochs") != 100
            or summary.get("resumed_from") is not None
            or len(history) != 100
            or checkpoint_audit.get("valid") is not True
            or checkpoint_audit.get("training_summary_completed_epochs") != 100
            or checkpoint_audit.get("history_json_rows") != 100
            or checkpoint_audit.get("last", {}).get("epoch") != 100
            or process_exit.get("complete_training_bundle") is not True
            or process_exit.get("seed") != seed
            or not isinstance(process_exit.get("exit_code"), int)
            or process_exit.get("zero_exit")
            is not (process_exit.get("exit_code") == 0)
        ):
            errors.append(f"seed {seed} training bundle is invalid")
        best_epoch = checkpoint_audit.get("best", {}).get("epoch")
        val = _validate_evaluation(
            val_path,
            label=f"seed {seed} validation",
            split="val",
            expected_epoch=best_epoch,
            errors=errors,
        )
        test = _validate_evaluation(
            test_path,
            label=f"seed {seed} test",
            split="test",
            expected_epoch=best_epoch,
            errors=errors,
        )
        evaluations["validation"][f"seed{seed}"] = val
        evaluations["test"][f"seed{seed}"] = test
        run_rows.append(
            {
                "seed": seed,
                "process_exit_code": process_exit.get("exit_code"),
                "best_epoch": checkpoint_audit.get("best", {}).get("epoch"),
                "best_checkpoint_sha256": checkpoint_audit.get("best", {}).get(
                    "sha256"
                ),
                "validation_accuracy": val.get("metrics", {}).get("accuracy"),
                "validation_macro_f1": val.get("metrics", {}).get("macro_f1"),
                "test_accuracy": test.get("metrics", {}).get("accuracy"),
                "test_macro_f1": test.get("metrics", {}).get("macro_f1"),
            }
        )
        for label, path in (
            (f"seed{seed}_training_summary", summary_path),
            (f"seed{seed}_history", history_path),
            (f"seed{seed}_checkpoint_audit", audit_path),
            (f"seed{seed}_process_exit", exit_path),
            (f"seed{seed}_best_checkpoint", checkpoint_dir / "best.pt"),
            (f"seed{seed}_last_checkpoint", checkpoint_dir / "last.pt"),
            (f"seed{seed}_val", val_path),
            (
                f"seed{seed}_val_predictions",
                val_path.parent / "predictions.jsonl",
            ),
            (f"seed{seed}_test", test_path),
            (
                f"seed{seed}_test_predictions",
                test_path.parent / "predictions.jsonl",
            ),
        ):
            if path.is_file():
                source_bindings[label] = _binding(path)

    validation_summary_path = report_root / "VALIDATION_SEED_SUMMARY.json"
    test_summary_path = report_root / "TEST_SEED_SUMMARY.json"
    for label, path in (
        ("validation_seed_summary", validation_summary_path),
        ("test_seed_summary", test_summary_path),
    ):
        if not path.is_file():
            errors.append(f"{label} is missing")
            continue
        summary = _read_json(path)
        if (
            summary.get("valid") is not True
            or summary.get("run_count") != 3
            or summary.get("sample_count_per_run") != 826
            or [item.get("name") for item in summary.get("individual", [])]
            != ["seed42", "seed43", "seed44"]
        ):
            errors.append(f"{label} identity mismatch")
        split = "validation" if label.startswith("validation") else "test"
        if not _summary_matches_evaluations(summary, evaluations[split]):
            errors.append(f"{label} does not match source evaluations")
        source_bindings[label] = _binding(path)
        csv_path = path.with_suffix(".csv")
        if not csv_path.is_file():
            errors.append(f"{label} CSV is missing")
        else:
            source_bindings[f"{label}_csv"] = _binding(csv_path)

    if errors:
        return {
            "schema_version": 1,
            "state": "invalid",
            "valid": False,
            "errors": errors,
        }
    validation_summary = _read_json(validation_summary_path)
    test_summary = _read_json(test_summary_path)
    outcome = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "confirmatory_v3_training_seed_replication",
        "protocol": {
            "path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(protocol_path),
        },
        "training": {
            "seeds": [42, 43, 44],
            "new_replicate_seeds": [43, 44],
            "epochs_per_run": 100,
            "resume_used": False,
            "best_seed_replacement_performed": False,
            "runs": run_rows,
        },
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "boundaries": [
            "three seeds provide descriptive training-seed variation only",
            "the fixed internal test has already been inspected",
            "Qwen sidecars are image-conditioned soft sensors",
            "seed 42 remains the preregistered main model",
        ],
        "source_bindings": source_bindings,
        "errors": [],
    }
    output_path = _resolve(output_path)
    _atomic_write(
        output_path,
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
    )
    validation_metrics = validation_summary["aggregate"]
    test_metrics = test_summary["aggregate"]
    nonzero_exits = [
        row for row in run_rows if row["process_exit_code"] != 0
    ]
    exit_disclosure = (
        "All new replicate training processes returned exit code 0."
        if not nonzero_exits
        else "Nonzero training exits accepted after independent bundle audit: "
        + ", ".join(
            f"seed {row['seed']} (exit {row['process_exit_code']})"
            for row in nonzero_exits
        )
        + "."
    )
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# v3 confirmatory training-seed replication outcome",
                "",
                "| Split | Metric | Mean | Sample SD | Min | Max |",
                "|---|---|---:|---:|---:|---:|",
                (
                    "| validation | Accuracy | "
                    f"{validation_metrics['accuracy']['mean']:.4f} | "
                    f"{validation_metrics['accuracy']['sample_std']:.4f} | "
                    f"{validation_metrics['accuracy']['min']:.4f} | "
                    f"{validation_metrics['accuracy']['max']:.4f} |"
                ),
                (
                    "| validation | Macro F1 | "
                    f"{validation_metrics['macro_f1']['mean']:.4f} | "
                    f"{validation_metrics['macro_f1']['sample_std']:.4f} | "
                    f"{validation_metrics['macro_f1']['min']:.4f} | "
                    f"{validation_metrics['macro_f1']['max']:.4f} |"
                ),
                (
                    "| test | Accuracy | "
                    f"{test_metrics['accuracy']['mean']:.4f} | "
                    f"{test_metrics['accuracy']['sample_std']:.4f} | "
                    f"{test_metrics['accuracy']['min']:.4f} | "
                    f"{test_metrics['accuracy']['max']:.4f} |"
                ),
                (
                    "| test | Macro F1 | "
                    f"{test_metrics['macro_f1']['mean']:.4f} | "
                    f"{test_metrics['macro_f1']['sample_std']:.4f} | "
                    f"{test_metrics['macro_f1']['min']:.4f} | "
                    f"{test_metrics['macro_f1']['max']:.4f} |"
                ),
                "",
                "Seed 42 remains the preregistered main model. These three-seed "
                "statistics are descriptive and do not convert the inspected "
                "internal test into an external blind benchmark.",
                "",
                exit_disclosure,
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
            key: value
            for key, value in result.items()
            if key not in {"document", "protocol"}
        }
        printable["protocol"] = result["protocol"].as_posix()
    else:
        result = audit_completed_replication(
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
