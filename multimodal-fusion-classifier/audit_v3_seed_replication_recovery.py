"""Audit the amended v3 seed-replication queue and its final outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_v3_seed_replication import (
    _summary_matches_evaluations,
    _validate_evaluation,
    validate_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "full_cross_modal_quality"
    / "seed-replication"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_ADDENDUM = REPORT_ROOT / "QUEUE_RECOVERY_ADDENDUM.json"
DEFAULT_OUTPUT = REPORT_ROOT / "OUTCOME.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the recovered v3 seed-replication experiment"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
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


def validate_recovery_addendum(
    *,
    protocol_path: Path,
    addendum_path: Path,
) -> dict[str, Any]:
    protocol_validation = validate_protocol(protocol_path)
    errors = list(protocol_validation["errors"])
    addendum_path = _resolve(addendum_path)
    if not addendum_path.is_file():
        errors.append("queue recovery addendum is missing")
        return {
            "valid": False,
            "errors": errors,
            "protocol": protocol_validation,
            "addendum": {},
            "addendum_path": addendum_path,
        }
    try:
        addendum = _read_json(addendum_path)
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"queue recovery addendum is unreadable: {error}")
        addendum = {}
    if (
        addendum.get("status")
        != "frozen_after_seed43_queue_false_negative_before_recovery"
        or addendum.get("valid") is not True
        or addendum.get("errors") != []
        or addendum.get("invariants", {}).get("seed43_training_rerun")
        is not False
        or addendum.get("invariants", {}).get("seed44_training_started")
        is not False
    ):
        errors.append("queue recovery addendum identity mismatch")
    protocol_binding = addendum.get("protocol", {})
    resolved_protocol = _resolve(protocol_path)
    if (
        protocol_binding.get("sha256") != _sha256(resolved_protocol)
        or protocol_binding.get("size_bytes") != resolved_protocol.stat().st_size
    ):
        errors.append("queue recovery addendum protocol binding mismatch")
    bindings = addendum.get("source_bindings", {})
    if not isinstance(bindings, Mapping) or not bindings:
        errors.append("queue recovery addendum source bindings are missing")
        bindings = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(f"malformed recovery source binding: {name}")
            continue
        raw_path = binding.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"malformed recovery source path: {name}")
            continue
        path = _resolve(Path(raw_path))
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"recovery source escapes project: {name}")
            continue
        if not path.is_file():
            errors.append(f"recovery source is missing: {name}")
            continue
        if (
            path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            errors.append(f"recovery source hash mismatch: {name}")
    return {
        "valid": not errors,
        "errors": errors,
        "protocol": protocol_validation,
        "addendum": addendum,
        "addendum_path": addendum_path,
    }


def _history_rows(document: Any) -> list[Mapping[str, Any]]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, Mapping)]
    if isinstance(document, Mapping):
        rows = document.get("history", [])
        if isinstance(rows, list):
            return [item for item in rows if isinstance(item, Mapping)]
    return []


def audit_recovered_replication(
    *,
    protocol_path: Path,
    addendum_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    preflight = validate_recovery_addendum(
        protocol_path=protocol_path,
        addendum_path=addendum_path,
    )
    errors = list(preflight["errors"])
    if errors:
        return {
            "schema_version": 1,
            "state": "invalid",
            "valid": False,
            "errors": errors,
        }
    protocol_path = _resolve(protocol_path)
    addendum_path = _resolve(addendum_path)
    report_root = protocol_path.parent
    checkpoint_root = (
        PROJECT_ROOT / "checkpoints" / "ablations" / "v3_seed_replicates"
    )
    base_root = (
        PROJECT_ROOT / "reports" / "qwen-material" / "full_cross_modal_quality"
    )
    evaluations: dict[str, dict[str, Mapping[str, Any]]] = {
        "validation": {
            "seed42": _read_json(base_root / "val" / "evaluation.json")
        },
        "test": {
            "seed42": _read_json(base_root / "test" / "evaluation.json")
        },
    }
    bindings = {
        "protocol": _binding(protocol_path),
        "recovery_addendum": _binding(addendum_path),
    }
    run_rows = []
    for seed in (43, 44):
        name = f"full_cross_modal_quality_seed{seed}"
        checkpoint_dir = checkpoint_root / name
        seed_report = report_root / f"seed{seed}"
        paths = {
            "training_summary": checkpoint_dir / "training_summary.json",
            "history": checkpoint_dir / "history.json",
            "best_checkpoint": checkpoint_dir / "best.pt",
            "last_checkpoint": checkpoint_dir / "last.pt",
            "checkpoint_audit": seed_report / "CHECKPOINT_AUDIT.json",
            "process_exit": seed_report / "TRAINING_PROCESS_EXIT.json",
            "val_evaluation": seed_report / "val" / "evaluation.json",
            "val_predictions": seed_report / "val" / "predictions.jsonl",
            "test_evaluation": seed_report / "test" / "evaluation.json",
            "test_predictions": seed_report / "test" / "predictions.jsonl",
        }
        missing = [label for label, path in paths.items() if not path.is_file()]
        if missing:
            errors.extend(
                f"seed {seed} artifact is missing: {label}" for label in missing
            )
            continue
        summary = _read_json(paths["training_summary"])
        history = _read_json(paths["history"])
        history_rows = _history_rows(history)
        checkpoint_audit = _read_json(paths["checkpoint_audit"])
        process_exit = _read_json(paths["process_exit"])
        best_sha = _sha256(paths["best_checkpoint"])
        last_sha = _sha256(paths["last_checkpoint"])
        if (
            summary.get("completed_epochs") != 100
            or summary.get("resumed_from") is not None
            or summary.get("checkpoints_saved") is not True
            or len(history_rows) != 100
            or history_rows[-1].get("epoch") != 100
            or checkpoint_audit.get("valid") is not True
            or checkpoint_audit.get("training_summary_completed_epochs") != 100
            or checkpoint_audit.get("history_json_rows") != 100
            or checkpoint_audit.get("last", {}).get("epoch") != 100
            or checkpoint_audit.get("best", {}).get("sha256") != best_sha
            or checkpoint_audit.get("last", {}).get("sha256") != last_sha
            or process_exit.get("seed") != seed
            or process_exit.get("exit_code") != 0
            or process_exit.get("zero_exit") is not True
            or process_exit.get("complete_training_bundle") is not True
        ):
            errors.append(f"seed {seed} recovered training bundle is invalid")
        if seed == 43 and (
            process_exit.get("training_rerun") is not False
            or process_exit.get("recovered_from_false_negative") is not True
        ):
            errors.append("seed 43 process-exit recovery disclosure is missing")
        if seed == 44 and process_exit.get("training_rerun") is not True:
            errors.append("seed 44 fresh-training disclosure is missing")
        best_epoch = checkpoint_audit.get("best", {}).get("epoch")
        val = _validate_evaluation(
            paths["val_evaluation"],
            label=f"seed {seed} validation",
            split="val",
            expected_epoch=best_epoch,
            errors=errors,
        )
        test = _validate_evaluation(
            paths["test_evaluation"],
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
                "best_epoch": best_epoch,
                "best_checkpoint_sha256": best_sha,
                "last_checkpoint_sha256": last_sha,
                "training_rerun_during_recovery": process_exit.get(
                    "training_rerun"
                ),
                "validation_accuracy": val.get("metrics", {}).get("accuracy"),
                "validation_macro_f1": val.get("metrics", {}).get("macro_f1"),
                "test_accuracy": test.get("metrics", {}).get("accuracy"),
                "test_macro_f1": test.get("metrics", {}).get("macro_f1"),
            }
        )
        for label, path in paths.items():
            bindings[f"seed{seed}_{label}"] = _binding(path)

    summaries = {}
    for split, filename in (
        ("validation", "VALIDATION_SEED_SUMMARY.json"),
        ("test", "TEST_SEED_SUMMARY.json"),
    ):
        path = report_root / filename
        csv_path = path.with_suffix(".csv")
        if not path.is_file() or not csv_path.is_file():
            errors.append(f"{split} seed summary or CSV is missing")
            continue
        summary = _read_json(path)
        if (
            summary.get("valid") is not True
            or summary.get("run_count") != 3
            or summary.get("sample_count_per_run") != 826
            or [item.get("name") for item in summary.get("individual", [])]
            != ["seed42", "seed43", "seed44"]
            or not _summary_matches_evaluations(summary, evaluations[split])
        ):
            errors.append(f"{split} seed summary mismatch")
        summaries[split] = summary
        bindings[f"{split}_summary"] = _binding(path)
        bindings[f"{split}_summary_csv"] = _binding(csv_path)
    if errors:
        return {
            "schema_version": 1,
            "state": "invalid",
            "valid": False,
            "errors": errors,
        }

    outcome = {
        "schema_version": 2,
        "state": "complete",
        "valid": True,
        "scope": "confirmatory_v3_training_seed_replication",
        "protocol": _binding(protocol_path),
        "queue_recovery_addendum": _binding(addendum_path),
        "training": {
            "seeds": [42, 43, 44],
            "new_replicate_seeds": [43, 44],
            "epochs_per_run": 100,
            "resume_used": False,
            "best_seed_replacement_performed": False,
            "seed43_training_reused_after_queue_false_negative": True,
            "runs": run_rows,
        },
        "validation_summary": summaries["validation"],
        "test_summary": summaries["test"],
        "boundaries": [
            "three seeds provide descriptive training-seed variation only",
            "the fixed internal test has already been inspected",
            "Qwen sidecars are image-conditioned soft sensors",
            "seed 42 remains the preregistered main model",
            "seed 43 was not retrained after its complete bundle was audited",
        ],
        "source_bindings": bindings,
        "errors": [],
    }
    output_path = _resolve(output_path)
    _atomic_write(
        output_path,
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
    )
    validation = summaries["validation"]["aggregate"]
    test = summaries["test"]["aggregate"]
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
                    f"{validation['accuracy']['mean']:.4f} | "
                    f"{validation['accuracy']['sample_std']:.4f} | "
                    f"{validation['accuracy']['min']:.4f} | "
                    f"{validation['accuracy']['max']:.4f} |"
                ),
                (
                    "| validation | Macro F1 | "
                    f"{validation['macro_f1']['mean']:.4f} | "
                    f"{validation['macro_f1']['sample_std']:.4f} | "
                    f"{validation['macro_f1']['min']:.4f} | "
                    f"{validation['macro_f1']['max']:.4f} |"
                ),
                (
                    "| test | Accuracy | "
                    f"{test['accuracy']['mean']:.4f} | "
                    f"{test['accuracy']['sample_std']:.4f} | "
                    f"{test['accuracy']['min']:.4f} | "
                    f"{test['accuracy']['max']:.4f} |"
                ),
                (
                    "| test | Macro F1 | "
                    f"{test['macro_f1']['mean']:.4f} | "
                    f"{test['macro_f1']['sample_std']:.4f} | "
                    f"{test['macro_f1']['min']:.4f} | "
                    f"{test['macro_f1']['max']:.4f} |"
                ),
                "",
                "Seed 43's complete 100-epoch bundle was retained after an "
                "independent checkpoint audit; it was not retrained.",
                "",
                "Seed 42 remains the preregistered main model. These statistics "
                "are descriptive on an inspected internal test.",
                "",
            ]
        ),
    )
    return outcome


def main() -> None:
    args = parse_args()
    if args.preflight:
        result = validate_recovery_addendum(
            protocol_path=args.protocol,
            addendum_path=args.addendum,
        )
        printable = {
            "valid": result["valid"],
            "error_count": len(result["errors"]),
            "errors": result["errors"],
        }
    else:
        result = audit_recovered_replication(
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
