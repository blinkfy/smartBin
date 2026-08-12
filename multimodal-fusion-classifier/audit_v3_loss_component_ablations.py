"""Audit the frozen v3 auxiliary-loss component ablation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from prepare_v3_loss_component_ablations import SEEDS, VARIANTS


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-loss-components"
)
CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "confirmatory-ablations"
    / "v3_loss_components"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_OUTPUT = REPORT_ROOT / "OUTCOME.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit v3 auxiliary-loss component ablations"
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


def _valid_binding(binding: Mapping[str, Any]) -> bool:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    path = _resolve(Path(raw_path))
    return (
        path.is_file()
        and path.stat().st_size == binding.get("size_bytes")
        and _sha256(path) == binding.get("sha256")
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("state") != "frozen_before_component_training":
        errors.append("protocol state mismatch")
    if protocol.get("valid") is not True:
        errors.append("protocol is not valid")
    if protocol.get("scope") != "matched_seed_v3_auxiliary_loss_component_ablation":
        errors.append("protocol scope mismatch")
    training = protocol.get("training", {})
    if (
        training.get("matched_seeds") != list(SEEDS)
        or training.get("epochs") != 100
        or training.get("batch_size") != 16
        or training.get("num_workers") != 0
        or training.get("resume") is not False
    ):
        errors.append("training contract mismatch")
    if set(protocol.get("interventions", {})) != set(VARIANTS):
        errors.append("component intervention set mismatch")
    bindings = protocol.get("source_bindings", {})
    if not isinstance(bindings, Mapping) or not bindings:
        errors.append("source bindings missing")
    else:
        drifted = [
            name
            for name, binding in bindings.items()
            if not isinstance(binding, Mapping) or not _valid_binding(binding)
        ]
        if drifted:
            errors.append("source binding drift: " + ", ".join(drifted))
    return errors


def preflight(protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    protocol = _read_json(protocol_path)
    errors = validate_protocol(protocol)
    allowed_reports = {
        protocol_path.resolve(),
        protocol_path.with_suffix(".md").resolve(),
    }
    unexpected_reports = []
    if REPORT_ROOT.exists():
        unexpected_reports = [
            path
            for path in REPORT_ROOT.iterdir()
            if path.resolve() not in allowed_reports
        ]
    if unexpected_reports:
        errors.append("report root contains pre-training outputs")
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        errors.append("checkpoint root is not empty before training")
    return {
        "state": "preflight_complete",
        "valid": not errors,
        "protocol": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
        "variant_count": len(VARIANTS),
        "run_count": len(VARIANTS) * len(SEEDS),
        "errors": errors,
    }


def _finite_metric(document: Mapping[str, Any], name: str) -> float:
    value = document.get("aggregate", {}).get(name, {}).get("mean")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid aggregate metric: {name}")
    return float(value)


def audit_outcome(
    protocol_path: Path = DEFAULT_PROTOCOL,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    output_path = _resolve(output_path)
    protocol = _read_json(protocol_path)
    errors = validate_protocol(protocol)
    variants: dict[str, Any] = {}
    source_paths: dict[str, Path] = {"protocol": protocol_path}

    for variant in VARIANTS:
        runs: dict[str, Any] = {}
        for seed in SEEDS:
            checkpoint_dir = CHECKPOINT_ROOT / variant / f"seed{seed}"
            report_dir = REPORT_ROOT / variant / f"seed{seed}"
            required = {
                "best_checkpoint": checkpoint_dir / "best.pt",
                "last_checkpoint": checkpoint_dir / "last.pt",
                "history": checkpoint_dir / "history.json",
                "training_summary": checkpoint_dir / "training_summary.json",
                "checkpoint_audit": report_dir / "CHECKPOINT_AUDIT.json",
                "val_evaluation": report_dir / "val" / "evaluation.json",
                "test_evaluation": report_dir / "test" / "evaluation.json",
                "test_predictions": report_dir / "test" / "predictions.jsonl",
                "matched_comparison": report_dir / "MATCHED_TEST_COMPARISON.json",
                "training_exit": report_dir / "TRAINING_PROCESS_EXIT.json",
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                errors.append(
                    f"{variant} seed{seed} missing: " + ", ".join(missing)
                )
                continue
            summary = _read_json(required["training_summary"])
            history = _read_json(required["history"])
            checkpoint_audit = _read_json(required["checkpoint_audit"])
            validation = _read_json(required["val_evaluation"])
            test = _read_json(required["test_evaluation"])
            comparison = _read_json(required["matched_comparison"])
            process_exit = _read_json(required["training_exit"])
            history_rows = history.get("history", [])
            run_errors = []
            if (
                summary.get("completed_epochs") != 100
                or summary.get("resumed_from") is not None
                or summary.get("checkpoints_saved") is not True
                or not isinstance(history_rows, list)
                or len(history_rows) != 100
                or history_rows[-1].get("epoch") != 100
            ):
                run_errors.append("training bundle is not a fresh 100-epoch run")
            if (
                checkpoint_audit.get("valid") is not True
                or checkpoint_audit.get("last", {}).get("epoch") != 100
            ):
                run_errors.append("checkpoint audit failed")
            if (
                validation.get("metrics", {}).get("sample_count") != 826
                or test.get("metrics", {}).get("sample_count") != 826
            ):
                run_errors.append("evaluation sample count mismatch")
            paired = comparison.get("paired_bootstrap", {})
            if (
                paired.get("repeats") != 5000
                or paired.get("cluster_count") != 570
                or paired.get("sample_count") != 826
            ):
                run_errors.append("matched clustered bootstrap mismatch")
            if process_exit.get("complete_training_bundle") is not True:
                run_errors.append("training process exit lacks complete bundle")
            if run_errors:
                errors.extend(
                    f"{variant} seed{seed}: {message}" for message in run_errors
                )
            runs[str(seed)] = {
                "valid": not run_errors,
                "checkpoint_epoch": checkpoint_audit.get("best", {}).get("epoch"),
                "validation_metrics": validation.get("metrics"),
                "test_metrics": test.get("metrics"),
                "full_minus_ablation": {
                    name: values.get("point")
                    for name, values in paired.get("differences", {}).items()
                },
                "mcnemar_exact": comparison.get("mcnemar_exact"),
                "training_process_exit_code": process_exit.get("exit_code"),
            }
            for name, path in required.items():
                source_paths[f"{variant}_seed{seed}_{name}"] = path

        aggregate_paths = {
            "validation_summary": REPORT_ROOT / variant / "VALIDATION_SEED_SUMMARY.json",
            "test_summary": REPORT_ROOT / variant / "TEST_SEED_SUMMARY.json",
            "matched_summary": (
                REPORT_ROOT / variant / "MATCHED_TEST_SEED_SUMMARY.json"
            ),
            "robustness_comparison": (
                REPORT_ROOT / variant / "ROBUSTNESS_COMPARISON.json"
            ),
        }
        missing_aggregate = [
            name for name, path in aggregate_paths.items() if not path.is_file()
        ]
        if missing_aggregate:
            errors.append(
                f"{variant} aggregate missing: " + ", ".join(missing_aggregate)
            )
            continue
        validation_summary = _read_json(aggregate_paths["validation_summary"])
        test_summary = _read_json(aggregate_paths["test_summary"])
        matched_summary = _read_json(aggregate_paths["matched_summary"])
        robustness = _read_json(aggregate_paths["robustness_comparison"])
        if (
            validation_summary.get("valid") is not True
            or test_summary.get("valid") is not True
            or matched_summary.get("valid") is not True
            or robustness.get("valid") is not True
            or robustness.get("scenario_count") != 6
        ):
            errors.append(f"{variant} aggregate validation failed")
        variants[variant] = {
            "runs": runs,
            "ablation_test_mean": {
                "accuracy": _finite_metric(test_summary, "accuracy"),
                "macro_f1": _finite_metric(test_summary, "macro_f1"),
            },
            "full_minus_ablation_mean": {
                "accuracy": _finite_metric(matched_summary, "accuracy"),
                "macro_f1": _finite_metric(matched_summary, "macro_f1"),
            },
            "full_minus_ablation_seed_directions": {
                "accuracy_positive": (
                    matched_summary.get("aggregate", {})
                    .get("accuracy", {})
                    .get("positive_seed_count")
                ),
                "macro_f1_positive": (
                    matched_summary.get("aggregate", {})
                    .get("macro_f1", {})
                    .get("positive_seed_count")
                ),
            },
            "seed42_robustness_rows": robustness.get("rows"),
        }
        for name, path in aggregate_paths.items():
            source_paths[f"{variant}_{name}"] = path

    outcome = {
        "schema_version": 1,
        "state": "complete" if not errors else "failed",
        "valid": not errors,
        "scope": "matched_seed_v3_auxiliary_loss_component_ablation",
        "variants": variants,
        "claim_boundary": (
            "These independently retrained component ablations estimate the "
            "effects of the two auxiliary supervision terms on the reused "
            "internal benchmark. They do not establish external generalization "
            "or physical-sensor validity."
        ),
        "source_bindings": {
            name: {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sorted(source_paths.items())
            if path.is_file()
        },
        "errors": errors,
    }
    _atomic_write(
        output_path,
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# v3 auxiliary-loss component ablation outcome",
        "",
        f"- Status: {'PASS' if outcome['valid'] else 'FAIL'}.",
    ]
    if outcome["valid"]:
        lines.extend(
            [
                "",
                "| Variant | Ablation Accuracy | Ablation Macro F1 | Full-Ablation Accuracy (pp) | Full-Ablation Macro F1 (pp) |",
                "|---|---:|---:|---:|---:|",
                *[
                    (
                        f"| {variant} | "
                        f"{row['ablation_test_mean']['accuracy'] * 100:.2f} | "
                        f"{row['ablation_test_mean']['macro_f1'] * 100:.2f} | "
                        f"{row['full_minus_ablation_mean']['accuracy'] * 100:+.2f} | "
                        f"{row['full_minus_ablation_mean']['macro_f1'] * 100:+.2f} |"
                    )
                    for variant, row in variants.items()
                ],
            ]
        )
    else:
        lines.extend(["", *[f"- {error}" for error in errors]])
    lines.extend(
        [
            "",
            "Differences are matched full v3 minus ablation. The fixed internal "
            "test is reused and is not an external blind cohort.",
            "",
        ]
    )
    _atomic_write(output_path.with_suffix(".md"), "\n".join(lines))
    return outcome


def main() -> None:
    args = parse_args()
    if args.preflight:
        report = preflight(args.protocol)
    else:
        report = audit_outcome(args.protocol, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["valid"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
