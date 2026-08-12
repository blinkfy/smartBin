"""Preflight and outcome audit for matched QMF/TMC baseline adaptations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

import numpy as np

from evaluate_unseen_sensor_conditions import CONDITIONS
from mmfc.evaluation import grouped_paired_bootstrap_differences
from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path
from prepare_recent_robust_baselines import METHODS, SEEDS


REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "recent-robust-baselines"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_OUTPUT = REPORT_ROOT / "OUTCOME.json"
AMENDED_PROTOCOL_STATE = "amended_after_user_scope_reduction"
AMENDED_METHOD_TRAINING_SEEDS = {
    "qmf": (42, 43, 44),
    "tmc": (42,),
}
PRIMARY_PAIRED_CONDITIONS = (
    "clean",
    "noise_030_out_of_range",
    "spatial_permutation",
    "foreign_sample",
    "missing_false_available",
)
BOOTSTRAP_REPEATS = 2000
V3_CONDITION_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "unseen-sensor-conditions-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit matched recent robust-fusion baselines"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else resolve_project_path(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(value)
    return rows


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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _method_training_seeds(
    protocol: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    raw = protocol.get("method_training_seeds")
    if raw is None:
        return {method: tuple(SEEDS) for method in METHODS}
    if not isinstance(raw, Mapping) or set(raw) != set(METHODS):
        raise ValueError("recent baseline method/seed mapping mismatch")
    result = {}
    for method in METHODS:
        values = raw.get(method)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(seed, int) or seed not in SEEDS
                for seed in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"recent baseline seed mapping is invalid for {method}"
            )
        result[method] = tuple(values)
    return result


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    method_seeds = _method_training_seeds(protocol)
    state = protocol.get("state")
    common_valid = (
        protocol.get("valid") is True
        and protocol.get("methods") == list(METHODS)
        and protocol.get("training_seeds") == list(SEEDS)
        and protocol.get("run_count")
        == sum(len(values) for values in method_seeds.values())
        and protocol.get("conditions") == CONDITIONS
        and protocol.get("primary_paired_conditions")
        == list(PRIMARY_PAIRED_CONDITIONS)
        and protocol.get("bootstrap_repeats") == BOOTSTRAP_REPEATS
    )
    if state == "frozen_before_baseline_training":
        state_valid = all(
            values == tuple(SEEDS) for values in method_seeds.values()
        )
    elif state == AMENDED_PROTOCOL_STATE:
        state_valid = (
            method_seeds == AMENDED_METHOD_TRAINING_SEEDS
            and protocol.get("original_run_count") == 6
            and protocol.get("completed_run_count") == 4
            and protocol.get("scope_reduction", {}).get(
                "single_run_reported_as_replication"
            )
            is False
            and protocol.get("scope_reduction", {}).get(
                "partial_run_used_for_results"
            )
            is False
            and isinstance(protocol.get("original_protocol_binding"), Mapping)
        )
    else:
        state_valid = False
    if not common_valid or not state_valid:
        raise ValueError("recent baseline protocol identity mismatch")
    drifted = []
    for name, binding in protocol.get("source_bindings", {}).items():
        path = _resolve(Path(binding["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            drifted.append(name)
    if drifted:
        raise ValueError("recent baseline source drift: " + ", ".join(drifted))


def _run_paths(method: str, seed: int) -> dict[str, Path]:
    checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / "recent-robust-baselines"
        / method
        / f"seed{seed}"
    )
    report = REPORT_ROOT / method / f"seed{seed}"
    return {
        "checkpoint_dir": checkpoint,
        "best": checkpoint / "best.pt",
        "last": checkpoint / "last.pt",
        "training": checkpoint / "training_summary.json",
        "test": report / "test" / "evaluation.json",
        "test_predictions": report / "test" / "predictions.jsonl",
        "conditions": report / "conditions" / "evaluation.json",
        "condition_predictions": (
            report / "conditions" / "condition_predictions.jsonl"
        ),
    }


def preflight(protocol: Mapping[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    method_seeds = _method_training_seeds(protocol)
    occupied = []
    for method in METHODS:
        for seed in method_seeds[method]:
            paths = _run_paths(method, seed)
            checkpoint_dir = paths["checkpoint_dir"]
            report_dir = paths["test"].parents[1]
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                occupied.append(str(checkpoint_dir))
            if report_dir.exists() and any(report_dir.iterdir()):
                occupied.append(str(report_dir))
    if DEFAULT_OUTPUT.exists():
        occupied.append(str(DEFAULT_OUTPUT))
    if occupied:
        raise FileExistsError(
            "recent baseline preflight found occupied outputs: "
            + ", ".join(occupied)
        )
    return {
        "valid": True,
        "state": "preflight_passed",
        "methods": list(METHODS),
        "training_seeds": list(SEEDS),
        "method_training_seeds": {
            method: list(values) for method, values in method_seeds.items()
        },
        "run_count": sum(len(values) for values in method_seeds.values()),
    }


def _condition_rows(
    path: Path,
) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        condition = str(row["condition"])
        sample_id = str(row["sample_id"])
        key = f"{condition}\0{sample_id}"
        if key in grouped:
            raise ValueError(f"duplicate condition/sample row in {path}: {key}")
        grouped[key] = row
    return grouped


def _paired_comparison(
    reference_path: Path,
    candidate_path: Path,
    *,
    condition: str,
    class_names: list[str],
    seed: int,
) -> dict[str, Any]:
    reference = _condition_rows(reference_path)
    candidate = _condition_rows(candidate_path)
    keys = sorted(
        key
        for key in reference
        if key.startswith(condition + "\0")
    )
    if not keys or set(keys) != {
        key for key in candidate if key.startswith(condition + "\0")
    }:
        raise ValueError("paired condition prediction identity mismatch")
    targets = []
    reference_probabilities = []
    candidate_probabilities = []
    source_ids = []
    for key in keys:
        reference_row = reference[key]
        candidate_row = candidate[key]
        if (
            int(reference_row["target_id"]) != int(candidate_row["target_id"])
            or str(reference_row["source_image_id"])
            != str(candidate_row["source_image_id"])
        ):
            raise ValueError("paired condition target/source mismatch")
        targets.append(int(reference_row["target_id"]))
        reference_probabilities.append(reference_row["probabilities"])
        candidate_probabilities.append(candidate_row["probabilities"])
        source_ids.append(str(reference_row["source_image_id"]))
    return grouped_paired_bootstrap_differences(
        np.asarray(targets, dtype=np.int64),
        np.asarray(reference_probabilities, dtype=np.float64),
        np.asarray(candidate_probabilities, dtype=np.float64),
        source_ids,
        class_names,
        repeats=BOOTSTRAP_REPEATS,
        seed=seed,
    )


def _mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else None,
    }


def build_outcome(
    protocol: Mapping[str, Any],
    output_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    validate_protocol(protocol)
    method_seeds = _method_training_seeds(protocol)
    output_path = _resolve(output_path)
    protocol_path = _resolve(protocol_path)
    if output_path.exists() or output_path.with_suffix(".md").exists():
        raise FileExistsError("refusing to overwrite recent baseline outcome")
    class_names = list(protocol["class_names"])
    run_reports: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}

    for method_index, method in enumerate(METHODS):
        method_reports = {}
        seeds = method_seeds[method]
        for seed in seeds:
            paths = _run_paths(method, seed)
            missing = [
                name for name, path in paths.items()
                if name != "checkpoint_dir" and not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{method} seed{seed} missing: {', '.join(missing)}"
                )
            training = _read_json(paths["training"])
            test = _read_json(paths["test"])
            conditions = _read_json(paths["conditions"])
            if (
                training.get("valid") is not True
                or training.get("completed_epochs") != 100
                or training.get("baseline_method") != method
                or training.get("seed") != seed
                or test.get("dataset", {}).get("evaluated_samples") != 826
                or test.get("baseline", {}).get("method") != method
                or conditions.get("valid") is not True
                or conditions.get("baseline_method") != method
                or conditions.get("training_seed") != seed
                or conditions.get("sample_count") != 826
                or set(conditions.get("conditions", {})) != set(CONDITIONS)
            ):
                raise ValueError(f"{method} seed{seed} run audit failed")
            method_reports[str(seed)] = {
                "training": _binding(paths["training"]),
                "best_checkpoint": _binding(paths["best"]),
                "test": _binding(paths["test"]),
                "conditions": _binding(paths["conditions"]),
            }

            v3_predictions = (
                V3_CONDITION_ROOT / f"seed{seed}" / "predictions.jsonl"
            )
            if not v3_predictions.is_file():
                raise FileNotFoundError(
                    f"v3 unseen-condition predictions missing: {v3_predictions}"
                )
            for condition_index, condition in enumerate(
                PRIMARY_PAIRED_CONDITIONS
            ):
                key = f"{method}_seed{seed}_{condition}"
                paired[key] = _paired_comparison(
                    v3_predictions,
                    paths["condition_predictions"],
                    condition=condition,
                    class_names=class_names,
                    seed=(
                        20261210
                        + method_index * 10000
                        + seed * 100
                        + condition_index
                    ),
                )
        run_reports[method] = method_reports

        condition_evaluations = {
            seed: _read_json(_run_paths(method, seed)["conditions"])
            for seed in seeds
        }
        for condition in CONDITIONS:
            accuracies = [
                float(
                    condition_evaluations[seed]["conditions"][condition][
                        "metrics"
                    ]["accuracy"]
                )
                for seed in seeds
            ]
            macro_f1s = [
                float(
                    condition_evaluations[seed]["conditions"][condition][
                        "metrics"
                    ]["macro_f1"]
                )
                for seed in seeds
            ]
            sensor_weights = [
                float(
                    condition_evaluations[seed]["conditions"][condition][
                        "sensor_condition"
                    ]["sensor_branch_weight_mean"]
                )
                for seed in seeds
            ]
            summary_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "category": CONDITIONS[condition]["category"],
                    "training_seeds": list(seeds),
                    "seed_count": len(seeds),
                    "evidence_level": (
                        "confirmatory_multiseed"
                        if len(seeds) > 1
                        else "reference_single_seed"
                    ),
                    "accuracy_mean": _mean_std(accuracies)["mean"],
                    "accuracy_sample_std": _mean_std(accuracies)["sample_std"],
                    "macro_f1_mean": _mean_std(macro_f1s)["mean"],
                    "macro_f1_sample_std": _mean_std(macro_f1s)[
                        "sample_std"
                    ],
                    "sensor_branch_weight_mean": mean(sensor_weights),
                }
            )

    outcome = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": (
            "matched_same_input_qmf_three_seed_tmc_single_seed_reference"
        ),
        "methods": list(METHODS),
        "training_seeds": list(SEEDS),
        "method_training_seeds": {
            method: list(values) for method, values in method_seeds.items()
        },
        "run_count": sum(len(values) for values in method_seeds.values()),
        "sample_count_per_seed": 826,
        "conditions": CONDITIONS,
        "primary_paired_conditions": list(PRIMARY_PAIRED_CONDITIONS),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "summary": summary_rows,
        "paired_candidate_minus_v3": paired,
        "run_bindings": run_reports,
        "protocol_binding": _binding(protocol_path),
        "claim_boundary": (
            "QMF and TMC are transparent same-input adaptations using visual, "
            "visual_text candidate-prototype, and sensor decision branches. "
            "QMF is summarized over three seeds; TMC is a single-seed "
            "reference and no sample standard deviation or replication claim "
            "is made for TMC. "
            "They are not claimed as verbatim reproduction of the authors' "
            "original datasets or modality encoders."
        ),
        "errors": [],
    }
    _atomic_text(
        output_path,
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
    )
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0]),
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# Recent robust-fusion baseline outcome",
        "",
        "| Method | Condition | Accuracy (%) | Macro F1 (%) | Sensor branch weight |",
        "|---|---|---:|---:|---:|",
        *[
            (
                f"| {row['method']} | {row['condition']} | "
                f"{row['accuracy_mean'] * 100:.2f}"
                + (
                    f" ± {row['accuracy_sample_std'] * 100:.2f}"
                    if row["accuracy_sample_std"] is not None
                    else " (n=1; SD=N/A)"
                )
                + " | "
                f"{row['macro_f1_mean'] * 100:.2f}"
                + (
                    f" ± {row['macro_f1_sample_std'] * 100:.2f}"
                    if row["macro_f1_sample_std"] is not None
                    else " (n=1; SD=N/A)"
                )
                + " | "
                f"{row['sensor_branch_weight_mean']:.3f} |"
            )
            for row in summary_rows
        ],
        "",
        outcome["claim_boundary"],
        "",
    ]
    _atomic_text(output_path.with_suffix(".md"), "\n".join(lines))
    return outcome


def main() -> None:
    args = parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = _read_json(protocol_path)
    if args.preflight:
        result = preflight(protocol)
    else:
        result = build_outcome(protocol, args.output, protocol_path)
    print(
        json.dumps(
            {
                "valid": result["valid"],
                "state": result["state"],
                "methods": result["methods"],
                "training_seeds": result["training_seeds"],
                "method_training_seeds": result.get(
                    "method_training_seeds"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
