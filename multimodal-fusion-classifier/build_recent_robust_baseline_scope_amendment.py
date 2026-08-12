"""Record the user-directed mixed-seed scope for recent baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_recent_robust_baselines import (
    AMENDED_METHOD_TRAINING_SEEDS,
    AMENDED_PROTOCOL_STATE,
    BOOTSTRAP_REPEATS,
    PRIMARY_PAIRED_CONDITIONS,
)
from evaluate_unseen_sensor_conditions import CONDITIONS
from mmfc.model import PROJECT_ROOT
from prepare_recent_robust_baselines import METHODS, SEEDS


REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "recent-robust-baselines"
)
DEFAULT_ORIGINAL_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_OUTPUT = REPORT_ROOT / "PROTOCOL_AMENDMENT.json"
INTERRUPTED_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "interrupted"
    / "tmc_seed43_epoch34_user_scope_reduction_20260730"
)
TRAINING_LOG_ROOT = PROJECT_ROOT / "reports" / "qwen-material" / "training"
SOURCE_FILES = {
    "scope_amendment_builder": Path(__file__),
    "baseline_outcome_audit": (
        PROJECT_ROOT / "audit_recent_robust_baselines.py"
    ),
    "confirmatory_addendum_builder": (
        PROJECT_ROOT / "build_plan_a_confirmatory_addendum.py"
    ),
    "manuscript_updater": (
        PROJECT_ROOT / "apply_plan_a_confirmatory_manuscript_update.py"
    ),
    "original_queue_runner": (
        TRAINING_LOG_ROOT / "run_recent_robust_baselines_after_unseen.ps1"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the transparent recent-baseline scope amendment"
    )
    parser.add_argument(
        "--original-protocol",
        type=Path,
        default=DEFAULT_ORIGINAL_PROTOCOL,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


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


def _metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


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
        "training": checkpoint / "training_summary.json",
        "best_checkpoint": checkpoint / "best.pt",
        "last_checkpoint": checkpoint / "last.pt",
        "test": report / "test" / "evaluation.json",
        "conditions": report / "conditions" / "evaluation.json",
    }


def _validate_original(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("valid") is not True
        or protocol.get("state") != "frozen_before_baseline_training"
        or protocol.get("methods") != list(METHODS)
        or protocol.get("training_seeds") != list(SEEDS)
        or protocol.get("run_count") != 6
        or protocol.get("conditions") != CONDITIONS
        or protocol.get("primary_paired_conditions")
        != list(PRIMARY_PAIRED_CONDITIONS)
        or protocol.get("bootstrap_repeats") != BOOTSTRAP_REPEATS
    ):
        raise ValueError("original recent-baseline protocol identity mismatch")


def build(
    original_protocol_path: Path = DEFAULT_ORIGINAL_PROTOCOL,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    original_protocol_path = _resolve(original_protocol_path)
    output_path = _resolve(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite baseline scope amendment")
    original = _read_json(original_protocol_path)
    _validate_original(original)

    missing_sources = [
        name for name, path in SOURCE_FILES.items() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "scope-amendment sources missing: "
            + ", ".join(missing_sources)
        )

    completed_runs: dict[str, Any] = {}
    for method, seeds in AMENDED_METHOD_TRAINING_SEEDS.items():
        completed_runs[method] = {}
        for seed in seeds:
            paths = _run_paths(method, seed)
            missing = [
                name for name, path in paths.items() if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{method} seed{seed} is incomplete: {', '.join(missing)}"
                )
            training = _read_json(paths["training"])
            if (
                training.get("valid") is not True
                or training.get("baseline_method") != method
                or training.get("seed") != seed
                or training.get("completed_epochs") != 100
                or training.get("resumed_from") is not None
            ):
                raise ValueError(f"{method} seed{seed} training is not valid")
            completed_runs[method][str(seed)] = {
                name: _binding(path)
                for name, path in paths.items()
                if name not in {"last_checkpoint"}
            }
            completed_runs[method][str(seed)]["last_checkpoint"] = _metadata(
                paths["last_checkpoint"]
            )

    partial_history = INTERRUPTED_ROOT / "history.json"
    partial_best = INTERRUPTED_ROOT / "best.pt"
    partial_last = INTERRUPTED_ROOT / "last.pt"
    partial_log = TRAINING_LOG_ROOT / "tmc_seed43_training.stdout.log"
    queue_status = TRAINING_LOG_ROOT / "recent_robust_baseline_status.json"
    required_partial = {
        "history": partial_history,
        "best": partial_best,
        "last": partial_last,
        "training_log": partial_log,
        "queue_status": queue_status,
    }
    missing_partial = [
        name for name, path in required_partial.items() if not path.is_file()
    ]
    if missing_partial:
        raise FileNotFoundError(
            "scope-amendment interruption evidence missing: "
            + ", ".join(missing_partial)
        )
    history = _read_json(partial_history)
    epochs = history.get("history")
    if (
        not isinstance(epochs, list)
        or len(epochs) != 34
        or epochs[0].get("epoch") != 1
        or epochs[-1].get("epoch") != 34
    ):
        raise ValueError("excluded TMC seed43 history identity mismatch")

    config_bindings = original.get("configuration_bindings", {})
    included_names = {
        f"{method}_seed{seed}"
        for method, seeds in AMENDED_METHOD_TRAINING_SEEDS.items()
        for seed in seeds
    }
    if not included_names.issubset(config_bindings):
        raise ValueError("included baseline config bindings are missing")

    report = {
        "schema_version": 1,
        "state": AMENDED_PROTOCOL_STATE,
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
        "training_seeds": list(SEEDS),
        "method_training_seeds": {
            method: list(values)
            for method, values in AMENDED_METHOD_TRAINING_SEEDS.items()
        },
        "run_count": 4,
        "original_run_count": 6,
        "completed_run_count": 4,
        "epochs_per_run": 100,
        "selection_metric": original["selection_metric"],
        "num_workers": original["num_workers"],
        "conditions": original["conditions"],
        "primary_paired_conditions": original[
            "primary_paired_conditions"
        ],
        "bootstrap_repeats": original["bootstrap_repeats"],
        "class_names": original["class_names"],
        "sample_counts": original["sample_counts"],
        "matched_settings": original["matched_settings"],
        "method_sources": original["method_sources"],
        "adaptation_contract": original["adaptation_contract"],
        "scope_reduction": {
            "reason": (
                "User directed the queue to stop additional long-running "
                "seeds because of compute cost on 2026-07-30."
            ),
            "qmf_reporting": (
                "three-seed confirmatory mean and sample standard deviation"
            ),
            "tmc_reporting": (
                "single-seed reference point estimate and source-image "
                "cluster bootstrap interval; sample SD is not applicable"
            ),
            "single_run_reported_as_replication": False,
            "partial_run_used_for_results": False,
            "unstarted_run_used_for_results": False,
        },
        "excluded_runs": {
            "tmc_seed43": {
                "state": "interrupted_after_user_scope_reduction",
                "completed_epochs": 34,
                "used_for_training_summary": False,
                "used_for_test": False,
                "used_for_condition_evaluation": False,
                "history": _binding(partial_history),
                "best_checkpoint": _metadata(partial_best),
                "last_checkpoint": _metadata(partial_last),
                "training_log": _binding(partial_log),
            },
            "tmc_seed44": {
                "state": "not_started_after_user_scope_reduction",
                "completed_epochs": 0,
                "used_for_results": False,
            },
        },
        "result_inclusion_rule": (
            "Report all completed QMF seeds 42/43/44 and completed TMC "
            "seed42. Label TMC as n=1 with no sample standard deviation; "
            "exclude the 34-epoch TMC seed43 and unstarted seed44."
        ),
        "configuration_bindings": {
            name: config_bindings[name] for name in sorted(included_names)
        },
        "completed_run_bindings": completed_runs,
        "original_protocol_binding": _binding(original_protocol_path),
        "interruption_status_binding": _binding(queue_status),
        "source_bindings": {
            name: _binding(path) for name, path in SOURCE_FILES.items()
        },
        "boundaries": [
            *original["boundaries"],
            (
                "A single TMC seed is reference evidence only and cannot be "
                "described as replicated or assigned a sample standard "
                "deviation."
            ),
            (
                "The interrupted TMC seed43 checkpoint is archived solely "
                "for provenance and is excluded from every result table."
            ),
        ],
        "errors": [],
    }
    _atomic_text(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(
        markdown_path,
        "\n".join(
            [
                "# Recent robust-fusion baseline scope amendment",
                "",
                "- Original frozen plan: QMF/TMC × seeds 42/43/44.",
                "- Included QMF runs: seeds 42/43/44 (n=3).",
                "- Included TMC run: seed42 only (n=1; SD not applicable).",
                "- TMC seed43 stopped after epoch 34 and is excluded.",
                "- TMC seed44 was not started and is excluded.",
                "- A single run is never represented as repeated evidence.",
                "",
            ]
        ),
    )
    return report


def main() -> None:
    args = parse_args()
    report = build(args.original_protocol, args.output)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "state": report["state"],
                "method_training_seeds": report[
                    "method_training_seeds"
                ],
                "partial_run_used_for_results": report[
                    "scope_reduction"
                ]["partial_run_used_for_results"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
