"""Freeze a confirmatory three-training-seed replication of the v3 model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "full_cross_modal_quality"
    / "seed-replication"
)
DEFAULT_OUTPUT = REPORT_ROOT / "PROTOCOL.json"
BASE_CONFIG = PROJECT_ROOT / "configs" / "ablations" / "full_cross_modal_quality.json"
REPLICATE_CONFIGS = {
    43: (
        PROJECT_ROOT
        / "configs"
        / "replicates"
        / "full_cross_modal_quality_seed43.json"
    ),
    44: (
        PROJECT_ROOT
        / "configs"
        / "replicates"
        / "full_cross_modal_quality_seed44.json"
    ),
}
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "ablations" / "v3_seed_replicates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the v3 confirmatory seed-replication protocol"
    )
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
        "path": Path(os.path.relpath(resolved, PROJECT_ROOT)).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def normalized_replicate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the fields authorized to differ between training seeds."""

    value = copy.deepcopy(dict(config))
    training = value.setdefault("training", {})
    training.pop("seed", None)
    training.pop("checkpoint_dir", None)
    experiment = value.setdefault("experiment", {})
    experiment.pop("name", None)
    experiment.pop("description", None)
    experiment.pop("replicate_of", None)
    return value


def validate_replicate_config(
    base: Mapping[str, Any],
    replicate: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    base_name = str(base["experiment"]["name"])
    expected_name = f"{base_name}_seed{seed}"
    expected_checkpoint = (
        Path("checkpoints")
        / "ablations"
        / "v3_seed_replicates"
        / expected_name
    ).as_posix()
    errors = []
    if normalized_replicate_config(base) != normalized_replicate_config(
        replicate
    ):
        errors.append("non-seed scientific configuration differs")
    if replicate.get("training", {}).get("seed") != seed:
        errors.append("training seed mismatch")
    if (
        Path(str(replicate.get("training", {}).get("checkpoint_dir", ""))).as_posix()
        != expected_checkpoint
    ):
        errors.append("checkpoint path mismatch")
    experiment = replicate.get("experiment", {})
    if experiment.get("name") != expected_name:
        errors.append("experiment name mismatch")
    if experiment.get("replicate_of") != base_name:
        errors.append("replicate_of mismatch")
    return {
        "valid": not errors,
        "seed": seed,
        "experiment": expected_name,
        "checkpoint_dir": expected_checkpoint,
        "errors": errors,
    }


def build_protocol(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    if output_path.exists() or output_path.with_suffix(".md").exists():
        raise FileExistsError("refusing to overwrite seed-replication protocol")
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("seed-replication checkpoint root is not empty")
    if REPORT_ROOT.exists():
        unexpected = [
            path
            for path in REPORT_ROOT.iterdir()
            if path.resolve() not in {output_path, output_path.with_suffix(".md")}
        ]
        if unexpected:
            raise ValueError("seed-replication report root is not empty")

    base = _read_json(BASE_CONFIG)
    if (
        base.get("experiment", {}).get("name") != "full_cross_modal_quality"
        or base.get("training", {}).get("seed") != 42
        or base.get("training", {}).get("epochs") != 100
    ):
        raise ValueError("base v3 config identity mismatch")
    config_checks = {}
    for seed, path in REPLICATE_CONFIGS.items():
        check = validate_replicate_config(
            base,
            _read_json(path),
            seed=seed,
        )
        config_checks[str(seed)] = check
        if not check["valid"]:
            raise ValueError(
                f"seed {seed} config is not an exact seed replicate: "
                + ", ".join(check["errors"])
            )

    base_root = (
        PROJECT_ROOT / "reports" / "qwen-material" / "full_cross_modal_quality"
    )
    base_summary = _read_json(
        PROJECT_ROOT
        / "checkpoints"
        / "ablations"
        / "full_cross_modal_quality"
        / "training_summary.json"
    )
    base_audit = _read_json(base_root / "CHECKPOINT_AUDIT.json")
    base_val = _read_json(base_root / "val" / "evaluation.json")
    base_test = _read_json(base_root / "test" / "evaluation.json")
    if (
        base_summary.get("completed_epochs") != 100
        or base_summary.get("resumed_from") is not None
        or base_audit.get("valid") is not True
        or base_audit.get("best", {}).get("epoch") != 92
        or base_val.get("metrics", {}).get("sample_count") != 826
        or base_test.get("metrics", {}).get("sample_count") != 826
    ):
        raise ValueError("base seed-42 evidence is incomplete")

    sources = {
        "builder_source": Path(__file__),
        "audit_source": PROJECT_ROOT / "audit_v3_seed_replication.py",
        "queue_source": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "training"
            / "run_v3_seed_replication.ps1"
        ),
        "prepare_source": PROJECT_ROOT / "prepare_seed_replicates.py",
        "summary_source": PROJECT_ROOT / "summarize_seed_evaluations.py",
        "train_entrypoint": PROJECT_ROOT / "train.py",
        "evaluate_entrypoint": PROJECT_ROOT / "evaluate.py",
        "checkpoint_audit_entrypoint": PROJECT_ROOT / "audit_checkpoints.py",
        "model_source": PROJECT_ROOT / "mmfc" / "model.py",
        "loss_source": PROJECT_ROOT / "mmfc" / "losses.py",
        "training_source": PROJECT_ROOT / "mmfc" / "training.py",
        "data_source": PROJECT_ROOT / "mmfc" / "data.py",
        "base_config": BASE_CONFIG,
        "seed43_config": REPLICATE_CONFIGS[43],
        "seed44_config": REPLICATE_CONFIGS[44],
        "train_manifest": PROJECT_ROOT / "data" / "qwen-material" / "train.jsonl",
        "val_manifest": PROJECT_ROOT / "data" / "qwen-material" / "val.jsonl",
        "test_manifest": PROJECT_ROOT / "data" / "qwen-material" / "test.jsonl",
        "sensor_manifest": PROJECT_ROOT / "data" / "qwen-material" / "sensors.jsonl",
        "yolo_initial_weights": PROJECT_ROOT.parent / "yolo11s-seg.pt",
        "base_training_summary": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "full_cross_modal_quality"
            / "training_summary.json"
        ),
        "base_history": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "full_cross_modal_quality"
            / "history.json"
        ),
        "base_best_checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "full_cross_modal_quality"
            / "best.pt"
        ),
        "base_checkpoint_audit": base_root / "CHECKPOINT_AUDIT.json",
        "base_val_evaluation": base_root / "val" / "evaluation.json",
        "base_test_evaluation": base_root / "test" / "evaluation.json",
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "seed-replication protocol sources are missing: "
            + ", ".join(missing)
        )

    protocol = {
        "schema_version": 1,
        "status": "frozen_before_replicate_training",
        "valid": True,
        "scope": "confirmatory_v3_training_seed_replication",
        "base": {
            "experiment": "full_cross_modal_quality",
            "seed": 42,
            "completed_epochs": 100,
            "best_epoch": 92,
            "validation_sample_count": 826,
            "test_sample_count": 826,
        },
        "replicates": {
            "seeds": [43, 44],
            "all_training_seeds": [42, 43, 44],
            "config_checks": config_checks,
        },
        "invariants": {
            "epochs": 100,
            "batch_size": 16,
            "optimizer_and_schedule_unchanged": True,
            "model_and_loss_unchanged": True,
            "data_and_class_order_unchanged": True,
            "only_seed_paths_and_metadata_change": True,
            "resume": False,
            "best_checkpoint_selected_by_validation_accuracy": True,
            "test_used_for_description_not_seed_selection": True,
            "best_seed_replacement_forbidden": True,
        },
        "evaluation": {
            "splits": ["validation", "test"],
            "bootstrap_repeats_per_run": 2000,
            "aggregate_metrics": [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "negative_log_likelihood",
                "multiclass_brier_score",
                "ece",
            ],
            "aggregate_statistics": ["mean", "sample_std", "min", "max"],
            "primary_interpretation": (
                "three-seed descriptive stability; no hypothesis test and no "
                "post-hoc model selection"
            ),
        },
        "process_exit_policy": {
            "zero_exit_preferred": True,
            "nonzero_exit_may_continue_only_if": [
                "training_summary_completed_epochs_equals_100",
                "history_has_100_rows",
                "best_and_last_exist",
                "independent_checkpoint_audit_passes",
            ],
            "nonzero_exit_must_be_disclosed": True,
        },
        "source_bindings": {
            name: _binding(path) for name, path in sources.items()
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# v3 confirmatory training-seed replication protocol",
                "",
                "- Frozen before seeds 43 and 44 begin training.",
                "- Seed 42 remains the preregistered main model; no seed may "
                "replace it after test inspection.",
                "- Replicates change only training seed, isolated output paths, "
                "and experiment metadata.",
                "- Validation and fixed internal test metrics are summarized as "
                "three-seed mean, sample standard deviation, minimum, and maximum.",
                "- The fixed test remains an inspected internal benchmark, not an "
                "external blind set.",
                "",
            ]
        ),
    )
    return {
        "state": "complete",
        "valid": True,
        "output": output_path.as_posix(),
        "source_binding_count": len(sources),
        "replicate_seeds": [43, 44],
    }


def main() -> None:
    result = build_protocol(parse_args().output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
