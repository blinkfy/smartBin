"""Freeze the v3 auxiliary-loss component ablation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from prepare_v3_loss_component_ablations import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_OUTPUT_DIR as CONFIG_ROOT,
    SEEDS,
    VARIANTS,
    normalized_component_config,
)


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
DEFAULT_OUTPUT = REPORT_ROOT / "PROTOCOL.json"
WORKERS4_INCIDENT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability"
    / "WORKERS4_STALL_INCIDENT.json"
)
WORKERS0_RECOVERY = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability-workers0"
    / "RECOVERY_PROTOCOL_ADDENDUM.json"
)
COMPARISON_SEEDS = {
    "v3_no_grid_reconstruction": {
        "42": 20260942,
        "43": 20260943,
        "44": 20260944,
    },
    "v3_no_ranking": {
        "42": 20261042,
        "43": 20261043,
        "44": 20261044,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the v3 loss-component ablation protocol"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
        "path": Path(os.path.relpath(resolved, PROJECT_ROOT)).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _full_reference_paths(seed: int) -> dict[str, Path]:
    if seed == 42:
        root = PROJECT_ROOT / "reports" / "qwen-material" / "full_cross_modal_quality"
        return {
            "config": DEFAULT_BASE_CONFIG,
            "checkpoint": (
                PROJECT_ROOT
                / "checkpoints"
                / "ablations"
                / "full_cross_modal_quality"
                / "best.pt"
            ),
            "checkpoint_audit": root / "CHECKPOINT_AUDIT.json",
            "val_evaluation": root / "val" / "evaluation.json",
            "test_evaluation": root / "test" / "evaluation.json",
            "test_predictions": root / "test" / "predictions.jsonl",
        }
    experiment = f"full_cross_modal_quality_seed{seed}"
    root = (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "seed-replication"
        / f"seed{seed}"
    )
    return {
        "config": PROJECT_ROOT / "configs" / "replicates" / f"{experiment}.json",
        "checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "v3_seed_replicates"
            / experiment
            / "best.pt"
        ),
        "checkpoint_audit": root / "CHECKPOINT_AUDIT.json",
        "val_evaluation": root / "val" / "evaluation.json",
        "test_evaluation": root / "test" / "evaluation.json",
        "test_predictions": root / "test" / "predictions.jsonl",
    }


def validate_component_config(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
) -> list[str]:
    specification = VARIANTS[variant]
    loss_field = str(specification["loss_field"])
    errors: list[str] = []
    if normalized_component_config(
        candidate,
        loss_field=loss_field,
    ) != normalized_component_config(base, loss_field=loss_field):
        errors.append("unauthorized scientific configuration differs")
    expected_name = f"{variant}_seed{seed}"
    expected_checkpoint = (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        f"{variant}/seed{seed}"
    )
    if candidate.get("loss", {}).get(loss_field) != 0.0:
        errors.append(f"loss.{loss_field} must be zero")
    for other_variant, other_specification in VARIANTS.items():
        other_field = str(other_specification["loss_field"])
        if other_variant != variant and candidate.get("loss", {}).get(other_field) != (
            base.get("loss", {}).get(other_field)
        ):
            errors.append(f"loss.{other_field} must remain unchanged")
    training = candidate.get("training", {})
    experiment = candidate.get("experiment", {})
    if (
        training.get("seed") != seed
        or training.get("num_workers") != 0
        or training.get("epochs") != 100
        or training.get("batch_size") != 16
        or Path(str(training.get("checkpoint_dir", ""))).as_posix()
        != expected_checkpoint
    ):
        errors.append("fixed training identity mismatch")
    if (
        experiment.get("name") != expected_name
        or experiment.get("ablation_of") != "full_cross_modal_quality"
        or experiment.get("ablation_scope") != specification["ablation_scope"]
        or experiment.get("matched_training_seed") != seed
        or experiment.get("intervention_field") != f"loss.{loss_field}"
    ):
        errors.append("experiment metadata mismatch")
    return errors


def build_protocol(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite the frozen protocol")
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("component-ablation checkpoint root is not empty")
    if REPORT_ROOT.exists() and any(REPORT_ROOT.iterdir()):
        raise ValueError("component-ablation report root is not empty")

    base = _read_json(DEFAULT_BASE_CONFIG)
    if (
        base.get("experiment", {}).get("name") != "full_cross_modal_quality"
        or base.get("training", {}).get("num_workers") != 0
        or base.get("training", {}).get("epochs") != 100
        or base.get("training", {}).get("batch_size") != 16
    ):
        raise ValueError("v3 base configuration identity mismatch")

    incident = _read_json(WORKERS4_INCIDENT)
    recovery = _read_json(WORKERS0_RECOVERY)
    if (
        incident.get("state") != "closed_incomplete_and_archived"
        or incident.get("facts", {}).get("last_completed_epoch") != 69
        or recovery.get("state") != "frozen_before_workers0_recovery"
        or recovery.get("recovery", {}).get("num_workers") != 0
    ):
        raise ValueError("long-horizon workers=0 policy evidence is invalid")

    config_checks: dict[str, Any] = {}
    sources: dict[str, Path] = {
        "protocol_builder": Path(__file__),
        "config_builder": (
            PROJECT_ROOT / "prepare_v3_loss_component_ablations.py"
        ),
        "outcome_auditor": (
            PROJECT_ROOT / "audit_v3_loss_component_ablations.py"
        ),
        "queue": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "training"
            / "run_v3_loss_component_ablations.ps1"
        ),
        "base_config": DEFAULT_BASE_CONFIG,
        "workers4_stall_incident": WORKERS4_INCIDENT,
        "workers0_recovery_protocol": WORKERS0_RECOVERY,
        "train_entrypoint": PROJECT_ROOT / "train.py",
        "evaluate_entrypoint": PROJECT_ROOT / "evaluate.py",
        "robustness_entrypoint": PROJECT_ROOT / "evaluate_robustness.py",
        "compare_entrypoint": PROJECT_ROOT / "compare_predictions.py",
        "summarize_evaluations": PROJECT_ROOT / "summarize_seed_evaluations.py",
        "summarize_comparisons": (
            PROJECT_ROOT / "summarize_matched_seed_comparisons.py"
        ),
        "model_source": PROJECT_ROOT / "mmfc" / "model.py",
        "loss_source": PROJECT_ROOT / "mmfc" / "losses.py",
        "training_source": PROJECT_ROOT / "mmfc" / "training.py",
        "evaluation_source": PROJECT_ROOT / "mmfc" / "evaluation.py",
        "train_manifest": PROJECT_ROOT / "data" / "qwen-material" / "train.jsonl",
        "val_manifest": PROJECT_ROOT / "data" / "qwen-material" / "val.jsonl",
        "test_manifest": PROJECT_ROOT / "data" / "qwen-material" / "test.jsonl",
        "sensor_manifest": (
            PROJECT_ROOT / "data" / "qwen-material" / "sensors.jsonl"
        ),
        "initial_yolo_weights": PROJECT_ROOT.parent / "yolo11s-seg.pt",
    }
    for variant in VARIANTS:
        config_checks[variant] = {}
        for seed in SEEDS:
            config_path = CONFIG_ROOT / f"{variant}_seed{seed}.json"
            candidate = _read_json(config_path)
            errors = validate_component_config(
                base,
                candidate,
                variant=variant,
                seed=seed,
            )
            config_checks[variant][str(seed)] = {
                "valid": not errors,
                "errors": errors,
            }
            if errors:
                raise ValueError(
                    f"{variant} seed {seed} config invalid: " + "; ".join(errors)
                )
            sources[f"{variant}_seed{seed}_config"] = config_path
    for seed in SEEDS:
        for label, path in _full_reference_paths(seed).items():
            sources[f"full_seed{seed}_{label}"] = path

    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("protocol sources missing: " + ", ".join(missing))
    for seed in SEEDS:
        reference = _full_reference_paths(seed)
        audit = _read_json(reference["checkpoint_audit"])
        validation = _read_json(reference["val_evaluation"])
        test = _read_json(reference["test_evaluation"])
        if (
            audit.get("valid") is not True
            or audit.get("last", {}).get("epoch") != 100
            or validation.get("metrics", {}).get("sample_count") != 826
            or test.get("metrics", {}).get("sample_count") != 826
        ):
            raise ValueError(f"full v3 seed {seed} reference is incomplete")

    protocol = {
        "schema_version": 1,
        "state": "frozen_before_component_training",
        "valid": True,
        "scope": "matched_seed_v3_auxiliary_loss_component_ablation",
        "scientific_questions": {
            "v3_no_grid_reconstruction": (
                "Does direct clean-grid reconstruction supervision contribute "
                "to v3 classification or registered degradation behavior?"
            ),
            "v3_no_ranking": (
                "Does pairwise reliability-ranking supervision contribute to "
                "v3 classification or registered degradation behavior?"
            ),
        },
        "interventions": {
            variant: {
                "field": f"loss.{specification['loss_field']}",
                "reference_value": base["loss"][specification["loss_field"]],
                "ablation_value": 0.0,
                "all_other_scientific_fields_unchanged": True,
            }
            for variant, specification in VARIANTS.items()
        },
        "training": {
            "matched_seeds": list(SEEDS),
            "epochs": 100,
            "batch_size": 16,
            "num_workers": 0,
            "resume": False,
            "selection_metric": "validation_accuracy",
            "test_used_for_description_not_model_selection": True,
            "run_order": [
                f"{variant}_seed{seed}"
                for variant in VARIANTS
                for seed in SEEDS
            ],
        },
        "worker_policy": {
            "workers4_input_equivalence_acknowledged": True,
            "workers4_long_horizon_recommendation_withdrawn": True,
            "reason": (
                "The prior workers=4 confirmatory run stalled after epoch 69; "
                "formal component runs use the original stable workers=0 path."
            ),
        },
        "evaluation": {
            "splits": ["val", "test"],
            "primary_metrics": ["accuracy", "macro_f1"],
            "secondary_metrics": [
                "macro_precision",
                "macro_recall",
                "weighted_f1",
                "nll",
                "brier",
                "ece",
            ],
            "test_cluster_bootstrap_repeats": 5000,
            "cluster_key": "source_image_id",
            "matched_comparison_seeds": COMPARISON_SEEDS,
            "seed42_registered_robustness_scenarios": [
                "clean",
                "missing_100",
                "noise_015",
                "bias_008",
                "stuck",
                "mixed",
            ],
            "degradation_seeds": [2026, 2027, 2028, 2029, 2030],
        },
        "claim_rules": [
            (
                "A component effect requires the independently retrained "
                "matched-seed comparison; same-checkpoint masking is not used."
            ),
            (
                "Three matched seeds describe training variation but do not "
                "turn the reused internal test into an external blind set."
            ),
            (
                "A clean classification difference cannot by itself establish "
                "reliability calibration; registered degradation diagnostics "
                "must be reported separately."
            ),
            (
                "The two component hypotheses are reported together, including "
                "null or adverse results; no test-driven hyperparameter changes "
                "are allowed."
            ),
        ],
        "config_checks": config_checks,
        "source_bindings": {
            name: _binding(path) for name, path in sorted(sources.items())
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# v3 auxiliary-loss component ablation protocol",
        "",
        "- Status: frozen before any component training.",
        "- Variants: `v3_no_grid_reconstruction`, `v3_no_ranking`.",
        "- Matched training seeds: 42, 43, 44.",
        "- Training: 100 epochs, batch 16, workers 0, validation-Accuracy selection.",
        "- Comparison: matched-seed full v3, source-image clustered bootstrap.",
        (
            "- Worker boundary: the byte-equivalent workers=4 short benchmark "
            "is retained, but its long-run recommendation was withdrawn after "
            "the epoch-69 stall."
        ),
        "",
        "Both null and adverse results must be retained. The fixed internal test "
        "remains non-blind and cannot support external deployment claims.",
        "",
    ]
    _atomic_write(markdown_path, "\n".join(lines))
    return protocol


def main() -> None:
    args = parse_args()
    protocol = build_protocol(args.output)
    print(
        json.dumps(
            {
                "state": protocol["state"],
                "valid": protocol["valid"],
                "variants": list(protocol["interventions"]),
                "seeds": protocol["training"]["matched_seeds"],
                "output": _resolve(args.output).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
