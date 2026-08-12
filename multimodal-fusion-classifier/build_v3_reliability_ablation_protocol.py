"""Freeze the matched-seed v3 reliability-subsystem ablation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from prepare_v3_reliability_ablation import (
    LOSS_REMOVALS,
    MODEL_REMOVALS,
    SEEDS,
    normalized_reliability_ablation_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability"
)
DEFAULT_OUTPUT = REPORT_ROOT / "PROTOCOL.json"
BASE_CONFIG = PROJECT_ROOT / "configs" / "ablations" / "full_cross_modal_quality.json"
CONFIGS = {
    seed: (
        PROJECT_ROOT
        / "configs"
        / "confirmatory-ablations"
        / f"v3_no_reliability_seed{seed}.json"
    )
    for seed in SEEDS
}
CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "confirmatory-ablations"
    / "v3_no_reliability"
)
PERFORMANCE_PROVENANCE = (
    PROJECT_ROOT
    / "configs"
    / "performance"
    / "full_cross_modal_quality_workers4.provenance.json"
)
DEGRADATION_SEEDS = [2026, 2027, 2028, 2029, 2030]
ROBUSTNESS_SCENARIOS = [
    "clean",
    "missing_100",
    "noise_015",
    "bias_008",
    "stuck",
    "mixed",
]
COMPARISON_BOOTSTRAP_SEEDS = {
    "42": 20260842,
    "43": 20260843,
    "44": 20260844,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the v3 reliability-subsystem ablation protocol"
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


def validate_ablation_config(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_name = f"v3_no_reliability_seed{seed}"
    expected_checkpoint = (
        Path("checkpoints")
        / "confirmatory-ablations"
        / "v3_no_reliability"
        / f"seed{seed}"
    ).as_posix()
    if normalized_reliability_ablation_config(base) != (
        normalized_reliability_ablation_config(candidate)
    ):
        errors.append("unauthorized scientific configuration differs")
    model = candidate.get("model", {})
    loss = candidate.get("loss", {})
    training = candidate.get("training", {})
    experiment = candidate.get("experiment", {})
    for field in MODEL_REMOVALS:
        if model.get(field) is not False:
            errors.append(f"model.{field} must be false")
    for field in LOSS_REMOVALS:
        if loss.get(field) != 0.0:
            errors.append(f"loss.{field} must be zero")
    if training.get("seed") != seed:
        errors.append("training seed mismatch")
    if training.get("num_workers") != 4:
        errors.append("num_workers must be the equivalent workers=4 setting")
    if Path(str(training.get("checkpoint_dir", ""))).as_posix() != expected_checkpoint:
        errors.append("checkpoint path mismatch")
    if (
        training.get("epochs") != 100
        or training.get("batch_size") != 16
        or experiment.get("name") != expected_name
        or experiment.get("ablation_of") != "full_cross_modal_quality"
        or experiment.get("ablation_scope")
        != "entire_sensor_reliability_subsystem"
        or experiment.get("matched_training_seed") != seed
    ):
        errors.append("experiment identity or fixed training settings mismatch")
    return {
        "valid": not errors,
        "seed": seed,
        "experiment": expected_name,
        "checkpoint_dir": expected_checkpoint,
        "errors": errors,
    }


def _full_reference_paths(seed: int) -> dict[str, Path]:
    if seed == 42:
        report = (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "full_cross_modal_quality"
        )
        return {
            "config": BASE_CONFIG,
            "checkpoint": (
                PROJECT_ROOT
                / "checkpoints"
                / "ablations"
                / "full_cross_modal_quality"
                / "best.pt"
            ),
            "audit": report / "CHECKPOINT_AUDIT.json",
            "val": report / "val" / "evaluation.json",
            "val_predictions": report / "val" / "predictions.jsonl",
            "test": report / "test" / "evaluation.json",
            "test_predictions": report / "test" / "predictions.jsonl",
        }
    report = (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "seed-replication"
        / f"seed{seed}"
    )
    name = f"full_cross_modal_quality_seed{seed}"
    return {
        "config": PROJECT_ROOT / "configs" / "replicates" / f"{name}.json",
        "checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "v3_seed_replicates"
            / name
            / "best.pt"
        ),
        "audit": report / "CHECKPOINT_AUDIT.json",
        "val": report / "val" / "evaluation.json",
        "val_predictions": report / "val" / "predictions.jsonl",
        "test": report / "test" / "evaluation.json",
        "test_predictions": report / "test" / "predictions.jsonl",
    }


def build_protocol(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite the ablation protocol")
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("ablation checkpoint root is not empty")
    if REPORT_ROOT.exists():
        allowed = {output_path, markdown_path}
        unexpected = [
            path for path in REPORT_ROOT.iterdir() if path.resolve() not in allowed
        ]
        if unexpected:
            raise ValueError("ablation report root is not empty")

    base = _read_json(BASE_CONFIG)
    if (
        base.get("experiment", {}).get("name") != "full_cross_modal_quality"
        or base.get("training", {}).get("seed") != 42
        or base.get("training", {}).get("epochs") != 100
        or base.get("training", {}).get("batch_size") != 16
    ):
        raise ValueError("v3 base configuration identity mismatch")
    performance = _read_json(PERFORMANCE_PROVENANCE)
    if (
        performance.get("valid") is not True
        or performance.get("state") != "complete"
        or performance.get("authorized_changes", {})
        .get("training.num_workers", {})
        .get("to")
        != 4
        or performance.get("equivalence", {}).get("sample_order_identical")
        is not True
        or performance.get("equivalence", {})
        .get("image_sensor_and_label_bytes_identical")
        is not True
    ):
        raise ValueError("workers=4 equivalence evidence is invalid")

    config_checks = {}
    for seed, path in CONFIGS.items():
        check = validate_ablation_config(base, _read_json(path), seed=seed)
        config_checks[str(seed)] = check
        if not check["valid"]:
            raise ValueError(
                f"seed {seed} ablation config invalid: "
                + ", ".join(check["errors"])
            )

    sources: dict[str, Path] = {
        "builder_source": Path(__file__),
        "audit_source": PROJECT_ROOT / "audit_v3_reliability_ablation.py",
        "queue_source": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "training"
            / "run_v3_reliability_ablation.ps1"
        ),
        "prepare_source": PROJECT_ROOT / "prepare_v3_reliability_ablation.py",
        "paper_addendum_source": (
            PROJECT_ROOT / "build_v3_reliability_ablation_addendum.py"
        ),
        "matched_summary_source": (
            PROJECT_ROOT / "summarize_matched_seed_comparisons.py"
        ),
        "robustness_comparison_source": (
            PROJECT_ROOT / "compare_robustness_summaries.py"
        ),
        "training_seed_summary_source": (
            PROJECT_ROOT / "summarize_seed_evaluations.py"
        ),
        "train_entrypoint": PROJECT_ROOT / "train.py",
        "evaluate_entrypoint": PROJECT_ROOT / "evaluate.py",
        "robustness_entrypoint": PROJECT_ROOT / "evaluate_robustness.py",
        "prediction_comparison_entrypoint": PROJECT_ROOT / "compare_predictions.py",
        "checkpoint_audit_entrypoint": PROJECT_ROOT / "audit_checkpoints.py",
        "model_source": PROJECT_ROOT / "mmfc" / "model.py",
        "loss_source": PROJECT_ROOT / "mmfc" / "losses.py",
        "training_source": PROJECT_ROOT / "mmfc" / "training.py",
        "data_source": PROJECT_ROOT / "mmfc" / "data.py",
        "base_config": BASE_CONFIG,
        "performance_provenance": PERFORMANCE_PROVENANCE,
        "input_pipeline_benchmark": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "training"
            / "INPUT_PIPELINE_BENCHMARK.json"
        ),
        "train_manifest": PROJECT_ROOT / "data" / "qwen-material" / "train.jsonl",
        "val_manifest": PROJECT_ROOT / "data" / "qwen-material" / "val.jsonl",
        "test_manifest": PROJECT_ROOT / "data" / "qwen-material" / "test.jsonl",
        "sensor_manifest": (
            PROJECT_ROOT / "data" / "qwen-material" / "sensors.jsonl"
        ),
        "yolo_initial_weights": PROJECT_ROOT.parent / "yolo11s-seg.pt",
        "full_seed42_robustness": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "full_cross_modal_quality"
            / "robustness"
            / "robustness_multiseed_summary.json"
        ),
    }
    for seed, path in CONFIGS.items():
        sources[f"ablation_seed{seed}_config"] = path
        for label, reference_path in _full_reference_paths(seed).items():
            sources[f"full_seed{seed}_{label}"] = reference_path
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("protocol sources missing: " + ", ".join(missing))

    for seed in SEEDS:
        reference = _full_reference_paths(seed)
        audit = _read_json(reference["audit"])
        validation = _read_json(reference["val"])
        test = _read_json(reference["test"])
        if (
            audit.get("valid") is not True
            or audit.get("last", {}).get("epoch") != 100
            or validation.get("metrics", {}).get("sample_count") != 826
            or test.get("metrics", {}).get("sample_count") != 826
        ):
            raise ValueError(f"full v3 seed {seed} reference evidence is incomplete")

    protocol = {
        "schema_version": 1,
        "state": "frozen_before_ablation_training",
        "valid": True,
        "scope": "matched_seed_v3_entire_reliability_subsystem_ablation",
        "scientific_question": (
            "Does the complete v3 reliability subsystem improve clean-test "
            "classification and software-degradation robustness over an "
            "otherwise matched three-modal model?"
        ),
        "training": {
            "matched_seeds": list(SEEDS),
            "epochs": 100,
            "batch_size": 16,
            "num_workers": 4,
            "resume": False,
            "best_checkpoint_selected_by_validation_accuracy": True,
            "full_model_checkpoint_replacement_forbidden": True,
            "test_used_for_description_not_model_selection": True,
        },
        "intervention": {
            "scope": "entire_sensor_reliability_subsystem",
            "model_fields_set_false": list(MODEL_REMOVALS),
            "loss_fields_set_zero": list(LOSS_REMOVALS),
            "software_sensor_degradation_training_retained": True,
            "three_modal_backbone_and_attention_fusion_retained": True,
            "interpretation_boundary": (
                "This is a subsystem ablation, not an isolated gate-only toggle."
            ),
            "config_checks": config_checks,
        },
        "performance_equivalence": {
            "num_workers": 4,
            "sample_order_identical": True,
            "image_sensor_and_label_bytes_identical": True,
            "measured_scope": "input pipeline only",
        },
        "clean_evaluation": {
            "splits": ["val", "test"],
            "samples_per_split": 826,
            "classes": 19,
            "bootstrap_repeats_per_evaluation": 2000,
            "bootstrap_seed": 20260727,
            "matched_prediction_comparison": {
                "candidate": "full_v3",
                "reference": "v3_no_reliability",
                "cluster_bootstrap_repeats": 5000,
                "cluster_count": 570,
                "bootstrap_seeds": COMPARISON_BOOTSTRAP_SEEDS,
            },
        },
        "robustness_evaluation": {
            "training_seed": 42,
            "split": "test",
            "scenarios": ROBUSTNESS_SCENARIOS,
            "degradation_seeds": DEGRADATION_SEEDS,
            "boundary": (
                "Degradation seeds are repeated perturbations, not independent "
                "model-training seeds."
            ),
        },
        "process_exit_policy": {
            "zero_exit_required_by_queue": True,
            "complete_bundle_requires": [
                "training_summary_completed_epochs_equals_100",
                "training_summary_checkpoints_saved_true",
                "history_object_contains_100_rows",
                "history_last_epoch_equals_100",
                "best_and_last_checkpoints_exist",
                "independent_checkpoint_audit_valid",
            ],
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
        markdown_path,
        "\n".join(
            [
                "# v3 reliability-subsystem confirmatory ablation protocol",
                "",
                "- Frozen before seeds 42, 43, and 44 begin ablation training.",
                "- The intervention removes the entire learned sensor-reliability "
                "subsystem and all associated auxiliary losses.",
                "- Model, data, optimizer, software-degradation training, batch "
                "size, and 100-epoch schedule otherwise match v3.",
                "- Clean validation/test comparisons use matched training seeds; "
                "full v3 is the candidate and no-reliability v3 is the reference.",
                "- Seed 42 alone receives the six-scenario, five-degradation-seed "
                "robustness comparison.",
                "- The internal test is already inspected and remains descriptive.",
                "",
            ]
        ),
    )
    return {
        "state": "complete",
        "valid": True,
        "output": output_path.as_posix(),
        "source_binding_count": len(sources),
        "training_seeds": list(SEEDS),
    }


def main() -> None:
    result = build_protocol(parse_args().output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
