"""Freeze a workers=0 recovery after the workers=4 seed42 stall."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_v3_reliability_ablation import validate_protocol
from prepare_v3_reliability_ablation import SEEDS
from prepare_v3_reliability_ablation_recovery import (
    BASE_CONFIG,
    DEFAULT_OUTPUT_DIR as CONFIG_ROOT,
    validate_recovery_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability-workers0"
)
DEFAULT_OUTPUT = REPORT_ROOT / "RECOVERY_PROTOCOL_ADDENDUM.json"
ORIGINAL_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability"
)
ORIGINAL_PROTOCOL = ORIGINAL_ROOT / "PROTOCOL.json"
INCIDENT = ORIGINAL_ROOT / "WORKERS4_STALL_INCIDENT.json"
CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "confirmatory-ablations"
    / "v3_no_reliability_workers0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the workers=0 reliability-ablation recovery"
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
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_addendum(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite recovery protocol addendum")
    if CHECKPOINT_ROOT.exists() and any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("workers=0 recovery checkpoint root is not empty")
    if REPORT_ROOT.exists():
        allowed = {output_path, markdown_path}
        unexpected = [
            path for path in REPORT_ROOT.iterdir() if path.resolve() not in allowed
        ]
        if unexpected:
            raise ValueError("workers=0 recovery report root is not empty")
    original = validate_protocol(ORIGINAL_PROTOCOL)
    if original["valid"] is not True:
        raise ValueError("original frozen protocol no longer validates")
    incident = _read_json(INCIDENT)
    if (
        incident.get("state") != "closed_incomplete_and_archived"
        or incident.get("valid") is not True
        or incident.get("facts", {}).get("last_completed_epoch") != 69
        or incident.get("recovery_decision", {}).get(
            "fresh_training_from_epoch_1"
        )
        is not True
        or incident.get("recovery_decision", {}).get(
            "replacement_num_workers"
        )
        != 0
    ):
        raise ValueError("stall incident does not authorize workers=0 recovery")
    base = _read_json(BASE_CONFIG)
    configs = {
        seed: CONFIG_ROOT / f"v3_no_reliability_workers0_seed{seed}.json"
        for seed in SEEDS
    }
    config_checks = {}
    for seed, path in configs.items():
        check = validate_recovery_config(base, _read_json(path), seed=seed)
        config_checks[str(seed)] = check
        if not check["valid"]:
            raise ValueError(
                f"seed {seed} recovery config invalid: "
                + ", ".join(check["errors"])
            )

    sources = {
        "addendum_builder_source": Path(__file__),
        "recovery_audit_source": (
            PROJECT_ROOT / "audit_v3_reliability_ablation_recovery.py"
        ),
        "recovery_queue_source": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "training"
            / "run_v3_reliability_ablation_workers0_recovery.ps1"
        ),
        "recovery_prepare_source": (
            PROJECT_ROOT / "prepare_v3_reliability_ablation_recovery.py"
        ),
        "original_protocol": ORIGINAL_PROTOCOL,
        "stall_incident": INCIDENT,
        "stall_liveness": ORIGINAL_ROOT / "WORKERS4_STALL_LIVENESS.json",
        "partial_checkpoint_audit": (
            ORIGINAL_ROOT / "PARTIAL_CHECKPOINT_AUDIT.json"
        ),
        "stall_capture_source": (
            PROJECT_ROOT / "capture_v3_reliability_ablation_stall.py"
        ),
        "stall_incident_builder_source": (
            PROJECT_ROOT / "build_v3_reliability_ablation_stall_incident.py"
        ),
        "base_config": BASE_CONFIG,
        "input_pipeline_benchmark": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "training"
            / "INPUT_PIPELINE_BENCHMARK.json"
        ),
        "performance_provenance": (
            PROJECT_ROOT
            / "configs"
            / "performance"
            / "full_cross_modal_quality_workers4.provenance.json"
        ),
        "full_v3_seed_replication_outcome": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "full_cross_modal_quality"
            / "seed-replication"
            / "OUTCOME.json"
        ),
    }
    for seed, path in configs.items():
        sources[f"recovery_seed{seed}_config"] = path
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("recovery sources missing: " + ", ".join(missing))
    replication = _read_json(sources["full_v3_seed_replication_outcome"])
    if (
        replication.get("valid") is not True
        or replication.get("training", {}).get("seeds") != [42, 43, 44]
        or replication.get("training", {}).get("epochs_per_run") != 100
    ):
        raise ValueError("stable workers=0 v3 training evidence is incomplete")

    addendum = {
        "schema_version": 1,
        "state": "frozen_before_workers0_recovery",
        "valid": True,
        "scope": "operational_recovery_only",
        "original_protocol": {
            "path": ORIGINAL_PROTOCOL.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(ORIGINAL_PROTOCOL),
            "scientific_question_unchanged": True,
            "training_seeds_unchanged": True,
            "evaluation_and_comparison_settings_unchanged": True,
        },
        "incident": {
            "path": INCIDENT.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(INCIDENT),
            "stalled_seed": 42,
            "last_completed_epoch": 69,
            "partial_run_used_for_results": False,
            "partial_run_will_be_resumed": False,
        },
        "recovery": {
            "training_seeds": list(SEEDS),
            "epochs": 100,
            "fresh_epoch1_runs": True,
            "resume": False,
            "num_workers": 0,
            "persistent_workers": False,
            "new_checkpoint_root": (
                CHECKPOINT_ROOT.relative_to(PROJECT_ROOT).as_posix()
            ),
            "new_report_root": REPORT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
            "config_checks": config_checks,
        },
        "equivalence_interpretation": {
            "scientific_configuration_unchanged": True,
            "sample_order_and_tensor_bytes_workers0_vs4_identical": True,
            "workers4_short_benchmark_still_valid": True,
            "workers4_100_epoch_liveness_invalid_on_this_windows_run": True,
            "workers0_has_prior_three_seed_100_epoch_completion_evidence": True,
            "speed_claim_withdrawn_for_long_horizon_training": True,
        },
        "process_policy": {
            "zero_exit_required": True,
            "best_checkpoint_selected_by_validation_accuracy": True,
            "test_used_for_description_not_model_selection": True,
            "workers4_partial_checkpoint_reuse_forbidden": True,
        },
        "source_bindings": {
            name: _binding(path) for name, path in sources.items()
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        markdown_path,
        "\n".join(
            [
                "# workers=0 recovery protocol addendum",
                "",
                "- Frozen after the workers=4 seed42 run stalled at epoch 69 and "
                "before any recovery training begins.",
                "- The incomplete run is archived and cannot be resumed or used.",
                "- Seeds 42, 43, and 44 restart from epoch 1 in new output roots.",
                "- Only DataLoader workers change from 4 to the original stable "
                "setting 0; model, losses, data, optimizer, schedule, evaluation, "
                "and matched comparisons remain frozen.",
                "- The short workers=4 input benchmark remains an input-equivalence "
                "result, but its long-horizon speed recommendation is withdrawn.",
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
    result = build_addendum(parse_args().output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
