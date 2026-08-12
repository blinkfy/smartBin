"""Prepare fresh workers=0 recovery configs after the workers=4 stall."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from prepare_v3_reliability_ablation import (
    LOSS_REMOVALS,
    MODEL_REMOVALS,
    SEEDS,
    normalized_reliability_ablation_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_ROOT / "configs" / "ablations" / "full_cross_modal_quality.json"
INCIDENT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability"
    / "WORKERS4_STALL_INCIDENT.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "configs"
    / "confirmatory-ablations"
    / "recovery-workers0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare workers=0 reliability-ablation recovery configs"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_recovery_config(
    base: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unsupported matched seed: {seed}")
    value = copy.deepcopy(dict(base))
    for name in MODEL_REMOVALS:
        value["model"][name] = False
    for name in LOSS_REMOVALS:
        value["loss"][name] = 0.0
    value["training"]["seed"] = seed
    value["training"]["num_workers"] = 0
    value["training"]["checkpoint_dir"] = (
        "checkpoints/confirmatory-ablations/"
        f"v3_no_reliability_workers0/seed{seed}"
    )
    value["experiment"] = {
        **value.get("experiment", {}),
        "name": f"v3_no_reliability_workers0_seed{seed}",
        "group": "confirmatory_component_ablation",
        "description": (
            "Fresh workers=0 recovery of the matched-seed v3 ablation without "
            "the entire reliability subsystem"
        ),
        "ablation_of": "full_cross_modal_quality",
        "ablation_scope": "entire_sensor_reliability_subsystem",
        "matched_training_seed": seed,
        "performance_evidence": (
            "reports/qwen-material/confirmatory-ablations/"
            "v3-no-reliability/WORKERS4_STALL_INCIDENT.json"
        ),
        "reference_checkpoint": None,
    }
    if normalized_reliability_ablation_config(value) != (
        normalized_reliability_ablation_config(base)
    ):
        raise ValueError("recovery config changes unauthorized scientific fields")
    return value


def validate_recovery_config(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    errors = []
    expected = build_recovery_config(base, seed=seed)
    if candidate != expected:
        errors.append("recovery config does not equal the generated specification")
    if candidate.get("training", {}).get("num_workers") != 0:
        errors.append("recovery num_workers must equal zero")
    return {"valid": not errors, "seed": seed, "errors": errors}


def prepare_configs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("workers=0 recovery config directory is not empty")
    base = _read_json(BASE_CONFIG)
    incident = _read_json(INCIDENT)
    if (
        incident.get("state") != "closed_incomplete_and_archived"
        or incident.get("valid") is not True
        or incident.get("recovery_decision", {}).get(
            "fresh_training_from_epoch_1"
        )
        is not True
        or incident.get("recovery_decision", {}).get(
            "replacement_num_workers"
        )
        != 0
    ):
        raise ValueError("workers=4 stall incident does not authorize recovery")
    outputs = {}
    for seed in SEEDS:
        path = output_dir / f"v3_no_reliability_workers0_seed{seed}.json"
        config = build_recovery_config(base, seed=seed)
        _atomic_write(
            path,
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        )
        outputs[str(seed)] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(path),
            "checkpoint_dir": config["training"]["checkpoint_dir"],
        }
    return {
        "state": "complete",
        "valid": True,
        "base_config": BASE_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
        "base_config_sha256": _sha256(BASE_CONFIG),
        "incident": INCIDENT.relative_to(PROJECT_ROOT).as_posix(),
        "incident_sha256": _sha256(INCIDENT),
        "seeds": list(SEEDS),
        "num_workers": 0,
        "outputs": outputs,
    }


def main() -> None:
    result = prepare_configs(parse_args().output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
