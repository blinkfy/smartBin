"""Prepare matched-seed v3 auxiliary-loss component ablations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "configs" / "ablations" / "full_cross_modal_quality.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "configs"
    / "confirmatory-ablations"
    / "v3-loss-components"
)
SEEDS = (42, 43, 44)
VARIANTS = {
    "v3_no_grid_reconstruction": {
        "loss_field": "sensor_grid_reconstruction_weight",
        "ablation_scope": "sensor_grid_reconstruction_supervision",
        "description": (
            "Matched-seed v3 without the clean-grid Smooth L1 auxiliary "
            "supervision; all model paths and other losses remain enabled"
        ),
    },
    "v3_no_ranking": {
        "loss_field": "calibration_ranking_weight",
        "ablation_scope": "pairwise_reliability_ranking_supervision",
        "description": (
            "Matched-seed v3 without pairwise reliability-ranking "
            "supervision; all model paths and other losses remain enabled"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare v3 auxiliary-loss component ablation configs"
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
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


def normalized_component_config(
    config: Mapping[str, Any],
    *,
    loss_field: str,
) -> dict[str, Any]:
    """Remove only the declared intervention and run-identity fields."""

    value = copy.deepcopy(dict(config))
    value.setdefault("loss", {}).pop(loss_field, None)
    training = value.setdefault("training", {})
    for name in ("seed", "num_workers", "checkpoint_dir"):
        training.pop(name, None)
    experiment = value.setdefault("experiment", {})
    for name in (
        "name",
        "group",
        "description",
        "ablation_of",
        "ablation_scope",
        "intervention_field",
        "matched_training_seed",
        "long_horizon_worker_policy",
        "reference_checkpoint",
    ):
        experiment.pop(name, None)
    return value


def build_component_config(
    base: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported component ablation: {variant}")
    if seed not in SEEDS:
        raise ValueError(f"unsupported matched seed: {seed}")
    specification = VARIANTS[variant]
    loss_field = str(specification["loss_field"])
    base_weight = float(base.get("loss", {}).get(loss_field, 0.0))
    if base_weight <= 0.0:
        raise ValueError(f"base loss.{loss_field} must be positive")

    value = copy.deepcopy(dict(base))
    value["loss"][loss_field] = 0.0
    value["training"]["seed"] = seed
    value["training"]["num_workers"] = 0
    value["training"]["checkpoint_dir"] = (
        "checkpoints/confirmatory-ablations/v3_loss_components/"
        f"{variant}/seed{seed}"
    )
    experiment_name = f"{variant}_seed{seed}"
    value["experiment"] = {
        **value.get("experiment", {}),
        "name": experiment_name,
        "group": "confirmatory_v3_loss_component_ablation",
        "description": str(specification["description"]),
        "ablation_of": "full_cross_modal_quality",
        "ablation_scope": str(specification["ablation_scope"]),
        "intervention_field": f"loss.{loss_field}",
        "matched_training_seed": seed,
        "long_horizon_worker_policy": (
            "workers=0 after the workers=4 confirmatory run stalled at epoch 69"
        ),
        "reference_checkpoint": None,
    }
    if normalized_component_config(
        value,
        loss_field=loss_field,
    ) != normalized_component_config(base, loss_field=loss_field):
        raise ValueError("component config changes unauthorized scientific fields")
    return value


def prepare_configs(
    *,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    base_config_path = _resolve(base_config_path)
    output_dir = _resolve(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("v3 loss-component config directory is not empty")
    base = _read_json(base_config_path)
    if (
        base.get("experiment", {}).get("name") != "full_cross_modal_quality"
        or base.get("training", {}).get("epochs") != 100
        or base.get("training", {}).get("batch_size") != 16
        or base.get("training", {}).get("num_workers") != 0
    ):
        raise ValueError("v3 base configuration identity mismatch")

    outputs: dict[str, Any] = {}
    for variant, specification in VARIANTS.items():
        outputs[variant] = {}
        for seed in SEEDS:
            path = output_dir / f"{variant}_seed{seed}.json"
            config = build_component_config(base, variant=variant, seed=seed)
            _atomic_write(
                path,
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            )
            outputs[variant][str(seed)] = {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(path),
                "checkpoint_dir": config["training"]["checkpoint_dir"],
                "intervention": {
                    f"loss.{specification['loss_field']}": {
                        "from": base["loss"][specification["loss_field"]],
                        "to": 0.0,
                    }
                },
            }
    return {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "matched_seed_v3_auxiliary_loss_component_configs",
        "base_config": base_config_path.relative_to(PROJECT_ROOT).as_posix(),
        "base_config_sha256": _sha256(base_config_path),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "num_workers": 0,
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    report = prepare_configs(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
