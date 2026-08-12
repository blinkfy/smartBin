"""Generate the uniform three-seed independently trained modality baselines."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from mmfc.training import PROJECT_ROOT, load_config


SEEDS = (42, 43, 44)
VARIANTS: dict[str, dict[str, Any]] = {
    "visual_only": {
        "label": "Visual only",
        "modalities": ["visual"],
        "source": PROJECT_ROOT / "configs" / "ablations" / "vision_only.json",
        "legacy_seed42_checkpoint": (
            PROJECT_ROOT / "checkpoints" / "ablations" / "vision_only"
        ),
    },
    "sensor_only": {
        "label": "Sensor only",
        "modalities": ["sensor"],
        "source": PROJECT_ROOT / "configs" / "ablations" / "vision_sensor.json",
    },
    "visual_text": {
        "label": "Visual + text",
        "modalities": ["visual", "text"],
        "source": PROJECT_ROOT / "configs" / "ablations" / "vision_text.json",
        "legacy_seed42_checkpoint": (
            PROJECT_ROOT / "checkpoints" / "ablations" / "vision_text"
        ),
    },
    "visual_sensor": {
        "label": "Visual + sensor",
        "modalities": ["visual", "sensor"],
        "source": PROJECT_ROOT / "configs" / "ablations" / "vision_sensor.json",
        "legacy_seed42_checkpoint": (
            PROJECT_ROOT / "checkpoints" / "ablations" / "vision_sensor"
        ),
    },
    "text_sensor": {
        "label": "Text + sensor",
        "modalities": ["text", "sensor"],
        "source": PROJECT_ROOT / "configs" / "ablations" / "vision_sensor.json",
    },
}
CONFIG_ROOT = PROJECT_ROOT / "configs" / "modality-baselines"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "modality-baselines"


def build_config(variant: str, seed: int) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown modality baseline: {variant}")
    if seed not in SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    specification = VARIANTS[variant]
    config = copy.deepcopy(load_config(specification["source"]))
    config.pop("_config_path", None)
    model = config.setdefault("model", {})
    model.update(
        {
            "enabled_modalities": list(specification["modalities"]),
            "fusion_mode": "mean",
            "film_enabled": False,
            "sensor_reliability_enabled": False,
            "sensor_grid_quality_enabled": False,
            "sensor_cross_modal_quality_enabled": False,
        }
    )
    loss = config.setdefault("loss", {})
    for name in (
        "reconstruction_weight",
        "calibration_weight",
        "variance_prior_weight",
        "sensor_grid_reconstruction_weight",
        "calibration_ranking_weight",
    ):
        loss[name] = 0.0
    config.setdefault("robustness", {}).setdefault(
        "sensor_degradation", {}
    )["enabled"] = False
    training = config.setdefault("training", {})
    training["seed"] = seed
    training["epochs"] = 100
    training["num_workers"] = 0
    training["checkpoint_every_epochs"] = 5
    if seed == 42 and "legacy_seed42_checkpoint" in specification:
        checkpoint_dir = specification["legacy_seed42_checkpoint"]
    else:
        checkpoint_dir = CHECKPOINT_ROOT / variant / f"seed{seed}"
    training["checkpoint_dir"] = checkpoint_dir.relative_to(
        PROJECT_ROOT
    ).as_posix()
    config["experiment"] = {
        "name": variant,
        "label": specification["label"],
        "group": "independently_retrained_modality_baseline",
        "protocol": "uniform_three_seed_modality_matrix_v1",
        "seed": seed,
        "enabled_modalities": list(specification["modalities"]),
        "text_only_excluded": (
            "class descriptions contain no sample-specific observation"
        ),
    }
    return config


def config_path(variant: str, seed: int) -> Path:
    return CONFIG_ROOT / f"{variant}_seed{seed}.json"


def write_configs(
    variants: tuple[str, ...] = tuple(VARIANTS),
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, str]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for variant in variants:
        for seed in seeds:
            path = config_path(variant, seed)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(
                    build_config(variant, seed),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
            artifacts[f"{variant}_seed{seed}"] = str(path)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare uniform independently trained modality baselines"
    )
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = write_configs(tuple(args.variants), tuple(args.seeds))
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
