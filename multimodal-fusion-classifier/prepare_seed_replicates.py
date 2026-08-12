"""Derive isolated confirmatory training-seed configs from a locked config."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from mmfc.data import write_json
from mmfc.training import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate isolated confirmatory seed configs"
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    return parser.parse_args()


def build_seed_replicate(
    base_config: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_root: str | Path,
) -> dict[str, Any]:
    """Change only seed, isolated artifact paths, and experiment metadata."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    config = copy.deepcopy(dict(base_config))
    config.pop("_config_path", None)
    training = config.setdefault("training", {})
    original_seed = int(training.get("seed", 42))
    if seed == original_seed:
        raise ValueError("replicate seed must differ from the base seed")

    experiment = config.setdefault("experiment", {})
    base_name = str(experiment.get("name", "experiment"))
    replicate_name = f"{base_name}_seed{seed}"
    training["seed"] = seed
    training["checkpoint_dir"] = (
        Path(checkpoint_root) / replicate_name
    ).as_posix()
    base_description = str(experiment.get("description", "")).rstrip()
    experiment["name"] = replicate_name
    experiment["description"] = (
        f"{base_description}; confirmatory training seed {seed}"
        if base_description
        else f"Confirmatory training seed {seed}"
    )
    experiment["replicate_of"] = base_name
    return config


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")

    base_path = resolve_project_path(args.base_config)
    with base_path.open("r", encoding="utf-8") as handle:
        base_config = json.load(handle)
    output_dir = resolve_project_path(args.output)
    checkpoint_root = args.checkpoint_root
    artifacts: list[dict[str, Any]] = []
    for seed in args.seeds:
        config = build_seed_replicate(
            base_config,
            seed=seed,
            checkpoint_root=checkpoint_root,
        )
        name = config["experiment"]["name"]
        output_path = output_dir / f"{name}.json"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite config: {output_path}")
        checkpoint_dir = resolve_project_path(
            config["training"]["checkpoint_dir"]
        )
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError(
                f"checkpoint directory is not empty: {checkpoint_dir}"
            )
        write_json(output_path, config)
        artifacts.append(
            {
                "seed": seed,
                "experiment": name,
                "config": str(output_path),
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
    print(json.dumps({"replicates": artifacts}, indent=2))


if __name__ == "__main__":
    main()
