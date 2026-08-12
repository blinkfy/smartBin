from __future__ import annotations

import copy

import pytest

from prepare_seed_replicates import build_seed_replicate


def _base_config() -> dict:
    return {
        "model": {"feature_dim": 512},
        "loss": {"class_weighting": {"method": "sqrt_inverse"}},
        "training": {
            "seed": 42,
            "epochs": 100,
            "checkpoint_dir": "checkpoints/base",
        },
        "experiment": {
            "name": "balanced",
            "description": "Locked candidate",
        },
    }


def test_seed_replicate_only_changes_seed_paths_and_metadata():
    base = _base_config()
    unchanged = copy.deepcopy(base)

    replicate = build_seed_replicate(
        base,
        seed=43,
        checkpoint_root="checkpoints/replicates",
    )

    assert base == unchanged
    assert replicate["model"] == base["model"]
    assert replicate["loss"] == base["loss"]
    assert replicate["training"] == {
        "seed": 43,
        "epochs": 100,
        "checkpoint_dir": "checkpoints/replicates/balanced_seed43",
    }
    assert replicate["experiment"]["name"] == "balanced_seed43"
    assert replicate["experiment"]["replicate_of"] == "balanced"
    assert "seed 43" in replicate["experiment"]["description"]


@pytest.mark.parametrize("seed", [-1, 42])
def test_seed_replicate_rejects_invalid_or_base_seed(seed: int):
    with pytest.raises(ValueError):
        build_seed_replicate(
            _base_config(),
            seed=seed,
            checkpoint_root="checkpoints/replicates",
        )
