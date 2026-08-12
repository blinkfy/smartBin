from __future__ import annotations

import copy
import json

from mmfc.training import PROJECT_ROOT


def _load(name: str) -> dict:
    path = PROJECT_ROOT / "configs" / "ablations" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_balanced_candidate_only_changes_pre_registered_behavior():
    reference = _load("full_cross_modal_quality.json")
    candidate = _load("full_cross_modal_quality_balanced.json")

    for section in ("classes", "model", "robustness", "data", "preprocess"):
        assert candidate[section] == reference[section]

    candidate_loss = copy.deepcopy(candidate["loss"])
    assert candidate_loss.pop("class_weighting") == {
        "method": "sqrt_inverse",
        "apply_to_ce": True,
        "apply_to_focal": False,
    }
    assert candidate_loss == reference["loss"]

    candidate_training = copy.deepcopy(candidate["training"])
    assert candidate_training.pop("selection_metric") == "macro_f1"
    candidate_training["checkpoint_dir"] = reference["training"]["checkpoint_dir"]
    assert candidate_training == reference["training"]

    assert candidate["experiment"]["name"] == (
        "full_cross_modal_quality_balanced"
    )
    assert candidate["experiment"]["group"] == "class_balance"
    assert candidate["experiment"]["reference_checkpoint"] == (
        "checkpoints/ablations/full_cross_modal_quality/best.pt"
    )
