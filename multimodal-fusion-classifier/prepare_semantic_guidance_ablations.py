"""Prepare matched semantic-guidance fallback experiments."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "configs" / "semantic-guidance-v5"

SEMANTIC_TEXT = {
    "potato": (
        "whole potato",
        "A dense moist organic tuber with high water content, a cool thermal "
        "response, low electrical conductivity, and a mild earthy odor.",
    ),
    "rabbitcut": (
        "cut rabbit meat",
        "A moist animal-tissue piece with high water content, soft texture, "
        "moderate conductivity, and a noticeable organic odor response.",
    ),
    "battery1": (
        "large dry-cell battery",
        "A heavy rigid electrochemical cell with a metal casing, high thermal "
        "mass, conductive terminals, no moisture, and little odor.",
    ),
    "battery5": (
        "AA dry-cell battery",
        "A compact dense electrochemical cell with a cylindrical metal shell, "
        "conductive terminals, no moisture, and little odor.",
    ),
    "bottle": (
        "plastic beverage bottle",
        "A light hollow polymer container with low thermal mass, low "
        "conductivity, little retained moisture, and weak odor.",
    ),
    "brick": (
        "brick fragment",
        "A heavy porous mineral block with high thermal mass, low electrical "
        "conductivity, low moisture, and nearly no odor.",
    ),
    "can": (
        "aluminum beverage can",
        "A light hollow metal container with a conductive wall, rapid thermal "
        "response, little moisture, and weak residual odor.",
    ),
    "china": (
        "broken ceramic shard",
        "A rigid dense ceramic fragment with moderate thermal mass, very low "
        "conductivity, no moisture, and no odor.",
    ),
    "stone": (
        "natural stone",
        "A dense irregular mineral object with high thermal mass, low "
        "conductivity, low moisture, and no odor.",
    ),
    "drug": (
        "loose medicine or blister pack",
        "A small medicinal item or blister unit with low mass, limited "
        "moisture, mixed polymer and foil conductivity, and a weak chemical odor.",
    ),
    "drugbag": (
        "flexible medicine sachet",
        "A light thin pharmaceutical pouch with low thermal mass, low moisture, "
        "mixed film conductivity, and a weak chemical odor.",
    ),
    "mooli": (
        "whole white radish",
        "A heavy water-rich root vegetable with high moisture, a cool thermal "
        "response, moderate conductivity, and a fresh organic odor.",
    ),
    "battery": (
        "discarded household battery",
        "A dense rigid battery with metal electrodes, conductive surfaces, no "
        "moisture, high thermal mass, and little odor.",
    ),
    "bottle2": (
        "rigid bottle container",
        "A hollow bottle made of glass or rigid polymer with low moisture, low "
        "odor, and material-dependent thermal mass and conductivity.",
    ),
    "drugbox": (
        "medicine carton",
        "A light dry paperboard package with low thermal mass, very low "
        "conductivity, little moisture, and a weak paper or medicine odor.",
    ),
    "papercup": (
        "disposable paper cup",
        "A light paper-based cup with low conductivity and thermal mass, "
        "possibly slight retained moisture, and a weak beverage odor.",
    ),
    "capsule": (
        "medicine capsule",
        "A very light dry pharmaceutical capsule with low thermal mass, low "
        "conductivity, little moisture, and a weak medicinal odor.",
    ),
    "potatocut": (
        "cut potato piece",
        "A moist cut tuber with exposed water-rich tissue, moderate "
        "conductivity, a cool thermal response, and a mild earthy odor.",
    ),
    "paper": (
        "waste paper sheet",
        "A light dry cellulose sheet with very low conductivity and thermal "
        "mass, little moisture, and a weak paper odor.",
    ),
}

BASE_CONFIGS = {
    42: PROJECT_ROOT
    / "configs"
    / "semantic-guidance-v3"
    / "semantic_fallback_seed42.json",
    43: PROJECT_ROOT
    / "configs"
    / "semantic-guidance-v3"
    / "semantic_fallback_seed43.json",
    44: PROJECT_ROOT
    / "configs"
    / "semantic-guidance-v3"
    / "semantic_fallback_seed44.json",
}


def prepare_config(seed: int, base_path: Path) -> Path:
    config = deepcopy(json.loads(base_path.read_text(encoding="utf-8")))
    for item in config["classes"]:
        semantic_name, semantic_description = SEMANTIC_TEXT[item["name"]]
        item["semantic_name"] = semantic_name
        item["semantic_description"] = semantic_description

    model = config["model"]
    model["fallback_semantic_mode"] = "descriptions"
    model["fallback_semantic_shuffle_seed"] = 20260810
    model["fallback_text_encoder_independent"] = True
    model["fallback_semantic_memory_enabled"] = True
    model["fallback_semantic_direct_latent"] = True

    loss = config["loss"]
    loss["semantic_alignment_weight"] = 0.2
    loss["reconstruction_weight"] = 0.2
    loss["sensor_grid_reconstruction_weight"] = 1.0

    config["robustness"]["sensor_degradation"][
        "missing_probability"
    ] = 0.3

    training = config["training"]
    source_checkpoint = Path(training["checkpoint_dir"]) / "best.pt"
    training.update(
        {
            "seed": seed,
            "learning_rate": 2e-5,
            "epochs": 3,
            "warmup_epochs": 1,
            "selection_metric": "sensor_grid_reconstruction",
            "selection_mode": "min",
            "initialization_checkpoint": str(source_checkpoint),
            "freeze_visual_anchor": True,
            "freeze_modules": [
                "text_encoder",
                "fallback_text_encoder",
                "sensor_encoder",
            ],
            "initialize_fallback_sensor_value_bank": True,
            "semantic_memory_initialization_batch_size": 512,
            "semantic_memory_trainable": False,
            "checkpoint_dir": f"checkpoints/semantic-guidance-v5/seed{seed}",
        }
    )
    config["experiment"] = {
        "name": f"semantic_guidance_fallback_seed{seed}",
        "group": "semantic_guidance_fallback_v5",
        "description": (
            "Decoupled semantic memories use learned values for grid fallback "
            "and fixed train-split physical centers for latent fallback"
        ),
        "formal_retraining_required": True,
        "reference_checkpoint": str(source_checkpoint),
    }
    output = OUTPUT_DIR / f"semantic_fallback_seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    outputs = [prepare_config(seed, path) for seed, path in BASE_CONFIGS.items()]
    print(json.dumps([str(path) for path in outputs], indent=2))


if __name__ == "__main__":
    main()
