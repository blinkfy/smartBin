"""Create matched two-stage configs for the non-semantic class-ID control."""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    output = ROOT / "configs" / "class-id-key-control"
    for seed in (42, 43, 44):
        stage1_source = (
            ROOT
            / "configs"
            / "semantic-guidance-v3"
            / f"semantic_fallback_seed{seed}.json"
        )
        stage2_source = (
            ROOT
            / "configs"
            / "semantic-guidance-v5"
            / f"semantic_fallback_seed{seed}.json"
        )
        stage1 = copy.deepcopy(read(stage1_source))
        stage1["model"]["fallback_semantic_mode"] = "class_id_keys"
        stage1["training"]["checkpoint_dir"] = (
            f"checkpoints/class-id-key-control/stage1/seed{seed}"
        )
        stage1["experiment"].update(
            {
                "name": f"class_id_key_stage1_seed{seed}",
                "group": "non_semantic_class_key_control",
                "description": (
                    "Fixed orthogonal class-ID keys replace CLIP keys while "
                    "retaining the matched class-memory retrieval protocol"
                ),
            }
        )
        stage1_path = output / f"stage1_seed{seed}.json"
        write(stage1_path, stage1)

        stage2 = copy.deepcopy(read(stage2_source))
        stage2["model"]["fallback_semantic_mode"] = "class_id_keys"
        stage2["training"]["initialization_checkpoint"] = (
            f"checkpoints/class-id-key-control/stage1/seed{seed}/best.pt"
        )
        stage2["training"]["checkpoint_dir"] = (
            f"checkpoints/class-id-key-control/final/seed{seed}"
        )
        stage2["experiment"].update(
            {
                "name": f"class_id_key_final_seed{seed}",
                "group": "non_semantic_class_key_control",
                "description": (
                    "Direct latent fallback with fixed orthogonal class-ID "
                    "keys and training-split sensor centers"
                ),
                "reference_checkpoint": (
                    f"checkpoints/class-id-key-control/stage1/seed{seed}/best.pt"
                ),
            }
        )
        write(output / f"final_seed{seed}.json", stage2)

    print(output.resolve())


if __name__ == "__main__":
    main()
