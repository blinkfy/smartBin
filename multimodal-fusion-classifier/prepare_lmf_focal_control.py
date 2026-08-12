"""Create three matched LMF configs with CE plus focal classification loss."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    output = ROOT / "configs" / "standard-fusion-baselines-focal"
    output.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44):
        source = (
            ROOT
            / "configs"
            / "standard-fusion-baselines"
            / f"lmf_seed{seed}.json"
        )
        config = json.loads(source.read_text(encoding="utf-8"))
        config["recent_baseline"]["classification_focal_weight"] = 1.0
        config["training"]["checkpoint_dir"] = (
            f"checkpoints/standard-fusion-baselines-focal/lmf/seed{seed}"
        )
        config["experiment"].update(
            {
                "name": f"lmf_ce_focal_seed{seed}",
                "group": "standard_fusion_objective_control",
                "description": (
                    "Rank-4 LMF retrained with the proposed classifier's "
                    "unit-weight cross-entropy plus focal objective"
                ),
                "reference_checkpoint": None,
            }
        )
        destination = output / f"lmf_focal_seed{seed}.json"
        destination.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(output.resolve())


if __name__ == "__main__":
    main()
