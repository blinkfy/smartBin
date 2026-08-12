"""Create three configs for matched dynamic-versus-constant gate retraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BASE_CONFIGS = {
    42: ROOT / "configs" / "ablations" / "full_cross_modal_quality.json",
    43: ROOT / "configs" / "replicates" / "full_cross_modal_quality_seed43.json",
    44: ROOT / "configs" / "replicates" / "full_cross_modal_quality_seed44.json",
}
OUTPUT = ROOT / "configs" / "submission-revision" / "constant-gate"
CHECKPOINT_ROOT = "C:/mmfc-checkpoints/submission-revision/constant-gate-workers4"
FINAL_CHECKPOINT_ROOT = "checkpoints/submission-revision/constant-gate-matched"
NUM_WORKERS = 4
ALLOWED_CONFIG_DIFFERENCES = {
    "model.constant_sensor_reliability_gate",
    "training.num_workers",
    "training.preserve_single_worker_order",
    "training.checkpoint_dir",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def leaf_differences(
    left: Any,
    right: Any,
    prefix: str = "",
) -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(
                leaf_differences(left.get(key), right.get(key), path)
            )
        return rows
    if left == right:
        return []
    return [{"path": prefix, "base": left, "constant": right}]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = {
        "experiment": "dynamic_gate_vs_constant_learned_gate",
        "matched_difference": (
            "The only model or optimization change is "
            "model.constant_sensor_reliability_gate=false to true. The constant gate "
            "reuses the dynamic head output bias, retains the same parameter inventory, "
            "and is multiplied by archived-record availability. Constant-gate runs use "
            "four persistent data-loader workers instead of zero solely for deterministic "
            "input prefetch. A separate worker RNG and legacy-order sampler preserve the "
            "single-worker shuffle sequence; audit_dataloader_worker_equivalence.py "
            "verifies identical sample order for each seed."
        ),
        "runs": [],
    }
    for seed, source in BASE_CONFIGS.items():
        base_config = load_json(source)
        config = json.loads(json.dumps(base_config))
        if int(config["training"]["seed"]) != seed:
            raise ValueError(f"training seed mismatch in {source}")
        config["model"]["constant_sensor_reliability_gate"] = True
        config["training"]["num_workers"] = NUM_WORKERS
        config["training"]["preserve_single_worker_order"] = True
        config["training"]["checkpoint_dir"] = (
            f"{CHECKPOINT_ROOT}/seed{seed}"
        )
        differences = leaf_differences(base_config, config)
        difference_paths = {row["path"] for row in differences}
        if difference_paths != ALLOWED_CONFIG_DIFFERENCES:
            raise ValueError(
                "unexpected matched-control config differences: "
                f"{sorted(difference_paths)}"
            )
        destination = OUTPUT / f"constant_gate_seed{seed}.json"
        destination.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provenance["runs"].append(
            {
                "training_seed": seed,
                "base_config": str(source),
                "base_config_sha256": sha256(source),
                "constant_config": str(destination),
                "constant_config_sha256": sha256(destination),
                "checkpoint_dir": config["training"]["checkpoint_dir"],
                "final_checkpoint_dir": f"{FINAL_CHECKPOINT_ROOT}/seed{seed}",
                "operational_num_workers": NUM_WORKERS,
                "preserve_single_worker_order": True,
                "config_differences": differences,
            }
        )
    (OUTPUT / "PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
