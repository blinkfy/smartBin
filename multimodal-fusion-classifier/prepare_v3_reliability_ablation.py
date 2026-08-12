"""Prepare matched-seed v3 configs without the reliability subsystem."""

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
DEFAULT_PERFORMANCE_PROVENANCE = (
    PROJECT_ROOT
    / "configs"
    / "performance"
    / "full_cross_modal_quality_workers4.provenance.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "configs" / "confirmatory-ablations"
SEEDS = (42, 43, 44)
MODEL_REMOVALS = (
    "sensor_reliability_enabled",
    "sensor_grid_quality_enabled",
    "sensor_cross_modal_quality_enabled",
)
LOSS_REMOVALS = (
    "reconstruction_weight",
    "calibration_weight",
    "variance_prior_weight",
    "sensor_grid_reconstruction_weight",
    "calibration_ranking_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare v3 reliability-subsystem ablation configs"
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--performance-provenance",
        type=Path,
        default=DEFAULT_PERFORMANCE_PROVENANCE,
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


def normalized_reliability_ablation_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    model = value.setdefault("model", {})
    for name in MODEL_REMOVALS:
        model.pop(name, None)
    loss = value.setdefault("loss", {})
    for name in LOSS_REMOVALS:
        loss.pop(name, None)
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
        "matched_training_seed",
        "performance_evidence",
        "reference_checkpoint",
    ):
        experiment.pop(name, None)
    return value


def build_ablation_config(
    base: Mapping[str, Any],
    *,
    seed: int,
    performance_evidence: str,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unsupported matched seed: {seed}")
    value = copy.deepcopy(dict(base))
    for name in MODEL_REMOVALS:
        value["model"][name] = False
    for name in LOSS_REMOVALS:
        value["loss"][name] = 0.0
    experiment_name = f"v3_no_reliability_seed{seed}"
    value["training"]["seed"] = seed
    value["training"]["num_workers"] = 4
    value["training"]["checkpoint_dir"] = (
        "checkpoints/confirmatory-ablations/"
        f"v3_no_reliability/seed{seed}"
    )
    value["experiment"] = {
        **value.get("experiment", {}),
        "name": experiment_name,
        "group": "confirmatory_component_ablation",
        "description": (
            "Matched-seed v3 ablation without latent prediction, reliability "
            "gating, grid quality, cross-modal grid quality, or their "
            "auxiliary losses"
        ),
        "ablation_of": "full_cross_modal_quality",
        "ablation_scope": "entire_sensor_reliability_subsystem",
        "matched_training_seed": seed,
        "performance_evidence": performance_evidence,
        "reference_checkpoint": None,
    }
    if normalized_reliability_ablation_config(value) != (
        normalized_reliability_ablation_config(base)
    ):
        raise ValueError("ablation config changes unauthorized scientific fields")
    return value


def prepare_configs(
    *,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    performance_provenance_path: Path = DEFAULT_PERFORMANCE_PROVENANCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    base_config_path = _resolve(base_config_path)
    performance_provenance_path = _resolve(performance_provenance_path)
    output_dir = _resolve(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("confirmatory-ablation config directory is not empty")
    base = _read_json(base_config_path)
    performance = _read_json(performance_provenance_path)
    if (
        base.get("experiment", {}).get("name") != "full_cross_modal_quality"
        or base.get("training", {}).get("epochs") != 100
        or base.get("training", {}).get("batch_size") != 16
    ):
        raise ValueError("v3 base config identity mismatch")
    if (
        performance.get("state") != "complete"
        or performance.get("valid") is not True
        or performance.get("authorized_changes", {})
        .get("training.num_workers", {})
        .get("to")
        != 4
        or performance.get("equivalence", {}).get("sample_order_identical")
        is not True
        or performance.get("equivalence", {})
        .get("image_sensor_and_label_bytes_identical")
        is not True
    ):
        raise ValueError("4-worker performance equivalence is not valid")
    performance_relative = performance_provenance_path.relative_to(
        PROJECT_ROOT
    ).as_posix()
    outputs = {}
    for seed in SEEDS:
        path = output_dir / f"v3_no_reliability_seed{seed}.json"
        config = build_ablation_config(
            base,
            seed=seed,
            performance_evidence=performance_relative,
        )
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
        "base_config": base_config_path.relative_to(PROJECT_ROOT).as_posix(),
        "base_config_sha256": _sha256(base_config_path),
        "performance_provenance": performance_relative,
        "performance_provenance_sha256": _sha256(
            performance_provenance_path
        ),
        "seeds": list(SEEDS),
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    report = prepare_configs(
        base_config_path=args.base_config,
        performance_provenance_path=args.performance_provenance,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
