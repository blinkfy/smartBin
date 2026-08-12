"""Generate matched three-seed QMF/TMC baseline configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_CONFIG = (
    PROJECT_ROOT / "configs" / "ablations" / "full_cross_modal_quality.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "recent-robust-baselines"
METHODS = ("qmf", "tmc")
SEEDS = (42, 43, 44)
SOURCE_METADATA = {
    "qmf": {
        "paper": "Provable Dynamic Fusion for Low-Quality Multimodal Data",
        "venue": "ICML 2023",
        "paper_url": (
            "https://proceedings.mlr.press/v202/zhang23ar.html"
        ),
        "official_repository": "https://github.com/QingyangZhang/QMF",
        "inspected_commit": "fe6c4c6ef7cb23f0a89594ee413d485f1854268b",
        "energy_temperature": 10.0,
        "ranking_weight": 0.1,
    },
    "tmc": {
        "paper": "Trusted Multi-View Classification",
        "venue": "ICLR 2021",
        "paper_url": "https://openreview.net/forum?id=OOsR8BzCnl5",
        "official_repository": "https://github.com/hanmenghan/TMC",
        "inspected_commit": "a3272b8746861c76a3461943b5eee51df5b5a8fe",
        "annealing_epochs": 10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare matched recent robust-fusion baseline configs"
    )
    parser.add_argument("--reference", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized_matched_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    return {
        "classes": config.get("classes"),
        "visual_text_sensor_encoders": {
            key: config.get("model", {}).get(key)
            for key in (
                "yolo_weights",
                "clip_model",
                "clip_local_files_only",
                "feature_dim",
                "dropout",
                "scale_prior",
                "text_temperature",
                "freeze_yolo_layers",
                "freeze_clip_text_layers",
            )
        },
        "data": config.get("data"),
        "preprocess": config.get("preprocess"),
        "robustness": config.get("robustness"),
        "training": {
            key: training.get(key)
            for key in (
                "batch_size",
                "learning_rate",
                "weight_decay",
                "epochs",
                "warmup_epochs",
                "num_workers",
                "selection_metric",
            )
        },
    }


def build_configs(
    reference_path: Path = REFERENCE_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> list[Path]:
    reference_path = _resolve(reference_path)
    output_dir = _resolve(output_dir)
    reference = _read_json(reference_path)
    training = reference.get("training", {})
    required_training = {
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "warmup_epochs": 5,
        "num_workers": 0,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"reference training.{key} must remain {expected!r}"
            )

    created: list[Path] = []
    for method in METHODS:
        for seed in SEEDS:
            config = deepcopy(reference)
            config["recent_baseline"] = {
                "method": method,
                **SOURCE_METADATA[method],
                "adaptation": (
                    "same YOLO/CLIP candidate-prototype/sensor inputs; "
                    "three decision branches are visual, visual_text, sensor"
                ),
                "independent_text_observation_claimed": False,
            }
            config["training"].update(
                {
                    "seed": seed,
                    "selection_metric": "accuracy",
                    "checkpoint_dir": (
                        "checkpoints/recent-robust-baselines/"
                        f"{method}/seed{seed}"
                    ),
                }
            )
            config["training"].pop("resume", None)
            config["training"].pop("resume_checkpoint", None)
            config["experiment"] = {
                "name": f"{method}_same_input_seed{seed}",
                "group": "recent_robust_fusion_baseline",
                "description": (
                    f"{SOURCE_METADATA[method]['paper']} same-input adaptation"
                ),
                "formal_retraining_required": True,
                "reference_checkpoint": None,
                "method_source_is_external": True,
                "implementation_is_project_adaptation": True,
            }
            expected = normalized_matched_settings(reference)
            expected["training"]["selection_metric"] = "accuracy"
            if normalized_matched_settings(config) != expected:
                raise RuntimeError("matched baseline settings drifted")
            path = output_dir / f"{method}_seed{seed}.json"
            _atomic_write(path, config)
            created.append(path)
    return created


def main() -> None:
    args = parse_args()
    created = build_configs(args.reference, args.output)
    print(
        json.dumps(
            {
                "valid": True,
                "methods": list(METHODS),
                "seeds": list(SEEDS),
                "run_count": len(created),
                "configs": [
                    path.relative_to(PROJECT_ROOT).as_posix()
                    for path in created
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
