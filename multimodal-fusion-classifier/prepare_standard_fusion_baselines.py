"""Generate matched three-seed configs for conventional fusion baselines."""

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
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "standard-fusion-baselines"
METHODS = ("concat_mlp", "gmu", "lmf")
SEEDS = (42, 43, 44)
SOURCE_METADATA = {
    "concat_mlp": {
        "display_name": "Concat-MLP",
        "paper": "Multimodal Deep Learning",
        "venue": "ICML 2011",
        "paper_url": "https://icml.cc/2011/papers/399_icmlpaper.pdf",
        "official_repository": "not_applicable",
        "inspected_commit": "not_applicable",
        "adaptation": (
            "same YOLO/CLIP candidate-prototype/sensor features; concatenate "
            "the three 512-dimensional branches and classify with a two-layer MLP"
        ),
    },
    "gmu": {
        "display_name": "GMU",
        "paper": "Gated Multimodal Units for Information Fusion",
        "venue": "arXiv 2017",
        "paper_url": "https://arxiv.org/abs/1702.01992",
        "official_repository": "not_available_from_primary_source",
        "inspected_commit": "not_applicable",
        "fusion_hidden_dim": 512,
        "adaptation": (
            "same YOLO/CLIP candidate-prototype/sensor features; use "
            "modality-specific candidates and feature-wise softmax gates"
        ),
    },
    "lmf": {
        "display_name": "LMF",
        "paper": (
            "Efficient Low-rank Multimodal Fusion with Modality-Specific Factors"
        ),
        "venue": "ACL 2018",
        "paper_url": "https://aclanthology.org/P18-1209/",
        "official_repository": (
            "https://github.com/Justin1904/Low-rank-Multimodal-Fusion"
        ),
        "inspected_commit": "repository_source_checked_2026-08-02",
        "rank": 4,
        "adaptation": (
            "same YOLO/CLIP candidate-prototype/sensor features; apply a "
            "rank-4 modality-factorized multiplicative classifier"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare matched conventional multimodal fusion baselines"
    )
    parser.add_argument("--reference", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=4)
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


def build_configs(
    reference_path: Path = REFERENCE_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    num_workers: int = 4,
) -> list[Path]:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    reference = _read_json(_resolve(reference_path))
    output_dir = _resolve(output_dir)
    created: list[Path] = []
    for method in METHODS:
        metadata = SOURCE_METADATA[method]
        for seed in SEEDS:
            config = deepcopy(reference)
            config["recent_baseline"] = {
                "method": method,
                **metadata,
                "independent_text_observation_claimed": False,
            }
            config["training"].update(
                {
                    "seed": seed,
                    "num_workers": num_workers,
                    "prefetch_factor": 2,
                    "selection_metric": "accuracy",
                    "checkpoint_dir": (
                        "checkpoints/standard-fusion-baselines/"
                        f"{method}/seed{seed}"
                    ),
                }
            )
            config["training"].pop("resume", None)
            config["training"].pop("resume_checkpoint", None)
            config["experiment"] = {
                "name": f"{method}_same_input_seed{seed}",
                "group": "standard_fusion_baseline",
                "description": (
                    f"{metadata['display_name']} same-input adaptation"
                ),
                "formal_retraining_required": True,
                "reference_checkpoint": None,
                "method_source_is_external": True,
                "implementation_is_project_adaptation": True,
                "single_run_reference": True,
            }
            path = output_dir / f"{method}_seed{seed}.json"
            _atomic_write(path, config)
            created.append(path)
    return created


def main() -> None:
    args = parse_args()
    created = build_configs(
        args.reference,
        args.output,
        num_workers=args.num_workers,
    )
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
