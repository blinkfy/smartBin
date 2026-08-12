"""Evaluate one checkpoint across seeded sensor degradation scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from evaluate import (
    _checkpoint_metadata,
    _runtime_metadata,
    apply_evaluation_interventions,
)
from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.evaluation import evaluate_model, write_evaluation_artifacts
from mmfc.model import SENSOR_RELIABILITY_EVIDENCE, build_production_model
from mmfc.robustness import (
    aggregate_robustness_runs,
    default_sensor_scenarios,
    summarize_robustness,
    write_multiseed_robustness_artifacts,
    write_robustness_artifacts,
)
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    scenario_names = tuple(default_sensor_scenarios())
    parser = argparse.ArgumentParser(
        description="Evaluate sensor reliability under deterministic degradation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "qwen-material" / "best.pt",
    )
    parser.add_argument(
        "--split", choices=("val", "test"), default="test"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material" / "robustness",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=scenario_names,
        default=None,
        help="Default: evaluate all scenarios; clean is always included",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single degradation seed (default: 2026)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Multiple degradation seeds for mean/std reporting",
    )
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument(
        "--disable-sensor-reliability",
        action="store_true",
        help=(
            "Post-hoc intervention: load the full checkpoint, then bypass "
            "sensor reliability gating during inference"
        ),
    )
    parser.add_argument(
        "--posthoc-modalities",
        choices=("visual", "visual_text", "visual_sensor", "trimodal"),
        default=None,
    )
    parser.add_argument(
        "--disable-sensor-grid-quality",
        action="store_true",
    )
    parser.add_argument(
        "--disable-sensor-cross-modal-quality",
        action="store_true",
    )
    parser.add_argument(
        "--disable-reliability-evidence",
        nargs="+",
        choices=SENSOR_RELIABILITY_EVIDENCE,
        default=(),
    )
    parser.add_argument("--disable-film", action="store_true")
    parser.add_argument("--disable-attention", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Smoke testing only; do not use for paper results",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than 0")
    if args.ece_bins < 2:
        raise ValueError("--ece-bins must be at least 2")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches cannot be negative")
    if args.seed is not None and args.seeds is not None:
        raise ValueError("--seed and --seeds cannot be used together")


def _resolve_seeds(
    seed: int | None, seeds: list[int] | None
) -> list[int]:
    if seed is not None and seeds is not None:
        raise ValueError("--seed and --seeds cannot be used together")
    if seeds is not None:
        return list(dict.fromkeys(seeds))
    return [2026 if seed is None else seed]


def _selected_scenarios(requested: list[str] | None) -> list[str]:
    available = list(default_sensor_scenarios())
    if requested is None:
        return available
    selected = list(dict.fromkeys(requested))
    if "clean" not in selected:
        selected.insert(0, "clean")
    return selected


def main() -> None:
    args = parse_args()
    _validate_args(args)
    config = load_config(args.config)
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    device = torch.device(
        args.device
        or training_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    checkpoint_path = resolve_project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = validated_checkpoint_class_names(checkpoint, config)
    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])
    intervention = apply_evaluation_interventions(
        model,
        disable_sensor_reliability=args.disable_sensor_reliability,
        posthoc_modalities=args.posthoc_modalities,
        disable_sensor_grid_quality=args.disable_sensor_grid_quality,
        disable_sensor_cross_modal_quality=(
            args.disable_sensor_cross_modal_quality
        ),
        disabled_reliability_evidence=tuple(
            args.disable_reliability_evidence
        ),
        disable_film=args.disable_film,
        disable_attention=args.disable_attention,
    )

    manifest_key = f"{args.split}_manifest"
    if manifest_key not in data_config:
        raise ValueError(f"config.data is missing {manifest_key}")
    full_dataset = ManifestDataset(
        data_config[manifest_key],
        class_names=class_names,
        image_size=int(data_config.get("image_size", 640)),
        sensor_manifest=data_config.get("sensor_manifest"),
        preprocess=config.get("preprocess", {}),
    )
    full_sample_count = len(full_dataset)
    dataset: Any = full_dataset
    if args.max_samples is not None:
        dataset = Subset(
            full_dataset, range(min(args.max_samples, full_sample_count))
        )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(training_config.get("batch_size", 16))
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training_config.get("num_workers", 0))
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    output_dir = resolve_project_path(args.output)
    scenarios = default_sensor_scenarios()
    selected = _selected_scenarios(args.scenarios)
    seeds = _resolve_seeds(args.seed, args.seeds)
    multiseed = len(seeds) > 1
    runtime = _runtime_metadata(device)
    checkpoint_metadata = _checkpoint_metadata(checkpoint, checkpoint_path)
    all_results: dict[int, dict[str, dict[str, Any]]] = {}
    global_index = 0
    for seed in seeds:
        seed_output_dir = (
            output_dir / "seeds" / f"seed-{seed}"
            if multiseed
            else output_dir
        )
        results: dict[str, dict[str, Any]] = {}
        for scenario_name in selected:
            scenario_config = scenarios[scenario_name]
            result = evaluate_model(
                model,
                loader,
                device=device,
                class_names=class_names,
                ece_bins=args.ece_bins,
                warmup_batches=(
                    args.warmup_batches if global_index == 0 else 0
                ),
                sensor_degradation=scenario_config,
                degradation_seed=seed,
            )
            result["scenario"] = {
                "name": scenario_name,
                "degradation": scenario_config,
                "seed": seed,
            }
            result["dataset"] = {
                "split": args.split,
                "manifest": str(
                    resolve_project_path(data_config[manifest_key])
                ),
                "evaluated_samples": len(dataset),
                "full_split_samples": full_sample_count,
                "sensor_manifest": str(
                    resolve_project_path(data_config["sensor_manifest"])
                ),
            }
            result["checkpoint"] = checkpoint_metadata
            result["class_names"] = list(class_names)
            result["runtime"] = runtime
            result["posthoc_intervention"] = intervention
            result["evaluation_config"] = {
                "config_path": config["_config_path"],
                "image_size": int(data_config.get("image_size", 640)),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "ece_bins": args.ece_bins,
                "warmup_batches": (
                    args.warmup_batches if global_index == 0 else 0
                ),
                "max_samples": args.max_samples,
            }
            artifacts = write_evaluation_artifacts(
                result,
                seed_output_dir / scenario_name,
                class_names=class_names,
                split=f"{args.split} / {scenario_name} / seed {seed}",
            )
            results[scenario_name] = result
            global_index += 1
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "scenario": scenario_name,
                        "accuracy": result["metrics"]["accuracy"],
                        "macro_f1": result["metrics"]["macro_f1"],
                        "output": artifacts["evaluation_json"],
                    },
                    ensure_ascii=False,
                )
            )

        all_results[seed] = results
        seed_artifacts = write_robustness_artifacts(
            results, seed_output_dir
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "summary": summarize_robustness(results),
                    "artifacts": seed_artifacts,
                },
                ensure_ascii=False,
            )
        )

    if not multiseed:
        return

    summary_artifacts = write_multiseed_robustness_artifacts(
        all_results, output_dir
    )
    print(
        json.dumps(
            {
                "seeds": seeds,
                "summary": aggregate_robustness_runs(all_results),
                "artifacts": summary_artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
