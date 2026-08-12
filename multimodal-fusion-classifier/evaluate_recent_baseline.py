"""Evaluate one same-input multimodal fusion baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mmfc.checkpoints import load_checkpoint
from mmfc.evaluation import (
    evaluate_model,
    grouped_bootstrap_classification_intervals,
    write_evaluation_artifacts,
)
from mmfc.model import PROJECT_ROOT
from mmfc.recent_baseline_training import _config_fingerprint
from mmfc.recent_baselines import build_recent_baseline_model
from mmfc.training import ManifestDataset, load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a same-input multimodal fusion baseline"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=0)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "max_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.ece_bins < 2 or args.warmup_batches < 0:
        raise ValueError("invalid evaluation bin/warmup setting")
    if args.bootstrap_repeats < 0 or args.bootstrap_repeats == 1:
        raise ValueError("bootstrap repeats must be 0 or at least 2")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0,1)")


def _validate_checkpoint(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, ...]:
    expected_method = str(config["recent_baseline"]["method"])
    if checkpoint.get("baseline_method") != expected_method:
        raise ValueError("checkpoint baseline method differs from config")
    if checkpoint.get("config_fingerprint") != _config_fingerprint(config):
        raise ValueError("checkpoint baseline config fingerprint mismatch")
    configured = tuple(
        value["name"] if isinstance(value, dict) else str(value)
        for value in config["classes"]
    )
    if tuple(checkpoint.get("class_names", [])) != configured:
        raise ValueError("checkpoint class order differs from config")
    if checkpoint.get("checkpoint_role") != "best_evaluation":
        raise ValueError("formal evaluation requires best_evaluation checkpoint")
    return configured


def main() -> None:
    args = parse_args()
    _validate_args(args)
    config = load_config(args.config)
    training = config.get("training", {})
    data = config.get("data", {})
    device = torch.device(
        args.device
        or training.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = _validate_checkpoint(checkpoint, config)
    model = build_recent_baseline_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    manifest_key = f"{args.split}_manifest"
    dataset = ManifestDataset(
        data[manifest_key],
        class_names=class_names,
        image_size=int(data.get("image_size", 480)),
        sensor_manifest=data.get("sensor_manifest"),
        preprocess=config.get("preprocess", {}),
    )
    full_samples = len(dataset)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else int(training.get("batch_size", 16))
        ),
        shuffle=False,
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(training.get("num_workers", 0))
        ),
        pin_memory=device.type == "cuda",
    )
    result = evaluate_model(
        model,
        loader,
        device=device,
        class_names=class_names,
        ece_bins=args.ece_bins,
        warmup_batches=args.warmup_batches,
    )
    if args.bootstrap_repeats:
        records = result["prediction_records"]
        result["confidence_intervals"] = (
            grouped_bootstrap_classification_intervals(
                np.asarray(
                    [record["target_id"] for record in records],
                    dtype=np.int64,
                ),
                np.asarray(
                    [record["probabilities"] for record in records],
                    dtype=np.float64,
                ),
                [record["source_image_id"] for record in records],
                class_names,
                repeats=args.bootstrap_repeats,
                confidence=args.bootstrap_confidence,
                seed=args.bootstrap_seed,
            )
        )
    result["dataset"] = {
        "split": args.split,
        "manifest": str(resolve_project_path(data[manifest_key])),
        "evaluated_samples": len(dataset),
        "full_split_samples": full_samples,
        "sensor_manifest": str(resolve_project_path(data["sensor_manifest"])),
        "sensor_degradation_enabled": False,
    }
    result["checkpoint"] = {
        "path": str(checkpoint_path.resolve()),
        "size_bytes": checkpoint_path.stat().st_size,
        "epoch": checkpoint.get("epoch"),
        "selection_val_accuracy": checkpoint.get("val_accuracy"),
        "selection_val_macro_f1": checkpoint.get("val_macro_f1"),
        "role": checkpoint.get("checkpoint_role"),
    }
    result["class_names"] = list(class_names)
    result["baseline"] = {
        "method": model.method,
        "adaptation": config["recent_baseline"]["adaptation"],
        "source": {
            key: config["recent_baseline"][key]
            for key in (
                "paper",
                "venue",
                "paper_url",
                "official_repository",
                "inspected_commit",
            )
        },
        "diagnostic_mapping": (
            "sensor_reliability in generic artifacts is the normalized sensor "
            "branch fusion weight, not the proposed v3 reliability target"
        ),
        "independent_text_observation_claimed": False,
    }
    result["evaluation_config"] = {
        "config_path": config["_config_path"],
        "batch_size": loader.batch_size,
        "num_workers": loader.num_workers,
        "ece_bins": args.ece_bins,
        "warmup_batches": args.warmup_batches,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_confidence": args.bootstrap_confidence,
        "bootstrap_seed": args.bootstrap_seed,
        "max_samples": args.max_samples,
    }
    output = resolve_project_path(args.output)
    artifacts = write_evaluation_artifacts(
        result,
        output,
        class_names=class_names,
        split=args.split,
    )
    print(
        json.dumps(
            {
                "metrics": result["metrics"],
                "baseline": result["baseline"],
                "checkpoint": result["checkpoint"],
                "dataset": result["dataset"],
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
