"""Train and evaluate a lightweight raw-grid sensor quality probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mmfc.quality_probe import (
    evaluate_quality_probe,
    fit_quality_probe,
    load_split_sensor_grids,
    write_quality_probe_artifacts,
)
from mmfc.training import PROJECT_ROOT, load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw-grid quality features")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "ablations"
        / "full_quality_aware.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "quality-probe",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--degradation-seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config.get("data", {})
    degradation = config.get("robustness", {}).get(
        "sensor_degradation", {}
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_grids = load_split_sensor_grids(
        data["train_manifest"], data["sensor_manifest"]
    )
    test_grids = load_split_sensor_grids(
        data["test_manifest"], data["sensor_manifest"]
    )
    model, history = fit_quality_probe(
        train_grids,
        degradation,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=int(
            config.get("model", {}).get(
                "sensor_grid_quality_hidden_dim", 16
            )
        ),
    )
    rows = evaluate_quality_probe(
        model,
        test_grids,
        device=device,
        seed=args.degradation_seed,
    )
    metadata = {
        "config_path": config["_config_path"],
        "train_samples": int(train_grids.shape[0]),
        "test_samples": int(test_grids.shape[0]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_seed": args.seed,
        "degradation_seed": args.degradation_seed,
        "device": str(device),
        "probe_scope": "raw_grid_quality_only",
    }
    artifacts = write_quality_probe_artifacts(
        model,
        history,
        rows,
        resolve_project_path(args.output),
        metadata=metadata,
    )
    print(
        json.dumps(
            {"summary": rows, "artifacts": artifacts},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
