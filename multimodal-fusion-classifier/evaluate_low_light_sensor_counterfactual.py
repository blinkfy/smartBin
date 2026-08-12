"""Paired low-light test with aligned, missing, and permuted sidecars."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.evaluation import compute_classification_metrics
from mmfc.model import build_production_model
from mmfc.training import (
    ManifestDataset,
    load_config,
    load_sensor_grid,
    resolve_project_path,
)


ROOT = Path(__file__).resolve().parent
RUNS = {
    42: (
        "configs/ablations/full_cross_modal_quality.json",
        "checkpoints/ablations/full_cross_modal_quality/best.pt",
    ),
    43: (
        "configs/replicates/full_cross_modal_quality_seed43.json",
        "checkpoints/ablations/v3_seed_replicates/full_cross_modal_quality_seed43/best.pt",
    ),
    44: (
        "configs/replicates/full_cross_modal_quality_seed44.json",
        "checkpoints/ablations/v3_seed_replicates/full_cross_modal_quality_seed44/best.pt",
    ),
}
SENSOR_CONDITIONS = ("aligned", "missing", "permuted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / "qwen-material"
        / "submission-revision"
        / "low-light-sensor-counterfactual-aligned",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--low-light-scale", type=float, default=0.35)
    parser.add_argument("--permutation-seed", type=int, default=20260811)
    return parser.parse_args()


def derangement(length: int, seed: int) -> torch.Tensor:
    if length < 2:
        raise ValueError("permuted-sensor evaluation requires at least two samples")
    generator = torch.Generator().manual_seed(int(seed))
    identity = torch.arange(length)
    for _ in range(1000):
        candidate = torch.randperm(length, generator=generator)
        if bool(torch.all(candidate != identity)):
            return candidate
    shift = 1 + int(seed) % (length - 1)
    return torch.roll(identity, shifts=shift)


def sensor_bank(dataset: ManifestDataset) -> torch.Tensor:
    return torch.stack(
        [
            load_sensor_grid(
                record,
                manifest_dir=dataset.manifest_dir,
                sidecar=dataset.sensor_sidecar,
            )
            for record in dataset.records
        ]
    )


def write_predictions(
    path: Path,
    *,
    sample_ids: list[str],
    source_image_ids: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, sample_id in enumerate(sample_ids):
            target = int(targets[index])
            predicted = int(predictions[index])
            record = {
                "sample_id": sample_id,
                "source_image_id": source_image_ids[index],
                "target_id": target,
                "target": class_names[target],
                "predicted_id": predicted,
                "predicted": class_names[predicted],
                "correct": predicted == target,
                "probabilities": probabilities[index].tolist(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_seed(
    training_seed: int,
    config_value: str,
    checkpoint_value: str,
    *,
    output: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    low_light_scale: float,
    permutation_seed: int,
) -> list[dict[str, Any]]:
    config_path = resolve_project_path(config_value)
    checkpoint_path = resolve_project_path(checkpoint_value)
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = tuple(validated_checkpoint_class_names(checkpoint, config))
    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    data = config["data"]
    dataset = ManifestDataset(
        data["test_manifest"],
        class_names=class_names,
        image_size=int(data.get("image_size", 480)),
        sensor_manifest=data["sensor_manifest"],
        preprocess=config.get("preprocess", {}),
    )
    clean_sensors = sensor_bank(dataset)
    permutation = derangement(len(dataset), permutation_seed)
    permuted_sensors = clean_sensors[permutation]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    collectors = {
        condition: {
            "targets": [],
            "probabilities": [],
            "sample_ids": [],
            "source_image_ids": [],
            "sensor_reliability": [],
        }
        for condition in SENSOR_CONDITIONS
    }
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            size = int(batch["label"].shape[0])
            images = (batch["image"] * low_light_scale).clamp(0.0, 1.0).to(
                device, non_blocking=True
            )
            target = batch["label"].numpy()
            sensor_inputs = {
                "aligned": clean_sensors[offset : offset + size],
                "missing": torch.zeros_like(clean_sensors[offset : offset + size]),
                "permuted": permuted_sensors[offset : offset + size],
            }
            availabilities = {
                "aligned": torch.ones(size, 1),
                "missing": torch.zeros(size, 1),
                "permuted": torch.ones(size, 1),
            }
            for condition in SENSOR_CONDITIONS:
                logits, aux = model(
                    images,
                    sensor_inputs[condition].to(device, non_blocking=True),
                    sensor_available=availabilities[condition].to(
                        device, non_blocking=True
                    ),
                    return_aux=True,
                )
                collector = collectors[condition]
                collector["targets"].append(target.copy())
                collector["probabilities"].append(
                    torch.softmax(logits, dim=1).cpu().numpy()
                )
                collector["sample_ids"].extend(str(value) for value in batch["sample_id"])
                collector["source_image_ids"].extend(
                    str(value) for value in batch["source_image_id"]
                )
                collector["sensor_reliability"].append(
                    aux["sensor_reliability"].detach().cpu().numpy().reshape(-1)
                )
            offset += size

    rows: list[dict[str, Any]] = []
    for condition, collector in collectors.items():
        targets = np.concatenate(collector["targets"])
        probabilities = np.concatenate(collector["probabilities"])
        reliability = np.concatenate(collector["sensor_reliability"])
        computed = compute_classification_metrics(
            targets, probabilities, class_names, ece_bins=15
        )
        metrics = computed["metrics"]
        directory = output / f"seed{training_seed}" / condition
        write_predictions(
            directory / "predictions.jsonl",
            sample_ids=collector["sample_ids"],
            source_image_ids=collector["source_image_ids"],
            targets=targets,
            probabilities=probabilities,
            class_names=class_names,
        )
        evaluation = {
            "valid": True,
            "training_seed": training_seed,
            "condition": condition,
            "metrics": metrics,
            "sensor_reliability_mean": float(reliability.mean()),
            "checkpoint": {
                "path": str(checkpoint_path),
                "epoch": int(checkpoint.get("epoch", -1)),
            },
            "evaluation_config": {
                "config_path": str(config_path),
                "split": "test",
                "sample_count": len(dataset),
                "batch_size": batch_size,
                "num_workers": num_workers,
                "low_light_scale": low_light_scale,
                "permutation_seed": permutation_seed,
                "permutation_fixed_points": int(
                    torch.sum(permutation == torch.arange(len(dataset))).item()
                ),
            },
        }
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "training_seed": training_seed,
                "condition": condition,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "nll": float(metrics["negative_log_likelihood"]),
                "sensor_reliability_mean": float(reliability.mean()),
                "evaluation_path": str(directory / "evaluation.json"),
                "predictions_path": str(directory / "predictions.jsonl"),
            }
        )
    del model, checkpoint, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for condition in SENSOR_CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        result: dict[str, Any] = {
            "condition": condition,
            "training_seed_count": len(selected),
        }
        for metric in ("accuracy", "macro_f1", "nll", "sensor_reliability_mean"):
            values = [float(row[metric]) for row in selected]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_sample_std"] = stdev(values)
        summary.append(result)
    return summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid loader settings")
    if not 0.0 < args.low_light_scale <= 1.0:
        raise ValueError("low-light scale must be in (0,1]")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = args.output.resolve()
    rows: list[dict[str, Any]] = []
    for training_seed, (config, checkpoint) in RUNS.items():
        seed_rows = evaluate_seed(
            training_seed,
            config,
            checkpoint,
            output=output,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            low_light_scale=args.low_light_scale,
            permutation_seed=args.permutation_seed,
        )
        rows.extend(seed_rows)
        print(json.dumps(seed_rows, ensure_ascii=False), flush=True)
    summary = aggregate(rows)
    aligned = next(row for row in summary if row["condition"] == "aligned")
    effects = []
    for condition in ("missing", "permuted"):
        candidate = next(row for row in summary if row["condition"] == condition)
        paired = {
            metric: [
                (
                    100.0
                    * (
                        next(
                            row[metric]
                            for row in rows
                            if row["training_seed"] == seed
                            and row["condition"] == "aligned"
                        )
                        - next(
                            row[metric]
                            for row in rows
                            if row["training_seed"] == seed
                            and row["condition"] == condition
                        )
                    )
                    if metric != "nll"
                    else next(
                        row[metric]
                        for row in rows
                        if row["training_seed"] == seed
                        and row["condition"] == condition
                    )
                    - next(
                        row[metric]
                        for row in rows
                        if row["training_seed"] == seed
                        and row["condition"] == "aligned"
                    )
                )
                for seed in sorted(RUNS)
            ]
            for metric in ("accuracy", "macro_f1", "nll")
        }
        effects.append(
            {
                "contrast": f"aligned_minus_{condition}",
                "accuracy_pp": 100.0
                * (aligned["accuracy_mean"] - candidate["accuracy_mean"]),
                "accuracy_pp_sample_std": stdev(paired["accuracy"]),
                "macro_f1_pp": 100.0
                * (aligned["macro_f1_mean"] - candidate["macro_f1_mean"]),
                "macro_f1_pp_sample_std": stdev(paired["macro_f1"]),
                "nll_reduction": candidate["nll_mean"] - aligned["nll_mean"],
                "nll_reduction_sample_std": stdev(paired["nll"]),
                "per_seed": {
                    "accuracy_pp": paired["accuracy"],
                    "macro_f1_pp": paired["macro_f1"],
                    "nll_reduction": paired["nll"],
                },
            }
        )
    report = {
        "valid": True,
        "purpose": (
            "Test whether low-light utility depends on the sample-aligned archived sidecar."
        ),
        "training_seeds": sorted(RUNS),
        "protocol": {
            "same_checkpoint_within_seed": True,
            "same_low_light_input_within_seed": True,
            "low_light_scale": args.low_light_scale,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "permutation_seed": args.permutation_seed,
            "permutation": "global sample derangement; availability remains one",
        },
        "summary": summary,
        "paired_mean_contrasts": effects,
        "per_seed": rows,
        "claim_boundary": (
            "Internal deterministic stress test; it supports dependence on "
            "sample-sidecar correspondence only for the predefined low-light "
            "transformation and does not establish independent sensing."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "SUMMARY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
