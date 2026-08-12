"""Evaluate matched fallback-semantic interventions under full sensor missingness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.evaluation import evaluate_model, write_evaluation_artifacts
from mmfc.model import build_production_model
from mmfc.robustness import default_sensor_scenarios
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)


DESCRIPTION_MODES = (
    "vision_only",
    "descriptions",
    "class_names",
    "shuffled_descriptions",
)
SEMANTIC_MODES = (*DESCRIPTION_MODES, "class_id_keys")
METRICS = (
    "accuracy",
    "macro_f1",
    "latent_mse",
    "grid_mae",
    "grid_mse",
    "latent_normalized_mse",
    "latent_r2",
    "grid_normalized_mse",
    "grid_r2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched semantic interventions for sensor fallback"
    )
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--class-id-configs",
        nargs="*",
        type=Path,
        default=(),
        help="Separately trained fixed orthogonal class-ID key controls",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "semantic-guidance-v5"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--degradation-seed", type=int, default=2026)
    return parser.parse_args()


def _metric_values(result: dict[str, Any]) -> dict[str, float]:
    reconstruction = result["fallback_reconstruction"]
    return {
        "accuracy": float(result["metrics"]["accuracy"]),
        "macro_f1": float(result["metrics"]["macro_f1"]),
        "latent_mse": float(reconstruction["latent_mse"]),
        "grid_mae": float(reconstruction["grid_mae"]),
        "grid_mse": float(reconstruction["grid_mse"]),
        "latent_normalized_mse": float(reconstruction["latent_normalized_mse"]),
        "latent_r2": float(reconstruction["latent_r2"]),
        "grid_normalized_mse": float(reconstruction["grid_normalized_mse"]),
        "grid_r2": float(reconstruction["grid_r2"]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for mode in SEMANTIC_MODES:
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        aggregate[mode] = {"run_count": len(selected)}
        for metric in METRICS:
            values = np.asarray(
                [row[metric] for row in selected],
                dtype=np.float64,
            )
            aggregate[mode][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "values": values.tolist(),
            }
    correct = aggregate["descriptions"]
    comparisons = {}
    for reference in ("vision_only", "class_names", "shuffled_descriptions"):
        comparisons[f"descriptions_vs_{reference}"] = {
            metric: (
                correct[metric]["mean"]
                - aggregate[reference][metric]["mean"]
            )
            for metric in METRICS
        }
    if "class_id_keys" in aggregate:
        comparisons["descriptions_vs_class_id_keys"] = {
            metric: (
                correct[metric]["mean"]
                - aggregate["class_id_keys"][metric]["mean"]
            )
            for metric in METRICS
        }
    claim_references = ["vision_only", "shuffled_descriptions"]
    if "class_id_keys" in aggregate:
        claim_references.append("class_id_keys")
    claim_supported = all(
        comparisons[f"descriptions_vs_{reference}"]["accuracy"] > 0.0
        and comparisons[f"descriptions_vs_{reference}"]["macro_f1"] > 0.0
        and comparisons[f"descriptions_vs_{reference}"]["latent_mse"] < 0.0
        and comparisons[f"descriptions_vs_{reference}"]["grid_mae"] < 0.0
        and comparisons[f"descriptions_vs_{reference}"]["grid_mse"] < 0.0
        for reference in claim_references
    )
    return {
        "modes": aggregate,
        "comparisons": comparisons,
        "semantic_guidance_claim_supported": claim_supported,
        "claim_rule": (
            "Correct descriptions must improve missing-100 accuracy and macro-F1 "
            "and reduce latent MSE, grid MAE, and grid MSE versus vision-only, "
            "shuffled descriptions, and the nonsemantic class-ID key control when present."
        ),
    }


def _write_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = ("seed", "mode", *METRICS, "checkpoint")
    with (output_dir / "runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)

    lines = [
        "# Class-Memory Fallback Control Study",
        "",
        "The first four conditions are paired inference interventions on the same "
        "three description-key checkpoints. The fixed orthogonal class-ID condition "
        "uses three separately trained, protocol-matched checkpoints; it is therefore "
        "a training-level control rather than a fixed-weight intervention.",
        "",
        "| Fallback conditioning | Missing-100 Acc. | Missing-100 Macro F1 | "
        "Latent MSE | Grid MAE | Grid MSE | Latent NMSE | Latent R2 | Grid NMSE | Grid R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "vision_only": "Vision only",
        "descriptions": "Vision + correct descriptions",
        "class_names": "Vision + class names only",
        "shuffled_descriptions": "Vision + shuffled descriptions",
        "class_id_keys": "Vision + fixed orthogonal class-ID keys",
    }
    for mode in summary["modes"]:
        values = summary["modes"][mode]
        formatted = []
        for metric in METRICS:
            mean = values[metric]["mean"]
            std = values[metric]["std"]
            formatted.append(f"{mean:.4f} +/- {std:.4f}")
        lines.append(f"| {labels[mode]} | " + " | ".join(formatted) + " |")
    lines.extend(
        [
            "",
            "Description-specific semantic claim check: **"
            + (
                "supported"
                if summary["semantic_guidance_claim_supported"]
                else "not supported"
            )
            + "** under the preregistered directional rule.",
            "",
            "Lower is better for latent MSE, grid MAE, and grid MSE. "
            "Classification is evaluated with the physical array set to 100% "
            "missing; reconstruction targets always come from the paired clean "
            "physical sensor observation.",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = resolve_project_path(args.output) / args.split
    rows: list[dict[str, Any]] = []
    for config_path in args.configs:
        config = load_config(config_path)
        training = config["training"]
        seed = int(training["seed"])
        checkpoint_path = resolve_project_path(training["checkpoint_dir"]) / "best.pt"
        checkpoint = load_checkpoint(checkpoint_path, device)
        class_names = validated_checkpoint_class_names(checkpoint, config)
        model = build_production_model(config).to(device)
        load_model_state_compatible(model, checkpoint["model_state"])

        data = config["data"]
        dataset = ManifestDataset(
            data[f"{args.split}_manifest"],
            class_names=class_names,
            image_size=int(data.get("image_size", 640)),
            sensor_manifest=data.get("sensor_manifest"),
            preprocess=config.get("preprocess", {}),
        )
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
        for mode_index, mode in enumerate(DESCRIPTION_MODES):
            model.set_fallback_semantic_mode(mode)
            result = evaluate_model(
                model,
                loader,
                device=device,
                class_names=class_names,
                warmup_batches=3 if mode_index == 0 else 0,
                sensor_degradation=default_sensor_scenarios()["missing_100"],
                degradation_seed=args.degradation_seed,
            )
            result["semantic_intervention"] = {
                "mode": mode,
                "scope": "fallback conditioning only",
                "weights_changed": False,
            }
            result["checkpoint"] = str(checkpoint_path)
            result["seed"] = seed
            write_evaluation_artifacts(
                result,
                output_dir / f"seed{seed}" / mode,
                class_names=class_names,
                split=f"{args.split} missing-100 {mode}",
            )
            row = {
                "seed": seed,
                "mode": mode,
                **_metric_values(result),
                "checkpoint": str(checkpoint_path),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    for config_path in args.class_id_configs:
        config = load_config(config_path)
        training = config["training"]
        seed = int(training["seed"])
        checkpoint_path = resolve_project_path(training["checkpoint_dir"]) / "best.pt"
        checkpoint = load_checkpoint(checkpoint_path, device)
        class_names = validated_checkpoint_class_names(checkpoint, config)
        model = build_production_model(config).to(device)
        load_model_state_compatible(model, checkpoint["model_state"])
        model.set_fallback_semantic_mode("class_id_keys")
        data = config["data"]
        dataset = ManifestDataset(
            data[f"{args.split}_manifest"],
            class_names=class_names,
            image_size=int(data.get("image_size", 640)),
            sensor_manifest=data.get("sensor_manifest"),
            preprocess=config.get("preprocess", {}),
        )
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
            warmup_batches=3,
            sensor_degradation=default_sensor_scenarios()["missing_100"],
            degradation_seed=args.degradation_seed,
        )
        result["semantic_intervention"] = {
            "mode": "class_id_keys",
            "scope": "separately trained non-semantic fallback control",
            "weights_changed": True,
        }
        result["checkpoint"] = str(checkpoint_path)
        result["seed"] = seed
        write_evaluation_artifacts(
            result,
            output_dir / f"seed{seed}" / "class_id_keys",
            class_names=class_names,
            split=f"{args.split} missing-100 class_id_keys",
        )
        row = {
            "seed": seed,
            "mode": "class_id_keys",
            **_metric_values(result),
            "checkpoint": str(checkpoint_path),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "protocol": {
            "split": args.split,
            "sensor_condition": "missing_100",
            "degradation_seed": args.degradation_seed,
            "run_count_per_condition": len(args.configs),
            "class_id_run_count": len(args.class_id_configs),
            "same_checkpoint_across_modes_within_seed": True,
            "classifier_text_branch_fixed": True,
        },
        **_aggregate(rows),
        "runs": rows,
    }
    _write_summary(output_dir, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
