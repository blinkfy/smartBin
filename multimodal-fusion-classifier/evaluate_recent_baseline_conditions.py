"""Evaluate one QMF/TMC baseline under the frozen 11 sensor conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_recent_baseline import _validate_checkpoint
from evaluate_unseen_sensor_conditions import (
    BOOTSTRAP_REPEATS,
    CONDITIONS,
    _sensor_bank,
    apply_condition,
    build_derangement,
)
from mmfc.checkpoints import load_checkpoint
from mmfc.evaluation import (
    compute_classification_metrics,
    grouped_paired_bootstrap_differences,
)
from mmfc.model import PROJECT_ROOT
from mmfc.recent_baselines import build_recent_baseline_model
from mmfc.training import ManifestDataset, load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a recent baseline under frozen sensor conditions"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=20261201)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _condition_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    computed = compute_classification_metrics(
        targets,
        probabilities,
        class_names,
        ece_bins=15,
    )
    predictions = computed.pop("predictions")
    return computed, predictions


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 2:
        raise ValueError("--bootstrap-repeats must be at least 2")
    config_path = resolve_project_path(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    output_path = resolve_project_path(args.output)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite condition evaluation")
    config = load_config(config_path)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = _validate_checkpoint(checkpoint, config)
    model = build_recent_baseline_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    data = config["data"]
    dataset = ManifestDataset(
        data["test_manifest"],
        class_names=class_names,
        image_size=int(data.get("image_size", 480)),
        sensor_manifest=data["sensor_manifest"],
        preprocess=config.get("preprocess", {}),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    sensor_bank = _sensor_bank(dataset)
    sample_order = [
        str(record.get("sample_id", record.get("id", "")))
        for record in dataset.records
    ]
    derangement = build_derangement(sample_order)
    collectors = {
        condition: {
            "targets": [],
            "probabilities": [],
            "sample_ids": [],
            "source_image_ids": [],
            "quality": [],
            "availability": [],
            "sensor_weight": [],
            "grid_change": [],
        }
        for condition in CONDITIONS
    }
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            clean = batch["sensor"]
            labels = batch["label"].numpy()
            sample_ids = [str(value) for value in batch["sample_id"]]
            source_ids = [str(value) for value in batch["source_image_id"]]
            for condition in CONDITIONS:
                observed, availability, quality = apply_condition(
                    condition,
                    clean,
                    sample_ids,
                    sensor_bank=sensor_bank,
                    derangement=derangement,
                )
                logits, auxiliary = model(
                    images,
                    observed.to(device, non_blocking=True),
                    sensor_available=availability.to(device, non_blocking=True),
                    return_aux=True,
                )
                collector = collectors[condition]
                collector["targets"].append(labels.copy())
                collector["probabilities"].append(
                    torch.softmax(logits, dim=1).cpu().numpy()
                )
                collector["sample_ids"].extend(sample_ids)
                collector["source_image_ids"].extend(source_ids)
                collector["quality"].append(quality.numpy().reshape(-1))
                collector["availability"].append(
                    availability.numpy().reshape(-1)
                )
                collector["sensor_weight"].append(
                    auxiliary["modality_weights"][:, 2]
                    .detach()
                    .cpu()
                    .numpy()
                )
                collector["grid_change"].append(
                    (observed - clean)
                    .abs()
                    .mean(dim=(1, 2, 3))
                    .numpy()
                )

    results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    condition_arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for condition, collector in collectors.items():
        targets = np.concatenate(collector["targets"])
        probabilities = np.concatenate(collector["probabilities"])
        quality = np.concatenate(collector["quality"])
        availability = np.concatenate(collector["availability"])
        sensor_weight = np.concatenate(collector["sensor_weight"])
        grid_change = np.concatenate(collector["grid_change"])
        metrics, predictions = _condition_metrics(
            targets,
            probabilities,
            class_names,
        )
        results[condition] = {
            "definition": CONDITIONS[condition],
            **metrics,
            "sensor_condition": {
                "reported_availability_mean": float(availability.mean()),
                "true_quality_mean": float(quality.mean()),
                "mean_absolute_grid_change": float(grid_change.mean()),
                "sensor_branch_weight_mean": float(sensor_weight.mean()),
            },
        }
        condition_arrays[condition] = (
            targets,
            probabilities,
            collector["source_image_ids"],
        )
        prediction_rows.extend(
            {
                "condition": condition,
                "sample_id": collector["sample_ids"][index],
                "source_image_id": collector["source_image_ids"][index],
                "target_id": int(targets[index]),
                "predicted_id": int(predictions[index]),
                "correct": bool(predictions[index] == targets[index]),
                "reported_availability": float(availability[index]),
                "true_quality": float(quality[index]),
                "sensor_branch_weight": float(sensor_weight[index]),
                "probabilities": probabilities[index].tolist(),
            }
            for index in range(targets.size)
        )

    clean_targets, clean_probabilities, clean_sources = condition_arrays["clean"]
    paired = {}
    for index, condition in enumerate(CONDITIONS):
        if condition == "clean":
            continue
        targets, probabilities, source_ids = condition_arrays[condition]
        if not np.array_equal(clean_targets, targets) or source_ids != clean_sources:
            raise RuntimeError("condition sample order changed")
        paired[condition] = grouped_paired_bootstrap_differences(
            clean_targets,
            clean_probabilities,
            probabilities,
            clean_sources,
            class_names,
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed + index,
        )

    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "baseline_method": config["recent_baseline"]["method"],
        "training_seed": int(config["training"]["seed"]),
        "sample_count": len(dataset),
        "source_image_count": len(set(clean_sources)),
        "class_names": list(class_names),
        "conditions": results,
        "condition_minus_clean_paired_bootstrap": paired,
        "derangement_self_assignment_count": sum(
            source == target for source, target in derangement.items()
        ),
        "source_bindings": {
            "config": _binding(config_path),
            "checkpoint": _binding(checkpoint_path),
            "test_manifest": _binding(
                resolve_project_path(data["test_manifest"])
            ),
            "sensor_manifest": _binding(
                resolve_project_path(data["sensor_manifest"])
            ),
        },
        "interpretation": {
            "sensor_weight": (
                "normalized baseline sensor-branch fusion weight; it is not "
                "the proposed v3 sensor reliability output"
            ),
            "visual_text_branch": (
                "candidate class prototypes queried by visual features; no "
                "independent per-object text observation is claimed"
            ),
        },
        "claim_boundary": (
            "Deterministic software interventions on image-conditioned "
            "sidecars and the reused internal test set; not hardware or "
            "external validation."
        ),
    }
    _atomic_text(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(
        output_path.with_name("condition_predictions.jsonl"),
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in prediction_rows
        ),
    )
    print(
        json.dumps(
            {
                "valid": True,
                "baseline_method": report["baseline_method"],
                "training_seed": report["training_seed"],
                "condition_count": len(CONDITIONS),
                "sample_count": len(dataset),
                "output": output_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
