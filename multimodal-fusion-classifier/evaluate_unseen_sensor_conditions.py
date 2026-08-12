"""Evaluate full v3 models under unseen sensor and availability failures."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.evaluation import (
    compute_classification_metrics,
    grouped_paired_bootstrap_differences,
    reliability_quality_metrics,
)
from mmfc.model import build_production_model
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    load_sensor_grid,
    resolve_project_path,
)


REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "unseen-sensor-conditions-v1"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_OUTPUT = REPORT_ROOT / "OUTCOME.json"
CORRUPTION_SEED = 20261101
BOOTSTRAP_REPEATS = 2000
SEED_RUNS = {
    42: {
        "config": PROJECT_ROOT
        / "configs"
        / "ablations"
        / "full_cross_modal_quality.json",
        "checkpoint": PROJECT_ROOT
        / "checkpoints"
        / "ablations"
        / "full_cross_modal_quality"
        / "best.pt",
    },
    43: {
        "config": PROJECT_ROOT
        / "configs"
        / "replicates"
        / "full_cross_modal_quality_seed43.json",
        "checkpoint": PROJECT_ROOT
        / "checkpoints"
        / "ablations"
        / "v3_seed_replicates"
        / "full_cross_modal_quality_seed43"
        / "best.pt",
    },
    44: {
        "config": PROJECT_ROOT
        / "configs"
        / "replicates"
        / "full_cross_modal_quality_seed44.json",
        "checkpoint": PROJECT_ROOT
        / "checkpoints"
        / "ablations"
        / "v3_seed_replicates"
        / "full_cross_modal_quality_seed44"
        / "best.pt",
    },
}
CONDITIONS = {
    "clean": {
        "category": "reference",
        "seen_during_training": True,
    },
    "noise_030_out_of_range": {
        "category": "severity_extrapolation",
        "seen_during_training": False,
        "detail": "Gaussian sigma 0.30; training maximum was 0.15",
    },
    "bias_030_out_of_range": {
        "category": "severity_extrapolation",
        "seen_during_training": False,
        "detail": "directional drift amplitude 0.30; training maximum was 0.15",
    },
    "impulse_025": {
        "category": "unseen_corruption",
        "seen_during_training": False,
        "detail": "25% cells replaced by deterministic binary impulses",
    },
    "gamma_2": {
        "category": "unseen_corruption",
        "seen_during_training": False,
        "detail": "nonlinear response compression x -> x^2",
    },
    "spatial_permutation": {
        "category": "unseen_corruption",
        "seen_during_training": False,
        "detail": "sample-specific permutation of the 16 array positions",
    },
    "foreign_sample": {
        "category": "unseen_corruption",
        "seen_during_training": False,
        "detail": "one-cycle derangement assigns another test object's grid",
    },
    "stuck_midpoint": {
        "category": "unseen_stuck_value",
        "seen_during_training": False,
        "detail": "all cells fixed at 0.5 instead of the trained top-left value",
    },
    "clean_false_missing": {
        "category": "availability_false_negative",
        "seen_during_training": False,
        "detail": "clean observation presented with availability=0",
    },
    "missing_false_available": {
        "category": "availability_false_positive",
        "seen_during_training": False,
        "detail": "zeroed missing observation presented with availability=1",
    },
    "missing_correct": {
        "category": "registered_missing_reference",
        "seen_during_training": True,
        "detail": "zeroed observation with availability=0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate unseen v3 sensor and availability failures"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else resolve_project_path(path)


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


def _sample_seed(sample_id: str, condition: str, seed: int) -> int:
    payload = f"{seed}\0{condition}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        2**63 - 1
    )


def build_derangement(
    sample_ids: Sequence[str],
    *,
    seed: int = CORRUPTION_SEED,
) -> dict[str, str]:
    """Build a deterministic one-cycle derangement with no self assignment."""

    values = [str(value) for value in sample_ids]
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError("derangement requires at least two unique sample IDs")
    indices = list(range(len(values)))
    generator = random.Random(int(seed))
    for index in range(len(indices) - 1, 0, -1):
        other = generator.randrange(index)
        indices[index], indices[other] = indices[other], indices[index]
    mapping = {
        sample_id: values[indices[index]]
        for index, sample_id in enumerate(values)
    }
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("Sattolo derangement unexpectedly produced a fixed point")
    return mapping


def _quality_target(clean: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    change = (observed - clean).abs().mean(dim=(1, 2, 3))
    return (1.0 - change / 0.25).clamp(0.0, 1.0).unsqueeze(1)


def apply_condition(
    condition: str,
    clean: torch.Tensor,
    sample_ids: Sequence[str],
    *,
    sensor_bank: Mapping[str, torch.Tensor],
    derangement: Mapping[str, str],
    seed: int = CORRUPTION_SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return observed grid, reported availability, and true quality."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown sensor condition: {condition}")
    if clean.ndim != 4 or tuple(clean.shape[1:]) != (1, 4, 4):
        raise ValueError("clean sensor batch must have shape [B,1,4,4]")
    if len(sample_ids) != clean.shape[0]:
        raise ValueError("sample IDs must align with the sensor batch")
    clean = clean.float()
    observed = clean.clone()
    availability = torch.ones(clean.shape[0], 1, dtype=clean.dtype)

    if condition == "clean":
        quality = torch.ones_like(availability)
    elif condition == "gamma_2":
        observed = clean.square()
        quality = _quality_target(clean, observed)
    elif condition == "stuck_midpoint":
        observed = torch.full_like(clean, 0.5)
        quality = _quality_target(clean, observed)
    elif condition == "clean_false_missing":
        availability.zero_()
        quality = torch.ones_like(availability)
    elif condition == "missing_false_available":
        observed.zero_()
        quality = torch.zeros_like(availability)
    elif condition == "missing_correct":
        observed.zero_()
        availability.zero_()
        quality = torch.zeros_like(availability)
    elif condition == "foreign_sample":
        rows = []
        for sample_id in sample_ids:
            foreign_id = derangement[str(sample_id)]
            rows.append(sensor_bank[foreign_id])
        observed = torch.stack(rows).to(dtype=clean.dtype)
        quality = _quality_target(clean, observed)
    else:
        rows = []
        for index, sample_id in enumerate(sample_ids):
            grid = clean[index].clone()
            generator = torch.Generator().manual_seed(
                _sample_seed(str(sample_id), condition, seed)
            )
            if condition == "noise_030_out_of_range":
                grid = grid + torch.randn(
                    grid.shape,
                    generator=generator,
                    dtype=grid.dtype,
                ) * 0.30
            elif condition == "bias_030_out_of_range":
                direction = torch.randn(2, generator=generator)
                direction = direction / direction.norm().clamp_min(1e-6)
                sign = -1.0 if torch.rand((), generator=generator) < 0.5 else 1.0
                coords = torch.linspace(-1.0, 1.0, steps=4)
                plane = (
                    direction[0] * coords.view(1, 1, 4)
                    + direction[1] * coords.view(1, 4, 1)
                )
                plane = plane / plane.abs().amax().clamp_min(1e-6)
                grid = grid + sign * 0.30 * plane
            elif condition == "impulse_025":
                mask = torch.rand(
                    grid.shape,
                    generator=generator,
                ) < 0.25
                impulses = (
                    torch.rand(grid.shape, generator=generator) >= 0.5
                ).to(grid.dtype)
                grid = torch.where(mask, impulses, grid)
            elif condition == "spatial_permutation":
                permutation = torch.randperm(16, generator=generator)
                grid = grid.reshape(-1)[permutation].reshape(1, 4, 4)
            else:
                raise AssertionError(f"unhandled condition: {condition}")
            rows.append(grid.clamp(0.0, 1.0))
        observed = torch.stack(rows)
        quality = _quality_target(clean, observed)
    return observed.clamp(0.0, 1.0), availability, quality


def _sensor_bank(dataset: ManifestDataset) -> dict[str, torch.Tensor]:
    sidecar = dataset.sensor_sidecar
    result: dict[str, torch.Tensor] = {}
    for record in dataset.records:
        sample_id = str(record.get("sample_id", record.get("id", "")))
        if not sample_id or sample_id in result:
            raise ValueError("test manifest sample IDs must be unique and non-empty")
        result[sample_id] = load_sensor_grid(
            record,
            manifest_dir=dataset.manifest_dir,
            sidecar=sidecar,
        )
    return result


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("state") != "frozen_before_unseen_condition_evaluation"
        or protocol.get("valid") is not True
        or protocol.get("conditions") != CONDITIONS
        or protocol.get("model_seeds") != list(SEED_RUNS)
        or protocol.get("corruption_seed") != CORRUPTION_SEED
        or protocol.get("bootstrap_repeats") != BOOTSTRAP_REPEATS
    ):
        raise ValueError("unseen-condition protocol identity mismatch")
    drifted = []
    for name, binding in protocol.get("source_bindings", {}).items():
        path = _resolve(Path(binding["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            drifted.append(name)
    if drifted:
        raise ValueError("unseen-condition source drift: " + ", ".join(drifted))


def _evaluate_seed(
    *,
    model_seed: int,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = validated_checkpoint_class_names(checkpoint, config)
    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    data = config["data"]
    dataset = ManifestDataset(
        data["test_manifest"],
        class_names=class_names,
        image_size=int(data["image_size"]),
        sensor_manifest=data["sensor_manifest"],
        preprocess=config.get("preprocess", {}),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    bank = _sensor_bank(dataset)
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
            "reliability": [],
            "availability": [],
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
                    sensor_bank=bank,
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
                    torch.softmax(logits, dim=-1).cpu().numpy()
                )
                collector["sample_ids"].extend(sample_ids)
                collector["source_image_ids"].extend(source_ids)
                collector["quality"].append(quality.numpy().reshape(-1))
                collector["reliability"].append(
                    auxiliary["sensor_reliability"]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                collector["availability"].append(
                    availability.numpy().reshape(-1)
                )
                collector["grid_change"].append(
                    (observed - clean)
                    .abs()
                    .mean(dim=(1, 2, 3))
                    .numpy()
                )

    results: dict[str, Any] = {}
    predictions_by_condition: dict[str, list[dict[str, Any]]] = {}
    common_targets = None
    common_source_ids = None
    for condition, collector in collectors.items():
        targets = np.concatenate(collector["targets"])
        probabilities = np.concatenate(collector["probabilities"])
        quality = np.concatenate(collector["quality"])
        reliability = np.concatenate(collector["reliability"])
        availability = np.concatenate(collector["availability"])
        grid_change = np.concatenate(collector["grid_change"])
        computed = compute_classification_metrics(
            targets,
            probabilities,
            class_names,
            ece_bins=15,
        )
        predicted = computed.pop("predictions")
        results[condition] = {
            "condition": CONDITIONS[condition],
            **computed,
            "sensor_condition": {
                "reported_availability_mean": float(availability.mean()),
                "true_quality_mean": float(quality.mean()),
                "mean_absolute_grid_change": float(grid_change.mean()),
            },
            "sensor_reliability_calibration": reliability_quality_metrics(
                quality,
                reliability,
            ),
        }
        predictions_by_condition[condition] = [
            {
                "condition": condition,
                "sample_id": collector["sample_ids"][index],
                "source_image_id": collector["source_image_ids"][index],
                "target_id": int(targets[index]),
                "predicted_id": int(predicted[index]),
                "correct": bool(predicted[index] == targets[index]),
                "reported_availability": float(availability[index]),
                "true_quality": float(quality[index]),
                "sensor_reliability": float(reliability[index]),
                "probabilities": probabilities[index].tolist(),
            }
            for index in range(targets.size)
        ]
        if common_targets is None:
            common_targets = targets
            common_source_ids = collector["source_image_ids"]
        elif not np.array_equal(common_targets, targets):
            raise RuntimeError("condition target order drift")

    clean_probabilities = np.asarray(
        [row["probabilities"] for row in predictions_by_condition["clean"]],
        dtype=np.float64,
    )
    paired: dict[str, Any] = {}
    assert common_targets is not None and common_source_ids is not None
    for index, condition in enumerate(CONDITIONS):
        if condition == "clean":
            continue
        condition_probabilities = np.asarray(
            [
                row["probabilities"]
                for row in predictions_by_condition[condition]
            ],
            dtype=np.float64,
        )
        paired[condition] = grouped_paired_bootstrap_differences(
            common_targets,
            clean_probabilities,
            condition_probabilities,
            common_source_ids,
            class_names,
            repeats=BOOTSTRAP_REPEATS,
            seed=20261110 + model_seed * 100 + index,
        )

    seed_dir = output_dir / f"seed{model_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = seed_dir / "evaluation.json"
    seed_report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "model_seed": model_seed,
        "sample_count": len(dataset),
        "source_image_count": len(set(common_source_ids)),
        "class_names": list(class_names),
        "conditions": results,
        "condition_minus_clean_paired_bootstrap": paired,
        "derangement": {
            "self_assignment_count": sum(
                source == target for source, target in derangement.items()
            ),
            "sample_count": len(derangement),
            "mapping_sha256": hashlib.sha256(
                json.dumps(
                    derangement,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "source_bindings": {
            "config": _binding(config_path),
            "checkpoint": _binding(checkpoint_path),
            "test_manifest": _binding(resolve_project_path(data["test_manifest"])),
            "sensor_manifest": _binding(
                resolve_project_path(data["sensor_manifest"])
            ),
        },
        "boundaries": [
            (
                "These conditions are deterministic software interventions on "
                "image-conditioned sidecars, not measured hardware failures."
            ),
            (
                "The fixed internal test has been reused during development "
                "and is not an external blind cohort."
            ),
        ],
    }
    _atomic_text(
        evaluation_path,
        json.dumps(seed_report, ensure_ascii=False, indent=2) + "\n",
    )
    prediction_path = seed_dir / "predictions.jsonl"
    _atomic_text(
        prediction_path,
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for condition in CONDITIONS
            for row in predictions_by_condition[condition]
        ),
    )
    del model, checkpoint, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return seed_report


def _summarize(seed_reports: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition in CONDITIONS:
        metrics = {}
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "negative_log_likelihood",
            "multiclass_brier_score",
            "ece",
        ):
            values = [
                float(report["conditions"][condition]["metrics"][metric])
                for report in seed_reports.values()
            ]
            metrics[f"{metric}_mean"] = mean(values)
            metrics[f"{metric}_sample_std"] = stdev(values)
        accuracy_effects = []
        macro_f1_effects = []
        if condition != "clean":
            for report in seed_reports.values():
                differences = report[
                    "condition_minus_clean_paired_bootstrap"
                ][condition]["differences"]
                accuracy_effects.append(
                    float(differences["accuracy"]["point"])
                )
                macro_f1_effects.append(
                    float(differences["macro_f1"]["point"])
                )
        rows.append(
            {
                "condition": condition,
                "category": CONDITIONS[condition]["category"],
                **metrics,
                "condition_minus_clean_accuracy_pp_mean": (
                    100.0 * mean(accuracy_effects)
                    if accuracy_effects
                    else 0.0
                ),
                "condition_minus_clean_macro_f1_pp_mean": (
                    100.0 * mean(macro_f1_effects)
                    if macro_f1_effects
                    else 0.0
                ),
                "reported_availability_mean": mean(
                    float(
                        report["conditions"][condition]["sensor_condition"][
                            "reported_availability_mean"
                        ]
                    )
                    for report in seed_reports.values()
                ),
                "true_quality_mean": mean(
                    float(
                        report["conditions"][condition]["sensor_condition"][
                            "true_quality_mean"
                        ]
                    )
                    for report in seed_reports.values()
                ),
                "predicted_reliability_mean": mean(
                    float(
                        report["conditions"][condition][
                            "sensor_reliability_calibration"
                        ]["prediction_mean"]
                    )
                    for report in seed_reports.values()
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    protocol_path = _resolve(args.protocol)
    protocol = _read_json(protocol_path)
    _validate_protocol(protocol)
    output_path = _resolve(args.output)
    if output_path.exists() or output_path.with_suffix(".md").exists():
        raise FileExistsError("refusing to overwrite unseen-condition outcome")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed_reports = {
        seed: _evaluate_seed(
            model_seed=seed,
            config_path=paths["config"],
            checkpoint_path=paths["checkpoint"],
            device=device,
            output_dir=REPORT_ROOT,
        )
        for seed, paths in SEED_RUNS.items()
    }
    rows = _summarize(seed_reports)
    outcome = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "three_seed_unseen_sensor_and_availability_failures",
        "model_seeds": list(seed_reports),
        "sample_count_per_seed": 826,
        "source_image_count_per_seed": 570,
        "conditions": CONDITIONS,
        "summary": rows,
        "protocol_binding": _binding(protocol_path),
        "seed_evaluation_bindings": {
            str(seed): _binding(REPORT_ROOT / f"seed{seed}" / "evaluation.json")
            for seed in seed_reports
        },
        "claim_boundary": (
            "This extension tests deterministic software failures outside the "
            "registered training corruption range and explicit availability "
            "flag errors on the reused internal benchmark. It is not hardware "
            "fault validation or external generalization evidence."
        ),
        "errors": [],
    }
    _atomic_text(
        output_path,
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
    )
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Unseen sensor and availability condition outcome",
        "",
        "| Condition | Category | Accuracy (%) | Macro F1 (%) | Accuracy vs clean (pp) | Macro F1 vs clean (pp) | Reliability | True quality |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {row['condition']} | {row['category']} | "
                f"{row['accuracy_mean'] * 100:.2f} | "
                f"{row['macro_f1_mean'] * 100:.2f} | "
                f"{row['condition_minus_clean_accuracy_pp_mean']:+.2f} | "
                f"{row['condition_minus_clean_macro_f1_pp_mean']:+.2f} | "
                f"{row['predicted_reliability_mean']:.3f} | "
                f"{row['true_quality_mean']:.3f} |"
            )
            for row in rows
        ],
        "",
        outcome["claim_boundary"],
        "",
    ]
    _atomic_text(output_path.with_suffix(".md"), "\n".join(lines))
    print(
        json.dumps(
            {
                "valid": True,
                "model_seeds": list(seed_reports),
                "condition_count": len(CONDITIONS),
                "output": output_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
