"""Run the preregistered one-time external object-classification analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate import _runtime_metadata
from evaluate_uncertainty import _source_binding
from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.evaluation import (
    evaluate_model,
    grouped_bootstrap_classification_intervals,
    grouped_paired_bootstrap_differences,
    write_evaluation_artifacts,
)
from mmfc.external_blind import validate_external_blind_bundle
from mmfc.model import build_production_model
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)
from validate_external_blind_intake import (
    DEFAULT_BUNDLE,
    DEFAULT_DHASH,
    DEFAULT_INTEGRITY,
    DEFAULT_PROTOCOL,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "ablations"
    / "full_cross_modal_quality.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "ablations"
    / "full_cross_modal_quality"
    / "best.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "external-blind-v1"
    / "evaluation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen external blind confirmation once"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_project_path(path)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _arrays(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records = result["prediction_records"]
    return (
        np.asarray([record["target_id"] for record in records], dtype=np.int64),
        np.asarray(
            [record["probabilities"] for record in records],
            dtype=np.float64,
        ),
        [str(record["source_image_id"]) for record in records],
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _resolve(args.bundle)
    protocol_path = _resolve(args.protocol)
    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    output = _resolve(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "refusing to repeat or overwrite the external blind evaluation"
        )
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid external evaluation runtime")
    protocol = _read(protocol_path)
    intake = validate_external_blind_bundle(
        bundle,
        protocol_path=protocol_path,
        development_integrity_path=DEFAULT_INTEGRITY,
        development_dhash_index_path=DEFAULT_DHASH,
    )
    if not intake["valid"]:
        raise ValueError(
            "external blind intake failed: " + "; ".join(intake["errors"][:5])
        )
    if (
        protocol.get("protocol") != "external_blind_confirmation_v1"
        or protocol.get("status") != "preregistered_awaiting_external_data"
        or protocol.get("checkpoint", {}).get("sha256")
        != _source_binding(checkpoint_path)["sha256"]
        or protocol.get("bindings", {})
        .get("config", {})
        .get("sha256")
        != _source_binding(config_path)["sha256"]
    ):
        raise ValueError("external blind runtime drifted from protocol")
    registered_runtime = protocol["evaluation"]
    if (
        int(args.batch_size) != int(registered_runtime["batch_size"])
        or int(args.num_workers) != int(registered_runtime["num_workers"])
        or str(args.device) != str(registered_runtime["device"])
    ):
        raise ValueError("external blind batch/workers/device drifted")
    runtime_bindings = (
        ("development_integrity", DEFAULT_INTEGRITY),
        ("development_dhash_index", DEFAULT_DHASH),
        ("contract_source", PROJECT_ROOT / "mmfc" / "external_blind.py"),
        ("validator_source", PROJECT_ROOT / "validate_external_blind_intake.py"),
        ("evaluator_source", Path(__file__)),
        ("evaluation_core", PROJECT_ROOT / "mmfc" / "evaluation.py"),
        ("dataset_core", PROJECT_ROOT / "mmfc" / "training.py"),
        ("model_core", PROJECT_ROOT / "mmfc" / "model.py"),
        ("checkpoint_core", PROJECT_ROOT / "mmfc" / "checkpoints.py"),
    )
    for name, path in runtime_bindings:
        if (
            protocol.get("bindings", {}).get(name, {}).get("sha256")
            != _source_binding(path)["sha256"]
        ):
            raise ValueError(f"external blind bound source drifted: {name}")

    config = load_config(config_path)
    device = torch.device(str(args.device))
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = validated_checkpoint_class_names(checkpoint, config)
    if list(class_names) != protocol["class_names"]:
        raise ValueError("checkpoint class order drifted from protocol")
    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])

    dataset = ManifestDataset(
        bundle / "manifest.jsonl",
        class_names=class_names,
        image_size=int(config["data"].get("image_size", 480)),
        sensor_manifest=bundle / "sensors.jsonl",
        preprocess=config.get("preprocess", {}),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    clean = evaluate_model(
        model,
        loader,
        device=device,
        class_names=class_names,
        ece_bins=int(protocol["evaluation"]["ece_bins"]),
        warmup_batches=int(protocol["evaluation"]["warmup_batches"]),
    )
    missing_config = {
        "enabled": True,
        "missing_probability": 1.0,
        "noise_probability": 0.0,
        "noise_std": 0.0,
        "bias_probability": 0.0,
        "bias_strength": 0.0,
        "stuck_probability": 0.0,
        "quality_scale": 0.25,
    }
    fallback = evaluate_model(
        model,
        loader,
        device=device,
        class_names=class_names,
        ece_bins=int(protocol["evaluation"]["ece_bins"]),
        warmup_batches=int(protocol["evaluation"]["warmup_batches"]),
        sensor_degradation=missing_config,
        degradation_seed=int(protocol["evaluation"]["seed"]),
    )
    targets, clean_probabilities, source_ids = _arrays(clean)
    fallback_targets, fallback_probabilities, fallback_source_ids = _arrays(
        fallback
    )
    if (
        not np.array_equal(targets, fallback_targets)
        or source_ids != fallback_source_ids
    ):
        raise ValueError("clean and fallback predictions do not align")
    repeats = int(protocol["evaluation"]["bootstrap_repeats"])
    confidence = float(protocol["evaluation"]["bootstrap_confidence"])
    seed = int(protocol["evaluation"]["bootstrap_seed"])
    clean["confidence_intervals"] = (
        grouped_bootstrap_classification_intervals(
            targets,
            clean_probabilities,
            source_ids,
            class_names,
            repeats=repeats,
            confidence=confidence,
            seed=seed,
        )
    )
    fallback["confidence_intervals"] = (
        grouped_bootstrap_classification_intervals(
            targets,
            fallback_probabilities,
            source_ids,
            class_names,
            repeats=repeats,
            confidence=confidence,
            seed=seed,
        )
    )
    paired = grouped_paired_bootstrap_differences(
        targets,
        fallback_probabilities,
        clean_probabilities,
        source_ids,
        class_names,
        repeats=repeats,
        confidence=confidence,
        seed=seed,
    )

    output.mkdir(parents=True, exist_ok=True)
    clean_artifacts = write_evaluation_artifacts(
        clean,
        output / "clean",
        class_names=class_names,
        split="external_blind_clean",
    )
    fallback_artifacts = write_evaluation_artifacts(
        fallback,
        output / "rgb_fallback",
        class_names=class_names,
        split="external_blind_rgb_fallback",
    )
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_external_blind_metric": True,
        "protocol": protocol["protocol"],
        "dataset_id": intake["dataset_id"],
        "sample_count": intake["sample_count"],
        "source_image_count": intake["source_image_count"],
        "intake": intake,
        "clean": {
            "metrics": clean["metrics"],
            "performance": clean["performance"],
            "confidence_intervals": clean["confidence_intervals"],
            "artifacts": clean_artifacts,
        },
        "rgb_fallback": {
            "metrics": fallback["metrics"],
            "performance": fallback["performance"],
            "confidence_intervals": fallback["confidence_intervals"],
            "artifacts": fallback_artifacts,
        },
        "paired_clean_minus_rgb_fallback": paired,
        "runtime": _runtime_metadata(device),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "protocol": _source_binding(protocol_path),
            "config": _source_binding(config_path),
            "checkpoint": _source_binding(checkpoint_path),
            "release": _source_binding(bundle / "RELEASE.json"),
            "manifest": _source_binding(bundle / "manifest.jsonl"),
            "sensors": _source_binding(bundle / "sensors.jsonl"),
            "evaluator_source": _source_binding(Path(__file__)),
        },
        "claim_boundary": (
            "This is the one-time external confirmation for the frozen "
            "19-class object model. Sensor-source and acquisition-site "
            "generalization must be interpreted according to RELEASE.json."
        ),
    }
    _atomic_json(output / "EVALUATION.json", report)
    return report


def main() -> None:
    result = evaluate(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_external_blind_metric": result[
                    "formal_external_blind_metric"
                ],
                "dataset_id": result["dataset_id"],
                "sample_count": result["sample_count"],
                "source_image_count": result["source_image_count"],
                "clean_metrics": result["clean"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
