"""Fit a validation-only global variance temperature for the v3 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from evaluate_uncertainty import (
    _collect_latents,
    _source_binding,
    _training_config_fingerprint,
)
from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.model import build_production_model
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)
from mmfc.uncertainty_temperature import (
    apply_variance_temperature,
    fit_global_variance_temperature,
    summarize_uncertainty_scale,
)


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "full_cross_modal_quality"
    / "uncertainty-temperature"
)
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "ablations"
    / "full_cross_modal_quality.json"
)
CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "ablations"
    / "full_cross_modal_quality"
    / "best.pt"
)
PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_OUTPUT = REPORT_ROOT / "PARAMETERS.json"
PROTOCOL_NAME = "v3_latent_global_variance_temperature_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit v3 latent variance scale on validation only"
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_project_path(path)


def _read_json(path: Path) -> dict[str, Any]:
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


def collect_split_latents(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid latent collection runtime")
    config_path = _resolve(config_path)
    checkpoint_path = _resolve(checkpoint_path)
    config = load_config(config_path)
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = validated_checkpoint_class_names(checkpoint, config)
    fingerprint = _training_config_fingerprint(config)
    if checkpoint.get("config_fingerprint") != fingerprint:
        raise ValueError("checkpoint and uncertainty config drifted")

    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])
    if bool(getattr(model, "legacy_sensor_passthrough", False)):
        raise ValueError("legacy checkpoint has no uncertainty head")
    if not bool(getattr(model, "sensor_reliability_enabled", False)):
        raise ValueError("sensor reliability heads are disabled")

    data = config["data"]
    manifest_path = _resolve(Path(data[f"{split}_manifest"]))
    sensor_manifest = _resolve(Path(data["sensor_manifest"]))
    dataset = ManifestDataset(
        manifest_path,
        class_names=class_names,
        image_size=int(data.get("image_size", 640)),
        sensor_manifest=sensor_manifest,
        preprocess=config.get("preprocess", {}),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    actual, predicted, log_variance, source_image_ids = _collect_latents(
        model,
        loader,
        device=device,
    )
    return {
        "actual": actual,
        "predicted": predicted,
        "log_variance": log_variance,
        "source_image_ids": source_image_ids,
        "sample_count": len(dataset),
        "source_image_count": len(set(source_image_ids)),
        "checkpoint": checkpoint,
        "config_fingerprint": fingerprint,
        "bindings": {
            "config": _source_binding(config_path),
            "checkpoint": _source_binding(checkpoint_path),
            "manifest": _source_binding(manifest_path),
            "sensor_manifest": _source_binding(sensor_manifest),
        },
    }


def fit(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = _resolve(args.protocol)
    output = _resolve(args.output)
    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    if output.is_file():
        raise FileExistsError(
            "refusing to overwrite frozen uncertainty parameters"
        )
    for path in (protocol_path, config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"calibration input is missing: {path}")
    protocol = _read_json(protocol_path)
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or protocol.get("status")
        != "development_protocol_frozen_before_calibration"
        or protocol.get("formal_paper_metric") is not False
        or protocol.get("fit_split") != "val"
        or protocol.get("test_split") != "test"
        or protocol.get("bindings", {})
        .get("config", {})
        .get("sha256")
        != _source_binding(config_path)["sha256"]
        or protocol.get("bindings", {})
        .get("checkpoint", {})
        .get("sha256")
        != _source_binding(checkpoint_path)["sha256"]
    ):
        raise ValueError("uncertainty-temperature protocol drifted")

    collected = collect_split_latents(
        config_path,
        checkpoint_path,
        "val",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device_name=str(args.device),
    )
    parameters = fit_global_variance_temperature(
        collected["actual"],
        collected["predicted"],
        collected["log_variance"],
        minimum_scale=float(protocol["scale_bounds"][0]),
        maximum_scale=float(protocol["scale_bounds"][1]),
    )
    raw = summarize_uncertainty_scale(
        collected["actual"],
        collected["predicted"],
        collected["log_variance"],
    )
    calibrated_log_variance = apply_variance_temperature(
        collected["log_variance"], parameters["scale"]
    )
    calibrated = summarize_uncertainty_scale(
        collected["actual"],
        collected["predicted"],
        calibrated_log_variance,
    )
    acceptance = {
        "scale_not_clipped": parameters["clipped"] is False,
        "calibrated_nll_below_raw": (
            calibrated["gaussian_nll_mean"]
            < raw["gaussian_nll_mean"]
        ),
        "standardized_residual_mse_near_one": (
            abs(
                calibrated["mean_squared_standardized_residual"] - 1.0
            )
            <= 1e-10
        ),
        "variance_error_rank_preserved": (
            calibrated["variance_error_correlation"]["elementwise"][
                "spearman"
            ]
            == raw["variance_error_correlation"]["elementwise"]["spearman"]
        ),
    }
    report = {
        "schema_version": 1,
        "state": "frozen_before_test_evaluation",
        "valid": all(acceptance.values()),
        "formal_paper_metric": False,
        "protocol": PROTOCOL_NAME,
        "fit_split": "val",
        "test_manifest_read": False,
        "parameters": parameters,
        "validation": {
            "sample_count": collected["sample_count"],
            "source_image_count": collected["source_image_count"],
            "raw": raw,
            "calibrated": calibrated,
        },
        "acceptance": acceptance,
        "runtime": {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(args.device),
        },
        "bindings": {
            "protocol": _source_binding(protocol_path),
            **collected["bindings"],
            "fitter_source": _source_binding(Path(__file__)),
            "numerical_source": _source_binding(
                PROJECT_ROOT / "mmfc" / "uncertainty_temperature.py"
            ),
        },
        "claim_boundary": (
            "This validation-only scale calibrates image-conditioned latent "
            "Gaussian variance; it does not alter predictions, the trained "
            "reliability gate, or establish physical sensor calibration."
        ),
    }
    if not report["valid"]:
        raise ValueError("validation uncertainty calibration was not accepted")
    _atomic_json(output, report)
    return report


def main() -> None:
    result = fit(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "scale": result["parameters"]["scale"],
                "validation_raw_nll": result["validation"]["raw"][
                    "gaussian_nll_mean"
                ],
                "validation_calibrated_nll": result["validation"][
                    "calibrated"
                ]["gaussian_nll_mean"],
                "test_manifest_read": result["test_manifest_read"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
