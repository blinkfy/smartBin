"""Audit the paper method contract against the frozen implementation and config."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from mmfc.data import write_json
from mmfc.model import (
    SENSOR_CROSS_MODAL_QUALITY_DIM,
    SENSOR_GRID_QUALITY_DIM,
    MultimodalFusionClassifier,
    sensor_grid_quality_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path("configs/ablations/full_cross_modal_quality.json")
DEFAULT_OUTPUT = Path("reports/qwen-material/METHOD_CONTRACT_AUDIT.json")

EXPECTED_CONFIG: tuple[tuple[str, Any], ...] = (
    ("class_count", 19),
    ("model.feature_dim", 512),
    ("model.attention_dim", 128),
    ("model.attention_heads", 8),
    ("model.dropout", 0.3),
    ("model.film_gamma_init", 1.0),
    ("model.film_beta_init", 0.1),
    ("model.scale_prior", [0.3, 0.5, 0.2]),
    ("model.text_temperature", 0.07),
    ("model.sensor_log_variance_min", -3.0),
    ("model.sensor_log_variance_max", 2.0),
    ("model.enabled_modalities", ["visual", "text", "sensor"]),
    ("model.fusion_mode", "attention"),
    ("model.film_enabled", True),
    ("model.sensor_reliability_enabled", True),
    ("model.sensor_grid_quality_enabled", True),
    ("model.sensor_cross_modal_quality_enabled", True),
    ("loss.ce_weight", 1.0),
    ("loss.focal_weight", 1.0),
    ("loss.reconstruction_weight", 0.1),
    ("loss.calibration_weight", 0.3),
    ("loss.variance_prior_weight", 0.1),
    ("loss.variance_prior_target", -2.0),
    ("loss.calibration_clean_weight", 3.0),
    ("loss.sensor_grid_reconstruction_weight", 0.5),
    ("loss.calibration_ranking_weight", 0.05),
    ("loss.calibration_ranking_margin", 0.05),
    ("robustness.sensor_degradation.enabled", True),
    ("robustness.sensor_degradation.missing_probability", 0.1),
    ("robustness.sensor_degradation.noise_probability", 0.5),
    ("robustness.sensor_degradation.noise_std_range", [0.02, 0.15]),
    ("robustness.sensor_degradation.bias_probability", 0.25),
    ("robustness.sensor_degradation.bias_strength", 0.15),
    ("robustness.sensor_degradation.stuck_probability", 0.15),
    ("robustness.sensor_degradation.quality_scale", 0.25),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the documented multimodal method contract"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _source_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _nested_value(document: Mapping[str, Any], dotted_path: str) -> Any:
    if dotted_path == "class_count":
        classes = document.get("classes")
        return len(classes) if isinstance(classes, list) else None
    value: Any = document
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _values_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(observed, (int, float)) and math.isclose(
            float(observed),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            return False
        return all(_values_equal(item, target) for item, target in zip(observed, expected))
    return observed == expected


def _configured_model(config: Mapping[str, Any]) -> MultimodalFusionClassifier:
    model_config = config["model"]
    return MultimodalFusionClassifier(
        num_classes=len(config["classes"]),
        feature_dim=int(model_config["feature_dim"]),
        attention_dim=int(model_config["attention_dim"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        film_gamma_init=float(model_config["film_gamma_init"]),
        film_beta_init=float(model_config["film_beta_init"]),
        scale_prior=tuple(float(value) for value in model_config["scale_prior"]),
        text_temperature=float(model_config["text_temperature"]),
        sensor_log_variance_min=float(
            model_config["sensor_log_variance_min"]
        ),
        sensor_log_variance_max=float(
            model_config["sensor_log_variance_max"]
        ),
        sensor_reliability_hidden_dim=int(
            model_config["sensor_reliability_hidden_dim"]
        ),
        sensor_reliability_init=float(model_config["sensor_reliability_init"]),
        enabled_modalities=tuple(model_config["enabled_modalities"]),
        fusion_mode=str(model_config["fusion_mode"]),
        film_enabled=bool(model_config["film_enabled"]),
        sensor_reliability_enabled=bool(
            model_config["sensor_reliability_enabled"]
        ),
        sensor_grid_quality_enabled=bool(
            model_config["sensor_grid_quality_enabled"]
        ),
        sensor_grid_quality_hidden_dim=int(
            model_config["sensor_grid_quality_hidden_dim"]
        ),
        sensor_cross_modal_quality_enabled=bool(
            model_config["sensor_cross_modal_quality_enabled"]
        ),
        sensor_grid_decoder_hidden_dim=int(
            model_config["sensor_grid_decoder_hidden_dim"]
        ),
        sensor_cross_modal_quality_hidden_dim=int(
            model_config["sensor_cross_modal_quality_hidden_dim"]
        ),
    )


def _query_weights(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    scores = F.normalize(query, dim=-1) @ F.normalize(
        prototypes,
        dim=-1,
    ).transpose(0, 1)
    return torch.softmax(scores / temperature, dim=-1)


def _tensor_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    torch.manual_seed(20260727)
    model = _configured_model(config).eval()
    batch_size = 2
    feature_dim = int(config["model"]["feature_dim"])
    num_classes = len(config["classes"])
    visual_scales = tuple(
        torch.randn(batch_size, feature_dim) for _ in range(3)
    )
    sensor_latent = torch.randn(batch_size, feature_dim)
    prototypes = torch.randn(num_classes, feature_dim)
    observed_grid = torch.rand(batch_size, 1, 4, 4)
    availability = torch.tensor([[1.0], [0.0]])
    raw_quality = sensor_grid_quality_features(observed_grid)

    with torch.no_grad():
        logits, aux = model.forward_from_features(
            visual_scales,
            sensor_latent,
            prototypes,
            sensor_available=availability,
            sensor_quality_features=raw_quality,
            sensor_grid_observation=observed_grid,
            return_aux=True,
        )

    expected_effective = aux["predicted_sensor"] + aux[
        "sensor_reliability"
    ] * (sensor_latent - aux["predicted_sensor"])
    effective_error = float(
        (aux["effective_sensor"] - expected_effective).abs().max()
    )
    visual_query_weights = _query_weights(
        aux["visual_base"],
        prototypes,
        float(config["model"]["text_temperature"]),
    )
    visual_context = visual_query_weights @ prototypes
    visual_context_error = float(
        (aux["visual_text_context"] - visual_context).abs().max()
    )
    fusion_query_weights = _query_weights(
        aux["visual_base"] + aux["effective_sensor"],
        prototypes,
        float(config["model"]["text_temperature"]),
    )
    fusion_weight_error = float(
        (aux["prototype_weights"] - fusion_query_weights).abs().max()
    )
    fusion_text_context = aux["prototype_weights"] @ prototypes
    adapted_tokens = torch.stack(
        (
            model.visual_adapter(aux["visual_base"]),
            model.text_adapter(fusion_text_context),
            model.sensor_adapter(aux["effective_sensor"]),
        ),
        dim=1,
    )
    film_tokens = (
        adapted_tokens * model.film_gamma.unsqueeze(0)
        + model.film_beta.unsqueeze(0)
    )
    normalized_tokens = F.normalize(film_tokens, dim=-1)
    pairwise_cosine = normalized_tokens @ normalized_tokens.transpose(1, 2)
    pairwise_average = (
        pairwise_cosine.sum(dim=-1)
        - pairwise_cosine.diagonal(dim1=1, dim2=2)
    ) / 2.0
    expected_modality_weights = torch.softmax(pairwise_average, dim=-1)
    modality_weight_error = float(
        (aux["modality_weights"] - expected_modality_weights).abs().max()
    )
    missing_fallback_error = float(
        (
            aux["effective_sensor"][1]
            - aux["predicted_sensor"][1]
        )
        .abs()
        .max()
    )
    return {
        "seed": 20260727,
        "logits_shape": list(logits.shape),
        "raw_grid_quality_shape": list(raw_quality.shape),
        "cross_modal_grid_quality_shape": list(
            aux["sensor_cross_modal_quality_features"].shape
        ),
        "predicted_grid_shape": list(aux["predicted_sensor_grid"].shape),
        "sensor_reliability_shape": list(aux["sensor_reliability"].shape),
        "sensor_grid_quality_dim": SENSOR_GRID_QUALITY_DIM,
        "sensor_cross_modal_quality_dim": SENSOR_CROSS_MODAL_QUALITY_DIM,
        "film_gamma_shape": list(model.film_gamma.shape),
        "film_beta_shape": list(model.film_beta.shape),
        "film_affine_parameter_count": int(
            model.film_gamma.numel() + model.film_beta.numel()
        ),
        "scale_prior_is_trainable": bool(
            model.scale_prior_logits.requires_grad
        ),
        "scale_prior_initial": [
            round(float(value), 6)
            for value in torch.softmax(model.scale_prior_logits, dim=0)
        ],
        "effective_sensor_formula_max_abs_error": effective_error,
        "visual_query_context_max_abs_error": visual_context_error,
        "fusion_query_weights_max_abs_error": fusion_weight_error,
        "modality_weight_formula_max_abs_error": modality_weight_error,
        "modality_weight_row_sums": [
            round(float(value), 6)
            for value in aux["modality_weights"].sum(dim=-1)
        ],
        "missing_reliability": float(aux["sensor_reliability"][1, 0]),
        "missing_fallback_max_abs_error": missing_fallback_error,
        "predicted_grid_min": float(aux["predicted_sensor_grid"].min()),
        "predicted_grid_max": float(aux["predicted_sensor_grid"].max()),
    }


def audit_method_contract(
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    resolved_config = _resolve(config_path, root)
    config = json.loads(resolved_config.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    for dotted_path, expected in EXPECTED_CONFIG:
        observed = _nested_value(config, dotted_path)
        passed = _values_equal(observed, expected)
        checks.append(
            {
                "name": f"config.{dotted_path}",
                "passed": passed,
                "observed": observed,
                "expected": expected,
            }
        )
        if not passed:
            errors.append(f"configuration contract mismatch: {dotted_path}")

    try:
        tensor_contract = _tensor_contract(config)
    except Exception as exc:
        tensor_contract = {"error": f"{type(exc).__name__}: {exc}"}
        errors.append("synthetic tensor contract could not be evaluated")
    else:
        tensor_checks = (
            ("logits shape", tensor_contract["logits_shape"] == [2, 19]),
            (
                "raw quality shape",
                tensor_contract["raw_grid_quality_shape"] == [2, 7],
            ),
            (
                "cross-modal quality shape",
                tensor_contract["cross_modal_grid_quality_shape"] == [2, 22],
            ),
            (
                "predicted grid shape",
                tensor_contract["predicted_grid_shape"] == [2, 1, 4, 4],
            ),
            (
                "FiLM-style affine parameter shape",
                tensor_contract["film_gamma_shape"] == [3, 512]
                and tensor_contract["film_beta_shape"] == [3, 512]
                and tensor_contract["film_affine_parameter_count"] == 3072,
            ),
            (
                "trainable scale prior initialization",
                tensor_contract["scale_prior_is_trainable"] is True
                and tensor_contract["scale_prior_initial"] == [0.3, 0.5, 0.2],
            ),
            (
                "effective sensor formula",
                tensor_contract["effective_sensor_formula_max_abs_error"]
                <= 1e-7,
            ),
            (
                "visual-only prototype query",
                tensor_contract["visual_query_context_max_abs_error"] <= 1e-6,
            ),
            (
                "fusion prototype re-query",
                tensor_contract["fusion_query_weights_max_abs_error"] <= 1e-7,
            ),
            (
                "average pairwise-cosine modality weights",
                tensor_contract["modality_weight_formula_max_abs_error"] <= 1e-7
                and tensor_contract["modality_weight_row_sums"] == [1.0, 1.0],
            ),
            (
                "missing reliability is zero",
                abs(tensor_contract["missing_reliability"]) <= 1e-8,
            ),
            (
                "missing falls back to predicted latent",
                tensor_contract["missing_fallback_max_abs_error"] <= 1e-7,
            ),
            (
                "predicted clean grid is bounded",
                0.0 <= tensor_contract["predicted_grid_min"]
                <= tensor_contract["predicted_grid_max"]
                <= 1.0,
            ),
        )
        for name, passed in tensor_checks:
            checks.append(
                {
                    "name": f"tensor.{name}",
                    "passed": bool(passed),
                }
            )
            if not passed:
                errors.append(f"tensor contract mismatch: {name}")

    source_paths = (
        root / "audit_method_contract.py",
        root / "mmfc" / "model.py",
        root / "mmfc" / "losses.py",
        root / "mmfc" / "training.py",
        resolved_config,
    )
    sources = [_source_record(path, root) for path in source_paths]
    fingerprint_payload = json.dumps(
        sources,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "failed",
        "valid": not errors,
        "config": _display_path(resolved_config, root),
        "pytorch_version": torch.__version__,
        "source_fingerprint_sha256": hashlib.sha256(
            fingerprint_payload
        ).hexdigest(),
        "sources": sources,
        "checks": checks,
        "tensor_contract": tensor_contract,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config, PROJECT_ROOT)
    output_path = _resolve(args.output, PROJECT_ROOT)
    report = audit_method_contract(PROJECT_ROOT, config_path)
    write_json(output_path, report)
    print(json.dumps({"output": str(output_path), **report}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
