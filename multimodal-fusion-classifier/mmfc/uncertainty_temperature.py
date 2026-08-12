"""Post-hoc scale calibration for latent Gaussian uncertainty."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .uncertainty_diagnostics import compute_uncertainty_diagnostics


def _matrices(
    actual: np.ndarray,
    predicted: np.ndarray,
    log_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (actual, predicted, log_variance)
    )
    if (
        any(value.ndim != 2 or not value.size for value in values)
        or values[0].shape != values[1].shape
        or values[0].shape != values[2].shape
    ):
        raise ValueError(
            "actual, predicted and log_variance must share a non-empty "
            "two-dimensional shape"
        )
    if not all(bool(np.isfinite(value).all()) for value in values):
        raise ValueError("uncertainty calibration inputs must be finite")
    return values


def fit_global_variance_temperature(
    actual: np.ndarray,
    predicted: np.ndarray,
    log_variance: np.ndarray,
    *,
    minimum_scale: float = 1e-6,
    maximum_scale: float = 1e6,
) -> dict[str, float]:
    """Return the closed-form Gaussian-NLL variance scale."""
    actual, predicted, log_variance = _matrices(
        actual, predicted, log_variance
    )
    if (
        not np.isfinite(minimum_scale)
        or not np.isfinite(maximum_scale)
        or minimum_scale <= 0.0
        or maximum_scale < minimum_scale
    ):
        raise ValueError("invalid variance-temperature scale bounds")
    standardized_squared_error = np.square(actual - predicted) * np.exp(
        -log_variance
    )
    unconstrained = float(standardized_squared_error.mean())
    scale = float(np.clip(unconstrained, minimum_scale, maximum_scale))
    return {
        "unconstrained_scale": unconstrained,
        "scale": scale,
        "log_scale": float(np.log(scale)),
        "minimum_scale": float(minimum_scale),
        "maximum_scale": float(maximum_scale),
        "clipped": bool(scale != unconstrained),
    }


def apply_variance_temperature(
    log_variance: np.ndarray,
    scale: float,
) -> np.ndarray:
    values = np.asarray(log_variance, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("log_variance must be a non-empty matrix")
    if not bool(np.isfinite(values).all()):
        raise ValueError("log_variance must be finite")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("variance temperature must be positive and finite")
    return values + np.log(float(scale))


def summarize_uncertainty_scale(
    actual: np.ndarray,
    predicted: np.ndarray,
    log_variance: np.ndarray,
) -> dict[str, Any]:
    actual, predicted, log_variance = _matrices(
        actual, predicted, log_variance
    )
    lower = float(log_variance.min() - 1.0)
    upper = float(log_variance.max() + 1.0)
    diagnostics = compute_uncertainty_diagnostics(
        actual,
        predicted,
        log_variance,
        lower_bound=lower,
        upper_bound=upper,
        bin_count=10,
    )
    residual = actual - predicted
    variance = np.exp(log_variance)
    squared_standardized = np.square(residual) / variance
    per_sample_nll = (
        0.5
        * (squared_standardized + log_variance)
    ).mean(axis=1)
    return {
        "sample_count": int(actual.shape[0]),
        "latent_dimension": int(actual.shape[1]),
        "mean_squared_error": float(np.square(residual).mean()),
        "mean_predicted_variance": float(variance.mean()),
        "mse_to_variance_ratio": float(
            diagnostics["variance_calibration"][
                "global_mse_to_variance_ratio"
            ]
        ),
        "mean_squared_standardized_residual": float(
            squared_standardized.mean()
        ),
        "gaussian_nll_mean": float(per_sample_nll.mean()),
        "coverage_absolute_le_1": float(
            diagnostics["standardized_residual"][
                "coverage_absolute_le_1"
            ]
        ),
        "coverage_absolute_le_1_96": float(
            diagnostics["standardized_residual"][
                "coverage_absolute_le_1_96"
            ]
        ),
        "coverage_absolute_le_3": float(
            diagnostics["standardized_residual"][
                "coverage_absolute_le_3"
            ]
        ),
        "variance_error_correlation": diagnostics[
            "variance_error_correlation"
        ],
        "log_variance": diagnostics["log_variance"],
    }


def per_sample_gaussian_nll(
    actual: np.ndarray,
    predicted: np.ndarray,
    log_variance: np.ndarray,
) -> np.ndarray:
    actual, predicted, log_variance = _matrices(
        actual, predicted, log_variance
    )
    return (
        0.5
        * (
            np.square(actual - predicted) * np.exp(-log_variance)
            + log_variance
        )
    ).mean(axis=1)


def grouped_bootstrap_mean_difference(
    baseline: np.ndarray,
    candidate: np.ndarray,
    group_ids: Sequence[str],
    *,
    repeats: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Cluster-bootstrap candidate minus baseline over source images."""
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    group_ids = [str(value) for value in group_ids]
    if (
        baseline.size != candidate.size
        or baseline.size != len(group_ids)
        or not baseline.size
        or not bool(np.isfinite(baseline).all())
        or not bool(np.isfinite(candidate).all())
    ):
        raise ValueError("bootstrap values and group ids must align")
    if repeats < 2 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(index)
    groups = sorted(grouped)
    differences = candidate - baseline
    group_means = np.asarray(
        [
            float(differences[np.asarray(grouped[group])].mean())
            for group in groups
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(groups), size=(int(repeats), len(groups))
    )
    bootstrap = group_means[sampled].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "metric": "candidate_minus_baseline_gaussian_nll",
        "estimate": float(differences.mean()),
        "group_count": len(groups),
        "sample_count": int(differences.size),
        "repeats": int(repeats),
        "confidence": float(confidence),
        "seed": int(seed),
        "interval": [
            float(np.quantile(bootstrap, alpha / 2.0)),
            float(np.quantile(bootstrap, 1.0 - alpha / 2.0)),
        ],
    }
