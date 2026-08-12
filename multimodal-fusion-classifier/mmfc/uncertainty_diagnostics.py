"""Numerical diagnostics for heteroscedastic sensor-latent predictions."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import Any

import numpy as np


DEFAULT_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
DEFAULT_BOUND_TOLERANCES = (1e-6, 1e-3, 1e-2, 5e-2, 1e-1)


def _finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _summary(
    values: np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    percentile_values = np.quantile(flat, quantiles)
    return {
        "mean": float(flat.mean()),
        "standard_deviation": float(flat.std()),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "quantiles": {
            f"{quantile:.6g}": float(value)
            for quantile, value in zip(quantiles, percentile_values)
        },
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ordered = flat[order]
    ranks = np.empty(flat.size, dtype=np.float64)
    start = 0
    while start < flat.size:
        stop = start + 1
        while stop < flat.size and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = float(
        np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(centered_x, centered_y) / denominator)


def _correlation_pair(
    predicted_variance: np.ndarray,
    squared_error: np.ndarray,
) -> dict[str, float | None]:
    return {
        "pearson": _correlation(predicted_variance, squared_error),
        "spearman": _correlation(
            _average_ranks(predicted_variance),
            _average_ranks(squared_error),
        ),
    }


def _bound_proximity(
    values: np.ndarray,
    bound: float,
    *,
    lower: bool,
    tolerances: Sequence[float],
) -> dict[str, float]:
    distance = values - bound if lower else bound - values
    return {
        f"within_{tolerance:.6g}": float(np.mean(distance <= tolerance))
        for tolerance in tolerances
    }


def _calibration_bins(
    predicted_variance: np.ndarray,
    squared_error: np.ndarray,
    *,
    bin_count: int,
) -> tuple[list[dict[str, Any]], float]:
    variance = predicted_variance.reshape(-1)
    error = squared_error.reshape(-1)
    order = np.argsort(variance, kind="mergesort")
    bins: list[dict[str, Any]] = []
    absolute_gap_total = 0.0
    for index, indices in enumerate(np.array_split(order, bin_count), start=1):
        if not indices.size:
            continue
        selected_variance = variance[indices]
        selected_error = error[indices]
        mean_variance = float(selected_variance.mean())
        mean_error = float(selected_error.mean())
        absolute_gap_total += abs(mean_error - mean_variance) * indices.size
        bins.append(
            {
                "bin": index,
                "count": int(indices.size),
                "predicted_variance_min": float(selected_variance.min()),
                "predicted_variance_max": float(selected_variance.max()),
                "mean_predicted_variance": mean_variance,
                "mean_squared_error": mean_error,
                "mse_to_variance_ratio": (
                    mean_error / mean_variance if mean_variance > 0.0 else None
                ),
            }
        )
    return bins, absolute_gap_total / max(variance.size, 1)


def compute_uncertainty_diagnostics(
    actual_sensor_latent: np.ndarray,
    predicted_sensor_latent: np.ndarray,
    sensor_log_variance: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    bin_count: int = 10,
    bound_tolerances: Sequence[float] = DEFAULT_BOUND_TOLERANCES,
) -> dict[str, Any]:
    """Compare predicted latent variance with realized encoder-space error."""

    actual = _finite_matrix("actual_sensor_latent", actual_sensor_latent)
    predicted = _finite_matrix(
        "predicted_sensor_latent", predicted_sensor_latent
    )
    log_variance = _finite_matrix("sensor_log_variance", sensor_log_variance)
    if actual.shape != predicted.shape or actual.shape != log_variance.shape:
        raise ValueError(
            "actual, predicted and log-variance arrays must share shape"
        )
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("variance bounds must be finite")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    tolerances = tuple(float(value) for value in bound_tolerances)
    if not tolerances or any(
        not np.isfinite(value) or value <= 0.0 for value in tolerances
    ):
        raise ValueError("bound tolerances must be finite and positive")
    bound_epsilon = 1e-6
    if bool((log_variance < lower_bound - bound_epsilon).any()) or bool(
        (log_variance > upper_bound + bound_epsilon).any()
    ):
        raise ValueError("sensor_log_variance exceeds configured bounds")

    residual = actual - predicted
    squared_error = np.square(residual)
    predicted_variance = np.exp(log_variance)
    standardized_residual = residual / np.sqrt(predicted_variance)
    gaussian_nll = 0.5 * (
        squared_error / predicted_variance + log_variance
    )

    sample_variance = predicted_variance.mean(axis=1)
    sample_mse = squared_error.mean(axis=1)
    dimension_variance = predicted_variance.mean(axis=0)
    dimension_mse = squared_error.mean(axis=0)
    bins, weighted_absolute_gap = _calibration_bins(
        predicted_variance,
        squared_error,
        bin_count=bin_count,
    )
    global_variance = float(predicted_variance.mean())
    global_mse = float(squared_error.mean())
    coverage_one = float(np.mean(np.abs(standardized_residual) <= 1.0))
    coverage_one_ninety_six = float(
        np.mean(np.abs(standardized_residual) <= 1.96)
    )
    lower_distance = float(log_variance.min() - lower_bound)

    return {
        "sample_count": int(actual.shape[0]),
        "latent_dimension": int(actual.shape[1]),
        "element_count": int(actual.size),
        "configured_bounds": {
            "log_variance_min": float(lower_bound),
            "log_variance_max": float(upper_bound),
        },
        "log_variance": _summary(log_variance),
        "predicted_variance": _summary(predicted_variance),
        "squared_error": _summary(squared_error),
        "gaussian_nll": _summary(gaussian_nll),
        "bound_proximity": {
            "lower": _bound_proximity(
                log_variance,
                lower_bound,
                lower=True,
                tolerances=tolerances,
            ),
            "upper": _bound_proximity(
                log_variance,
                upper_bound,
                lower=False,
                tolerances=tolerances,
            ),
        },
        "standardized_residual": {
            **_summary(standardized_residual),
            "coverage_absolute_le_1": coverage_one,
            "coverage_absolute_le_1_96": coverage_one_ninety_six,
            "coverage_absolute_le_3": float(
                np.mean(np.abs(standardized_residual) <= 3.0)
            ),
        },
        "variance_error_correlation": {
            "elementwise": _correlation_pair(
                predicted_variance,
                squared_error,
            ),
            "sample_mean": _correlation_pair(sample_variance, sample_mse),
            "latent_dimension_mean": _correlation_pair(
                dimension_variance,
                dimension_mse,
            ),
        },
        "variance_calibration": {
            "global_mean_predicted_variance": global_variance,
            "global_mean_squared_error": global_mse,
            "global_mse_to_variance_ratio": (
                global_mse / global_variance
                if global_variance > 0.0
                else None
            ),
            "weighted_absolute_variance_error_gap": float(
                weighted_absolute_gap
            ),
            "relative_weighted_absolute_gap": (
                float(weighted_absolute_gap / global_mse)
                if global_mse > 0.0
                else None
            ),
            "equal_count_bins": bins,
        },
        "derived_interpretation_values": {
            "predicted_variance_to_mse_ratio": (
                global_variance / global_mse if global_mse > 0.0 else None
            ),
            "minimum_log_variance_distance_above_lower_bound": lower_distance,
            "coverage_gap_absolute_le_1_vs_standard_normal": (
                coverage_one - 0.6826894921370859
            ),
            "coverage_gap_absolute_le_1_96_vs_standard_normal": (
                coverage_one_ninety_six - 0.950004209703559
            ),
        },
    }


def _sequence_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bootstrap_metric(
    estimate: float,
    replicates: np.ndarray,
    *,
    confidence: float,
) -> dict[str, Any]:
    alpha = 1.0 - confidence
    return {
        "estimate": float(estimate),
        "lower": float(np.quantile(replicates, alpha / 2.0)),
        "upper": float(np.quantile(replicates, 1.0 - alpha / 2.0)),
        "probability_greater_than_zero": float(np.mean(replicates > 0.0)),
        "replicates": replicates.tolist(),
    }


def grouped_bootstrap_uncertainty_intervals(
    sample_mean_predicted_variance: np.ndarray,
    sample_mean_squared_error: np.ndarray,
    source_image_ids: Sequence[str],
    *,
    repeats: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260727,
) -> dict[str, Any]:
    """Cluster-bootstrap sample-level ranking and variance-scale diagnostics."""

    variance = np.asarray(
        sample_mean_predicted_variance,
        dtype=np.float64,
    ).reshape(-1)
    error = np.asarray(
        sample_mean_squared_error,
        dtype=np.float64,
    ).reshape(-1)
    groups = [str(value) for value in source_image_ids]
    if (
        variance.size != error.size
        or variance.size != len(groups)
        or not variance.size
    ):
        raise ValueError(
            "sample variance, error and source_image_ids must share length"
        )
    if not bool(np.isfinite(variance).all()) or not bool(
        np.isfinite(error).all()
    ):
        raise ValueError("bootstrap inputs contain non-finite values")
    if bool((variance <= 0.0).any()) or bool((error < 0.0).any()):
        raise ValueError("bootstrap variance must be positive and error non-negative")
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    unique_groups = sorted(set(groups))
    group_indices = {
        group: np.flatnonzero(np.asarray(groups, dtype=object) == group)
        for group in unique_groups
    }
    generator = np.random.default_rng(int(seed))
    spearman_replicates = np.empty(repeats, dtype=np.float64)
    ratio_replicates = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled_groups = generator.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate(
            [group_indices[str(group)] for group in sampled_groups]
        )
        spearman = _correlation(
            _average_ranks(variance[indices]),
            _average_ranks(error[indices]),
        )
        if spearman is None:
            raise ValueError(
                "bootstrap replicate produced undefined Spearman correlation"
            )
        mean_error = float(error[indices].mean())
        if mean_error <= 0.0:
            raise ValueError(
                "bootstrap replicate produced zero mean squared error"
            )
        spearman_replicates[repeat] = spearman
        ratio_replicates[repeat] = (
            float(variance[indices].mean()) / mean_error
        )

    sample_spearman = _correlation(
        _average_ranks(variance),
        _average_ranks(error),
    )
    if sample_spearman is None:
        raise ValueError("sample-level Spearman correlation is undefined")
    ratio = float(variance.mean() / error.mean())
    return {
        "method": "source_image_cluster_percentile_bootstrap",
        "repeats": int(repeats),
        "confidence": float(confidence),
        "seed": int(seed),
        "sample_count": int(variance.size),
        "cluster_count": len(unique_groups),
        "source_group_assignment_sha256": _sequence_sha256(groups),
        "metrics": {
            "sample_mean_spearman": _bootstrap_metric(
                sample_spearman,
                spearman_replicates,
                confidence=confidence,
            ),
            "predicted_variance_to_mse_ratio": _bootstrap_metric(
                ratio,
                ratio_replicates,
                confidence=confidence,
            ),
        },
    }
