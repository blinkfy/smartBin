"""Independent split evaluation and paper-ready artifact generation."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .training import degrade_sensor_grid


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def reliability_quality_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    pairwise_margin: float = 0.05,
) -> dict[str, float | int | None]:
    """Measure continuous calibration and sample-pair quality ordering."""

    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.size == 0 or targets.shape != predictions.shape:
        raise ValueError("reliability targets/predictions must be non-empty and aligned")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("reliability targets/predictions must be finite")
    if pairwise_margin < 0.0:
        raise ValueError("pairwise_margin must be non-negative")

    error = predictions - targets
    pearson = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if np.std(predictions) > 1e-12 and np.std(targets) > 1e-12
        else None
    )
    prediction_ranks = _average_ranks(predictions)
    target_ranks = _average_ranks(targets)
    spearman = (
        float(np.corrcoef(prediction_ranks, target_ranks)[0, 1])
        if np.std(prediction_ranks) > 1e-12
        and np.std(target_ranks) > 1e-12
        else None
    )
    target_difference = targets[:, None] - targets[None, :]
    prediction_difference = predictions[:, None] - predictions[None, :]
    pair_mask = np.triu(
        np.ones_like(target_difference, dtype=bool),
        k=1,
    ) & (np.abs(target_difference) >= pairwise_margin)
    pair_count = int(pair_mask.sum())
    if pair_count:
        pair_prediction_difference = prediction_difference[pair_mask]
        signed_product = target_difference[pair_mask] * pair_prediction_difference
        pairwise_accuracy = float(
            (
                np.count_nonzero(signed_product > 0.0)
                + 0.5 * np.count_nonzero(pair_prediction_difference == 0.0)
            )
            / pair_count
        )
    else:
        pairwise_accuracy = None
    return {
        "target_mean": float(targets.mean()),
        "prediction_mean": float(predictions.mean()),
        "mean_error": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "pairwise_ordering_accuracy": pairwise_accuracy,
        "pairwise_margin": float(pairwise_margin),
        "pair_count": pair_count,
    }


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    """Compute top-label ECE with equal-width confidence bins."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.size:
        raise ValueError("probabilities 必须为 [N,C]，且与 targets 样本数一致")
    if probabilities.shape[1] < 2:
        raise ValueError("ECE 至少需要两个类别")
    if bins < 2:
        raise ValueError("ECE bins 必须大于等于 2")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities 包含 NaN 或无穷值")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("targets 包含越界类别索引")

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    bin_ids = np.digitize(
        confidences,
        np.linspace(0.0, 1.0, bins + 1)[1:-1],
        right=True,
    )
    ece = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidences[mask].mean())
        )
    return ece


def compute_classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    *,
    ece_bins: int = 15,
) -> dict[str, Any]:
    """Compute aggregate, per-class, calibration, and confusion metrics."""

    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_names = tuple(class_names)
    if probabilities.shape != (targets.size, len(class_names)):
        raise ValueError(
            "probabilities 形状必须为 [样本数, 类别数]，"
            f"实际为 {list(probabilities.shape)}"
        )
    if targets.size == 0:
        raise ValueError("评估样本不能为空")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities 包含 NaN 或无穷值")
    row_sums = probabilities.sum(axis=1)
    if np.any(probabilities < 0.0) or not np.allclose(
        row_sums, 1.0, atol=1e-5, rtol=1e-5
    ):
        raise ValueError("probabilities 必须是每行和为 1 的非负概率")
    if np.any(targets < 0) or np.any(targets >= len(class_names)):
        raise ValueError("targets 包含越界类别索引")

    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    true_probabilities = probabilities[np.arange(targets.size), targets]
    one_hot = np.eye(len(class_names), dtype=np.float64)[targets]
    top_k = min(3, len(class_names))
    top_k_predictions = np.argpartition(
        probabilities, kth=-top_k, axis=1
    )[:, -top_k:]

    per_class = [
        {
            "label_id": index,
            "label": name,
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, name in enumerate(class_names)
    ]
    return {
        "metrics": {
            "sample_count": int(targets.size),
            "accuracy": float((predictions == targets).mean()),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "top3_accuracy": float(
                np.any(top_k_predictions == targets[:, None], axis=1).mean()
            ),
            "negative_log_likelihood": float(
                -np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean()
            ),
            "multiclass_brier_score": float(
                np.square(probabilities - one_hot).sum(axis=1).mean()
            ),
            "ece": expected_calibration_error(
                probabilities, targets, bins=ece_bins
            ),
            "ece_bins": int(ece_bins),
        },
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions,
    }


_BOOTSTRAP_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "negative_log_likelihood",
    "multiclass_brier_score",
    "ece",
)
_LOWER_IS_BETTER = {
    "negative_log_likelihood",
    "multiclass_brier_score",
    "ece",
}


def _classification_summary_from_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    confusion = np.bincount(
        targets * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_positive = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1, dtype=np.float64)
    predicted = confusion.sum(axis=0, dtype=np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / max(1.0, actual.sum())),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(
            (f1 * actual).sum() / max(1.0, actual.sum())
        ),
    }


def _probability_summary(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    true_probabilities = probabilities[
        np.arange(targets.size),
        targets,
    ]
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    return {
        "negative_log_likelihood": float(
            -np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean()
        ),
        "multiclass_brier_score": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "ece": expected_calibration_error(
            probabilities,
            targets,
            bins=15,
        ),
    }


def _bootstrap_inputs(
    targets: np.ndarray,
    probabilities: np.ndarray,
    source_image_ids: Sequence[str],
    class_names: Sequence[str],
    repeats: int,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    source_ids = np.asarray([str(value) for value in source_image_ids])
    num_classes = len(tuple(class_names))
    if probabilities.shape != (targets.size, num_classes):
        raise ValueError("bootstrap probabilities must have shape [N,C]")
    if source_ids.shape != (targets.size,):
        raise ValueError("bootstrap source_image_ids must align with targets")
    if repeats < 2:
        raise ValueError("bootstrap repeats must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be in (0,1)")
    if targets.size == 0:
        raise ValueError("bootstrap inputs must not be empty")
    grouped: dict[str, list[int]] = {}
    for index, source_id in enumerate(source_ids):
        grouped.setdefault(str(source_id), []).append(index)
    clusters = [
        np.asarray(indices, dtype=np.int64)
        for indices in grouped.values()
    ]
    return targets, probabilities, clusters, num_classes


def grouped_bootstrap_classification_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    source_image_ids: Sequence[str],
    class_names: Sequence[str],
    *,
    repeats: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260727,
) -> dict[str, Any]:
    """Percentile intervals from source-image cluster resampling."""

    targets, probabilities, clusters, num_classes = _bootstrap_inputs(
        targets,
        probabilities,
        source_image_ids,
        class_names,
        repeats,
        confidence,
    )
    predictions = probabilities.argmax(axis=1)
    generator = np.random.default_rng(int(seed))
    samples = {
        name: np.empty(repeats, dtype=np.float64)
        for name in _BOOTSTRAP_METRICS
    }
    for repeat in range(repeats):
        selected = generator.integers(
            0,
            len(clusters),
            size=len(clusters),
        )
        indices = np.concatenate([clusters[index] for index in selected])
        summary = _classification_summary_from_predictions(
            targets[indices],
            predictions[indices],
            num_classes,
        )
        summary.update(
            _probability_summary(
                targets[indices],
                probabilities[indices],
            )
        )
        for name in _BOOTSTRAP_METRICS:
            samples[name][repeat] = summary[name]

    alpha = 0.5 * (1.0 - confidence)
    return {
        "method": "source_image_cluster_percentile_bootstrap",
        "repeats": int(repeats),
        "confidence": float(confidence),
        "seed": int(seed),
        "cluster_count": len(clusters),
        "sample_count": int(targets.size),
        "intervals": {
            name: {
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
            }
            for name, values in samples.items()
        },
    }


def grouped_paired_bootstrap_differences(
    targets: np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    source_image_ids: Sequence[str],
    class_names: Sequence[str],
    *,
    repeats: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260727,
) -> dict[str, Any]:
    """Paired candidate-minus-reference intervals on source-image clusters."""

    targets, reference, clusters, num_classes = _bootstrap_inputs(
        targets,
        reference_probabilities,
        source_image_ids,
        class_names,
        repeats,
        confidence,
    )
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(
            "candidate probabilities must match reference probabilities"
        )
    reference_predictions = reference.argmax(axis=1)
    candidate_predictions = candidate.argmax(axis=1)
    reference_point = _classification_summary_from_predictions(
        targets,
        reference_predictions,
        num_classes,
    )
    reference_point.update(_probability_summary(targets, reference))
    candidate_point = _classification_summary_from_predictions(
        targets,
        candidate_predictions,
        num_classes,
    )
    candidate_point.update(_probability_summary(targets, candidate))
    differences = {
        name: np.empty(repeats, dtype=np.float64)
        for name in _BOOTSTRAP_METRICS
    }
    generator = np.random.default_rng(int(seed))
    for repeat in range(repeats):
        selected = generator.integers(
            0,
            len(clusters),
            size=len(clusters),
        )
        indices = np.concatenate([clusters[index] for index in selected])
        reference_summary = _classification_summary_from_predictions(
            targets[indices],
            reference_predictions[indices],
            num_classes,
        )
        reference_summary.update(
            _probability_summary(targets[indices], reference[indices])
        )
        candidate_summary = _classification_summary_from_predictions(
            targets[indices],
            candidate_predictions[indices],
            num_classes,
        )
        candidate_summary.update(
            _probability_summary(targets[indices], candidate[indices])
        )
        for name in _BOOTSTRAP_METRICS:
            differences[name][repeat] = (
                candidate_summary[name] - reference_summary[name]
            )

    alpha = 0.5 * (1.0 - confidence)
    return {
        "method": "paired_source_image_cluster_percentile_bootstrap",
        "repeats": int(repeats),
        "confidence": float(confidence),
        "seed": int(seed),
        "cluster_count": len(clusters),
        "sample_count": int(targets.size),
        "differences": {
            name: {
                "point": candidate_point[name] - reference_point[name],
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
                "probability_greater_than_zero": float(
                    np.mean(values > 0.0)
                ),
                "preferred_direction": (
                    "lower" if name in _LOWER_IS_BETTER else "higher"
                ),
                "probability_candidate_better": float(
                    np.mean(values < 0.0)
                    if name in _LOWER_IS_BETTER
                    else np.mean(values > 0.0)
                ),
            }
            for name, values in differences.items()
        },
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_auxiliary(
    auxiliary_totals: dict[str, np.ndarray | float],
    sample_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, total in auxiliary_totals.items():
        value = np.asarray(total, dtype=np.float64) / max(sample_count, 1)
        result[name] = float(value) if value.ndim == 0 else value.tolist()
    return result


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    class_names: Sequence[str],
    ece_bins: int = 15,
    warmup_batches: int = 3,
    sensor_degradation: dict[str, Any] | None = None,
    degradation_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a model with an optional deterministic sensor condition.

    Forward timing excludes image loading and preprocessing. The first batch is
    reused for warmup and then evaluated normally.
    """

    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sample_ids: list[str] = []
    source_image_ids: list[str] = []
    forward_times: list[float] = []
    auxiliary_totals: dict[str, np.ndarray | float] = {}
    sample_auxiliaries: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sensor_reliability",
            "sensor_consistency",
            "sensor_discrepancy",
            "sensor_uncertainty",
            "sensor_available",
            "sensor_reliability_features",
            "sensor_grid_quality_features",
            "sensor_cross_modal_quality_features",
        )
    }
    quality_targets: list[np.ndarray] = []
    condition_totals = {
        "availability": 0.0,
        "mean_absolute_change": 0.0,
        "quality_target": 0.0,
    }
    reconstruction_totals = {
        "latent_squared_error": 0.0,
        "latent_absolute_error": 0.0,
        "latent_elements": 0,
        "grid_squared_error": 0.0,
        "grid_absolute_error": 0.0,
        "grid_elements": 0,
    }
    latent_predictions: list[np.ndarray] = []
    latent_targets: list[np.ndarray] = []
    grid_predictions: list[np.ndarray] = []
    grid_targets: list[np.ndarray] = []
    degradation_generator = torch.Generator().manual_seed(int(degradation_seed))
    warmed_up = False

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            clean_sensors = batch["sensor"]
            labels = batch["label"].to(device, non_blocking=True)
            if sensor_degradation is not None:
                degradation = degrade_sensor_grid(
                    clean_sensors,
                    sensor_degradation,
                    generator=degradation_generator,
                )
                evaluated_sensors = degradation["degraded_grid"]
                availability_cpu = degradation["availability"]
                quality_target = degradation["quality_target"]
            else:
                evaluated_sensors = clean_sensors
                availability_cpu = torch.ones(
                    labels.shape[0], 1, dtype=clean_sensors.dtype
                )
                quality_target = availability_cpu.clone()
            sensors = evaluated_sensors.to(device, non_blocking=True)
            availability = availability_cpu.to(device, non_blocking=True)
            condition_totals["availability"] += float(availability_cpu.sum())
            condition_totals["mean_absolute_change"] += float(
                (evaluated_sensors - clean_sensors)
                .abs()
                .mean(dim=(1, 2, 3))
                .sum()
            )
            condition_totals["quality_target"] += float(quality_target.sum())
            quality_targets.append(
                quality_target.detach().cpu().numpy().reshape(-1)
            )
            if not warmed_up:
                for _ in range(max(0, int(warmup_batches))):
                    model(
                        images,
                        sensors,
                        sensor_available=availability,
                        return_aux=True,
                    )
                _synchronize(device)
                warmed_up = True

            _synchronize(device)
            started = time.perf_counter()
            logits, aux = model(
                images,
                sensors,
                sensor_available=availability,
                return_aux=True,
            )
            _synchronize(device)
            forward_times.append(time.perf_counter() - started)

            sensor_encoder = getattr(model, "sensor_encoder", None)
            if sensor_encoder is not None and "predicted_sensor" in aux:
                clean_sensor_device = clean_sensors.to(
                    device,
                    non_blocking=True,
                )
                clean_sensor_target = sensor_encoder(clean_sensor_device)
                latent_difference = (
                    aux["predicted_sensor"] - clean_sensor_target
                )
                reconstruction_totals["latent_squared_error"] += float(
                    latent_difference.square().sum().item()
                )
                reconstruction_totals["latent_absolute_error"] += float(
                    latent_difference.abs().sum().item()
                )
                reconstruction_totals["latent_elements"] += int(
                    latent_difference.numel()
                )
                latent_predictions.append(
                    aux["predicted_sensor"].detach().cpu().numpy()
                )
                latent_targets.append(
                    clean_sensor_target.detach().cpu().numpy()
                )
            if (
                bool(getattr(model, "sensor_cross_modal_quality_enabled", False))
                and "predicted_sensor_grid" in aux
            ):
                clean_grid_target = clean_sensors.to(
                    device,
                    dtype=aux["predicted_sensor_grid"].dtype,
                    non_blocking=True,
                )
                grid_difference = (
                    aux["predicted_sensor_grid"] - clean_grid_target
                )
                reconstruction_totals["grid_squared_error"] += float(
                    grid_difference.square().sum().item()
                )
                reconstruction_totals["grid_absolute_error"] += float(
                    grid_difference.abs().sum().item()
                )
                reconstruction_totals["grid_elements"] += int(
                    grid_difference.numel()
                )
                grid_predictions.append(
                    aux["predicted_sensor_grid"].detach().cpu().numpy()
                )
                grid_targets.append(clean_grid_target.detach().cpu().numpy())

            batch_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            probabilities.append(batch_probabilities)
            targets.append(labels.cpu().numpy())
            batch_sample_ids = [str(value) for value in batch["sample_id"]]
            sample_ids.extend(batch_sample_ids)
            source_image_ids.extend(
                str(value)
                for value in batch.get(
                    "source_image_id",
                    batch_sample_ids,
                )
            )
            batch_size = labels.numel()
            for name in (
                "sensor_reliability",
                "sensor_consistency",
                "sensor_discrepancy",
                "sensor_uncertainty",
                "sensor_available",
                "sensor_grid_quality_features",
                "sensor_cross_modal_quality_features",
                "sensor_reliability_features",
                "modality_weights",
                "scale_weights",
            ):
                if name not in aux:
                    continue
                detached = aux[name].detach().cpu().numpy()
                total = detached.sum(axis=0)
                auxiliary_totals[name] = auxiliary_totals.get(name, 0.0) + total
                if name in sample_auxiliaries:
                    values = detached.reshape(batch_size, -1)
                    sample_auxiliaries[name].append(
                        values[:, 0] if values.shape[1] == 1 else values
                    )

    target_array = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    probability_array = (
        np.concatenate(probabilities)
        if probabilities
        else np.empty((0, len(class_names)), dtype=np.float64)
    )
    computed = compute_classification_metrics(
        target_array,
        probability_array,
        class_names,
        ece_bins=ece_bins,
    )
    prediction_ids = computed.pop("predictions")
    total_forward_seconds = sum(forward_times)
    sample_count = int(target_array.size)
    quality_array = (
        np.concatenate(quality_targets)
        if quality_targets
        else np.empty(0, dtype=np.float64)
    )
    sample_auxiliary_arrays = {
        name: (
            np.concatenate(values)
            if values
            else np.empty(0, dtype=np.float64)
        )
        for name, values in sample_auxiliaries.items()
    }
    reliability_array = sample_auxiliary_arrays["sensor_reliability"]
    if reliability_array.size != sample_count or quality_array.size != sample_count:
        raise RuntimeError(
            "sensor reliability diagnostics do not match evaluated sample count"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    active_parameters_getter = getattr(model, "active_parameters", None)
    if callable(active_parameters_getter):
        active_parameters = list(active_parameters_getter())
    else:
        active_parameters = list(model.parameters())
    active_parameter_count = sum(
        parameter.numel() for parameter in active_parameters
    )
    active_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in active_parameters
        if parameter.requires_grad
    )

    def feature_row(name: str, index: int) -> list[float]:
        values = sample_auxiliary_arrays[name]
        if values.size == 0:
            return []
        return np.asarray(values[index], dtype=np.float64).reshape(-1).tolist()

    records = []
    for index, sample_id in enumerate(sample_ids):
        predicted_id = int(prediction_ids[index])
        target_id = int(target_array[index])
        records.append(
            {
                "sample_id": sample_id,
                "source_image_id": source_image_ids[index],
                "target_id": target_id,
                "target": class_names[target_id],
                "predicted_id": predicted_id,
                "predicted": class_names[predicted_id],
                "confidence": float(probability_array[index, predicted_id]),
                "correct": predicted_id == target_id,
                "sensor_available": float(
                    sample_auxiliary_arrays["sensor_available"][index]
                ),
                "sensor_quality_target": float(quality_array[index]),
                "sensor_reliability": float(reliability_array[index]),
                "sensor_consistency": float(
                    sample_auxiliary_arrays["sensor_consistency"][index]
                ),
                "sensor_discrepancy": float(
                    sample_auxiliary_arrays["sensor_discrepancy"][index]
                ),
                "sensor_uncertainty": float(
                    sample_auxiliary_arrays["sensor_uncertainty"][index]
                ),
                "sensor_reliability_features": feature_row(
                    "sensor_reliability_features", index
                ),
                "sensor_grid_quality_features": feature_row(
                    "sensor_grid_quality_features", index
                ),
                "sensor_cross_modal_quality_features": feature_row(
                    "sensor_cross_modal_quality_features", index
                ),
                "probabilities": probability_array[index].tolist(),
            }
        )

    computed["diagnostics"] = _mean_auxiliary(auxiliary_totals, sample_count)
    computed["sensor_reliability_calibration"] = reliability_quality_metrics(
        quality_array,
        reliability_array,
    )
    computed["sensor_condition"] = {
        "degradation_enabled": sensor_degradation is not None,
        "degradation_seed": int(degradation_seed),
        "availability_mean": condition_totals["availability"]
        / max(sample_count, 1),
        "mean_absolute_grid_change": condition_totals["mean_absolute_change"]
        / max(sample_count, 1),
        "quality_target_mean": condition_totals["quality_target"]
        / max(sample_count, 1),
    }
    latent_elements = int(reconstruction_totals["latent_elements"])
    grid_elements = int(reconstruction_totals["grid_elements"])
    latent_target_variance = None
    latent_normalized_mse = None
    latent_r2 = None
    if latent_targets:
        latent_prediction_array = np.concatenate(latent_predictions, axis=0)
        latent_target_array = np.concatenate(latent_targets, axis=0)
        latent_residual_sum = float(
            np.square(latent_prediction_array - latent_target_array).sum()
        )
        latent_center = latent_target_array.mean(axis=0, keepdims=True)
        latent_total_sum = float(
            np.square(latent_target_array - latent_center).sum()
        )
        latent_target_variance = latent_total_sum / latent_target_array.size
        if latent_target_variance > 0.0:
            latent_normalized_mse = (
                latent_residual_sum / latent_target_array.size
            ) / latent_target_variance
            latent_r2 = 1.0 - latent_residual_sum / latent_total_sum
    grid_target_variance = None
    grid_normalized_mse = None
    grid_r2 = None
    if grid_targets:
        grid_prediction_array = np.concatenate(grid_predictions, axis=0)
        grid_target_array = np.concatenate(grid_targets, axis=0)
        grid_residual_sum = float(
            np.square(grid_prediction_array - grid_target_array).sum()
        )
        grid_center = grid_target_array.mean(axis=0, keepdims=True)
        grid_total_sum = float(np.square(grid_target_array - grid_center).sum())
        grid_target_variance = grid_total_sum / grid_target_array.size
        if grid_target_variance > 0.0:
            grid_normalized_mse = (
                grid_residual_sum / grid_target_array.size
            ) / grid_target_variance
            grid_r2 = 1.0 - grid_residual_sum / grid_total_sum
    computed["fallback_reconstruction"] = {
        "latent_mse": (
            reconstruction_totals["latent_squared_error"] / latent_elements
            if latent_elements
            else None
        ),
        "latent_mae": (
            reconstruction_totals["latent_absolute_error"] / latent_elements
            if latent_elements
            else None
        ),
        "latent_target_variance": latent_target_variance,
        "latent_normalized_mse": latent_normalized_mse,
        "latent_r2": latent_r2,
        "grid_mse": (
            reconstruction_totals["grid_squared_error"] / grid_elements
            if grid_elements
            else None
        ),
        "grid_mae": (
            reconstruction_totals["grid_absolute_error"] / grid_elements
            if grid_elements
            else None
        ),
        "grid_target_variance": grid_target_variance,
        "grid_normalized_mse": grid_normalized_mse,
        "grid_r2": grid_r2,
        "latent_element_count": latent_elements,
        "grid_element_count": grid_elements,
        "target": "clean physical sensor observation",
    }
    computed["performance"] = {
        "device": str(device),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "active_parameter_count": int(active_parameter_count),
        "active_trainable_parameter_count": int(
            active_trainable_parameter_count
        ),
        "evaluation_batch_size": loader.batch_size,
        "batch_count": len(forward_times),
        "warmup_batches": max(0, int(warmup_batches)),
        "forward_seconds": float(total_forward_seconds),
        "amortized_forward_ms_per_sample": float(
            1000.0 * total_forward_seconds / max(sample_count, 1)
        ),
        "throughput_samples_per_second": float(
            sample_count / total_forward_seconds
            if total_forward_seconds > 0.0
            else math.inf
        ),
        "timing_scope": "model_forward_only",
        "timing_note": (
            "Amortized over evaluation batches; not end-to-end single-sample latency."
        ),
    }
    computed["prediction_records"] = records
    return computed


def _normalized_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )


def _write_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    class_names: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row.tolist()])


def _write_classification_report_csv(
    path: Path, per_class: Iterable[dict[str, Any]]
) -> None:
    fields = ("label_id", "label", "support", "precision", "recall", "f1")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_class)


def _write_confusion_plot(
    path: Path,
    matrix: np.ndarray,
    class_names: Sequence[str],
    *,
    split: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if any(any(ord(character) > 127 for character in name) for name in class_names):
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False

    normalized = _normalized_confusion_matrix(matrix)
    size = max(8.0, min(13.0, len(class_names) * 0.58))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized rate")
    indices = np.arange(len(class_names))
    axis.set(
        xticks=indices,
        yticks=indices,
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual class",
        xlabel="Predicted class",
        title=f"Normalized Confusion Matrix ({split})",
    )
    plt.setp(axis.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
    axis.tick_params(axis="both", labelsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_evaluation_artifacts(
    result: dict[str, Any],
    output_dir: Path,
    *,
    class_names: Sequence[str],
    split: str,
) -> dict[str, str]:
    """Write stable UTF-8 evaluation files and return their absolute paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
    normalized = _normalized_confusion_matrix(matrix)
    predictions = result.get("prediction_records", [])
    summary = {
        key: value for key, value in result.items() if key != "prediction_records"
    }

    evaluation_path = output_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "classification_report.csv"
    _write_classification_report_csv(report_path, result["per_class"])
    confusion_path = output_dir / "confusion_matrix.csv"
    _write_matrix_csv(confusion_path, matrix, class_names)
    normalized_path = output_dir / "confusion_matrix_normalized.csv"
    _write_matrix_csv(normalized_path, normalized, class_names)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    plot_path = output_dir / "confusion_matrix.png"
    _write_confusion_plot(plot_path, matrix, class_names, split=split)

    return {
        "evaluation_json": str(evaluation_path.resolve()),
        "classification_report_csv": str(report_path.resolve()),
        "confusion_matrix_csv": str(confusion_path.resolve()),
        "normalized_confusion_matrix_csv": str(normalized_path.resolve()),
        "predictions_jsonl": str(predictions_path.resolve()),
        "confusion_matrix_png": str(plot_path.resolve()),
    }
