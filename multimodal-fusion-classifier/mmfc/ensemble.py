"""Validation-selected probability ensembles for aligned prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .evaluation import compute_classification_metrics


@dataclass(frozen=True)
class PredictionTable:
    """One aligned prediction artifact loaded from JSONL."""

    path: Path
    records: tuple[dict[str, Any], ...]
    sample_ids: tuple[str, ...]
    source_image_ids: tuple[str, ...]
    targets: np.ndarray
    probabilities: np.ndarray


def load_prediction_table(path: str | Path) -> PredictionTable:
    """Load, validate, and deterministically order one prediction artifact."""

    resolved = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "sample_id",
                "source_image_id",
                "target_id",
                "probabilities",
            }
            missing = sorted(required.difference(row))
            if missing:
                raise ValueError(
                    f"{resolved}:{line_number} missing fields: {missing}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"prediction artifact is empty: {resolved}")

    rows.sort(key=lambda row: str(row["sample_id"]))
    sample_ids = tuple(str(row["sample_id"]) for row in rows)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"duplicate sample_id in {resolved}")

    probabilities = np.asarray(
        [row["probabilities"] for row in rows],
        dtype=np.float64,
    )
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"invalid probability shape in {resolved}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError(f"invalid probability values in {resolved}")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or np.any(np.abs(row_sums - 1.0) > 1e-3):
        raise ValueError(f"probability rows do not sum to one in {resolved}")
    probabilities = probabilities / row_sums

    targets = np.asarray([row["target_id"] for row in rows], dtype=np.int64)
    return PredictionTable(
        path=resolved,
        records=tuple(rows),
        sample_ids=sample_ids,
        source_image_ids=tuple(
            str(row["source_image_id"]) for row in rows
        ),
        targets=targets,
        probabilities=probabilities,
    )


def validate_aligned_tables(tables: Sequence[PredictionTable]) -> None:
    """Require identical sample identity, source grouping, labels, and shape."""

    if len(tables) < 2:
        raise ValueError("an ensemble requires at least two prediction tables")
    reference = tables[0]
    for table in tables[1:]:
        if table.sample_ids != reference.sample_ids:
            raise ValueError("ensemble member sample_id ordering differs")
        if table.source_image_ids != reference.source_image_ids:
            raise ValueError("ensemble member source_image_id ordering differs")
        if not np.array_equal(table.targets, reference.targets):
            raise ValueError("ensemble member targets differ")
        if table.probabilities.shape != reference.probabilities.shape:
            raise ValueError("ensemble member probability shapes differ")


def weighted_probability_pool(
    tables: Sequence[PredictionTable],
    weights: Sequence[float],
) -> np.ndarray:
    """Return a normalized arithmetic pool of aligned class probabilities."""

    validate_aligned_tables(tables)
    vector = np.asarray(weights, dtype=np.float64)
    if vector.shape != (len(tables),):
        raise ValueError("weight count must match ensemble member count")
    if not np.isfinite(vector).all() or np.any(vector < 0.0):
        raise ValueError("ensemble weights must be finite and non-negative")
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("at least one ensemble weight must be positive")
    vector = vector / total
    pooled = np.zeros_like(tables[0].probabilities, dtype=np.float64)
    for weight, table in zip(vector, tables):
        pooled += float(weight) * table.probabilities
    pooled /= pooled.sum(axis=1, keepdims=True)
    return pooled


def simplex_grid(member_count: int, denominator: int) -> Iterator[np.ndarray]:
    """Enumerate non-negative simplex weights on a fixed integer grid."""

    if member_count < 2:
        raise ValueError("member_count must be at least two")
    if denominator < 1:
        raise ValueError("denominator must be positive")

    def compositions(remaining: int, width: int) -> Iterator[tuple[int, ...]]:
        if width == 1:
            yield (remaining,)
            return
        for value in range(remaining + 1):
            for suffix in compositions(remaining - value, width - 1):
                yield (value, *suffix)

    for values in compositions(denominator, member_count):
        yield np.asarray(values, dtype=np.float64) / denominator


def select_validation_weights(
    tables: Sequence[PredictionTable],
    class_names: Sequence[str],
    *,
    denominator: int = 20,
) -> dict[str, Any]:
    """Select weights by validation Macro F1 with deterministic tie-breaks."""

    validate_aligned_tables(tables)
    uniform = np.full(len(tables), 1.0 / len(tables), dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best: dict[str, Any] | None = None
    for weights in simplex_grid(len(tables), denominator):
        probabilities = weighted_probability_pool(tables, weights)
        result = compute_classification_metrics(
            tables[0].targets,
            probabilities,
            class_names,
        )
        metrics = result["metrics"]
        candidate = {
            "weights": weights.tolist(),
            "metrics": metrics,
        }
        candidates.append(candidate)
        key = (
            float(metrics["macro_f1"]),
            float(metrics["accuracy"]),
            -float(metrics["negative_log_likelihood"]),
            -float(np.square(weights - uniform).sum()),
            *(-weights).tolist(),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = candidate
    assert best is not None
    return {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "objective": (
            "maximize Macro F1, then Accuracy, then minimize NLL, then "
            "prefer weights nearest uniform"
        ),
        "grid_denominator": int(denominator),
        "candidate_count": len(candidates),
        "selected": best,
        "candidates": candidates,
    }


def build_ensemble_result(
    tables: Sequence[PredictionTable],
    member_names: Sequence[str],
    weights: Sequence[float],
    class_names: Sequence[str],
) -> tuple[dict[str, Any], np.ndarray]:
    """Build standard evaluation output and auditable prediction records."""

    if len(member_names) != len(tables):
        raise ValueError("member name count must match prediction table count")
    probabilities = weighted_probability_pool(tables, weights)
    result = compute_classification_metrics(
        tables[0].targets,
        probabilities,
        class_names,
    )
    predictions = result.pop("predictions")
    records: list[dict[str, Any]] = []
    for index, target_id in enumerate(tables[0].targets):
        predicted_id = int(predictions[index])
        member_outputs = {
            name: {
                "predicted_id": int(table.probabilities[index].argmax()),
                "confidence": float(table.probabilities[index].max()),
            }
            for name, table in zip(member_names, tables)
        }
        records.append(
            {
                "sample_id": tables[0].sample_ids[index],
                "source_image_id": tables[0].source_image_ids[index],
                "target_id": int(target_id),
                "target": str(class_names[int(target_id)]),
                "predicted_id": predicted_id,
                "predicted": str(class_names[predicted_id]),
                "confidence": float(probabilities[index, predicted_id]),
                "correct": bool(predicted_id == target_id),
                "probabilities": probabilities[index].tolist(),
                "member_outputs": member_outputs,
            }
        )
    result["prediction_records"] = records
    result["ensemble"] = {
        "pooling": "weighted_arithmetic_probability_average",
        "member_names": list(member_names),
        "weights": [float(value) for value in weights],
    }
    result["class_names"] = list(class_names)
    return result, probabilities


def selective_risk_profile(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    coverage_points: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 1.0),
) -> dict[str, Any]:
    """Measure confidence-ranked selective risk without dropping headline data."""

    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.size:
        raise ValueError("selective probabilities must align with targets")
    if targets.size == 0:
        raise ValueError("selective evaluation requires at least one sample")
    points = tuple(float(value) for value in coverage_points)
    if any(value <= 0.0 or value > 1.0 for value in points):
        raise ValueError("coverage points must lie in (0,1]")

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    cumulative_errors = np.cumsum(predictions[order] != targets[order])
    accepted = np.arange(1, targets.size + 1, dtype=np.int64)
    risks = cumulative_errors / accepted
    profile = []
    for requested in points:
        count = min(targets.size, max(1, int(np.ceil(requested * targets.size))))
        index = count - 1
        profile.append(
            {
                "requested_coverage": requested,
                "coverage": float(count / targets.size),
                "accepted_samples": int(count),
                "threshold": float(confidence[order[index]]),
                "selective_accuracy": float(1.0 - risks[index]),
                "selective_risk": float(risks[index]),
            }
        )
    return {
        "ranking": "descending_max_probability",
        "aurc": float(risks.mean()),
        "sample_count": int(targets.size),
        "points": profile,
    }


def validation_locked_selective_points(
    validation_targets: np.ndarray,
    validation_probabilities: np.ndarray,
    test_targets: np.ndarray,
    test_probabilities: np.ndarray,
    *,
    target_accuracies: Sequence[float] = (0.95, 0.975),
) -> list[dict[str, Any]]:
    """Choose maximum-coverage confidence thresholds on validation only."""

    validation_targets = np.asarray(validation_targets, dtype=np.int64)
    test_targets = np.asarray(test_targets, dtype=np.int64)
    validation_probabilities = np.asarray(
        validation_probabilities,
        dtype=np.float64,
    )
    test_probabilities = np.asarray(test_probabilities, dtype=np.float64)
    if validation_probabilities.shape[0] != validation_targets.size:
        raise ValueError("validation probabilities must align with targets")
    if test_probabilities.shape[0] != test_targets.size:
        raise ValueError("test probabilities must align with targets")

    validation_prediction = validation_probabilities.argmax(axis=1)
    validation_confidence = validation_probabilities.max(axis=1)
    test_prediction = test_probabilities.argmax(axis=1)
    test_confidence = test_probabilities.max(axis=1)
    outcomes: list[dict[str, Any]] = []
    for target_accuracy in target_accuracies:
        target_accuracy = float(target_accuracy)
        if not 0.0 < target_accuracy <= 1.0:
            raise ValueError("target accuracies must lie in (0,1]")
        selected: dict[str, Any] | None = None
        for threshold in np.unique(validation_confidence):
            mask = validation_confidence >= threshold
            accuracy = float(
                np.mean(validation_prediction[mask] == validation_targets[mask])
            )
            coverage = float(mask.mean())
            candidate = {
                "threshold": float(threshold),
                "coverage": coverage,
                "accepted_samples": int(mask.sum()),
                "accuracy": accuracy,
            }
            if accuracy >= target_accuracy and (
                selected is None
                or coverage > selected["coverage"]
                or (
                    coverage == selected["coverage"]
                    and threshold < selected["threshold"]
                )
            ):
                selected = candidate
        if selected is None:
            outcomes.append(
                {
                    "target_validation_accuracy": target_accuracy,
                    "available": False,
                }
            )
            continue
        test_mask = test_confidence >= selected["threshold"]
        outcomes.append(
            {
                "target_validation_accuracy": target_accuracy,
                "available": True,
                "threshold": selected["threshold"],
                "validation": selected,
                "test": {
                    "coverage": float(test_mask.mean()),
                    "accepted_samples": int(test_mask.sum()),
                    "accuracy": float(
                        np.mean(test_prediction[test_mask] == test_targets[test_mask])
                    ),
                },
            }
        )
    return outcomes
