"""Evaluate fixed linear probes on Qwen-assisted soft-sensor attributes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mmfc.data import write_json
from mmfc.evaluation import compute_classification_metrics
from mmfc.training import load_config, read_jsonl, resolve_project_path


SENSOR_PROPERTY_NAMES = (
    "weight_g",
    "temperature_c",
    "moisture_pct",
    "conductivity_us_cm",
    "odor_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed logistic-regression probes for soft-sensor features"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/qwen-material/sensor-linear-probe"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=3000)
    return parser.parse_args()


def per_sample_standardize_grid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("grid values must have shape [N,16]")
    mean = values.mean(axis=1, keepdims=True)
    standard_deviation = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(standard_deviation, 1e-6)


def load_sensor_feature_table(
    manifest_path: Path,
    sensor_records: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    grids: list[np.ndarray] = []
    properties: list[list[float]] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    for record in read_jsonl(manifest_path):
        sample_id = str(record["sample_id"])
        if sample_id not in sensor_records:
            raise ValueError(f"sensor record is missing for sample: {sample_id}")
        sensor = sensor_records[sample_id]
        grid = np.asarray(sensor["normalized_grid"], dtype=np.float64)
        if grid.shape != (4, 4) or not bool(np.isfinite(grid).all()):
            raise ValueError(f"invalid normalized_grid for sample: {sample_id}")
        sensor_properties = sensor.get("properties", {})
        values = [float(sensor_properties[name]) for name in SENSOR_PROPERTY_NAMES]
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"non-finite sensor property for sample: {sample_id}")
        grids.append(grid.reshape(-1))
        properties.append(values)
        labels.append(int(record["label_id"]))
        sample_ids.append(sample_id)
    if not grids:
        raise ValueError(f"manifest has no samples: {manifest_path}")
    return {
        "sample_ids": sample_ids,
        "grid": np.asarray(grids, dtype=np.float64),
        "properties": np.asarray(properties, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
    }


def _metric_payload(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    computed = compute_classification_metrics(
        targets,
        probabilities,
        class_names,
    )
    return {
        "metrics": computed["metrics"],
        "per_class": computed["per_class"],
        "confusion_matrix": computed["confusion_matrix"],
    }


def fit_linear_sensor_probes(
    train: Mapping[str, Any],
    test: Mapping[str, Any],
    class_names: Sequence[str],
    *,
    seed: int = 42,
    max_iter: int = 3000,
) -> dict[str, Any]:
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0")
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    test_labels = np.asarray(test["labels"], dtype=np.int64)
    train_grid = np.asarray(train["grid"], dtype=np.float64)
    test_grid = np.asarray(test["grid"], dtype=np.float64)
    train_properties = np.asarray(train["properties"], dtype=np.float64)
    test_properties = np.asarray(test["properties"], dtype=np.float64)
    feature_sets = {
        "normalized_grid_raw": (train_grid, test_grid),
        "normalized_grid_per_sample_zscore": (
            per_sample_standardize_grid(train_grid),
            per_sample_standardize_grid(test_grid),
        ),
        "five_properties": (train_properties, test_properties),
        "grid_plus_properties": (
            np.concatenate((train_grid, train_properties), axis=1),
            np.concatenate((test_grid, test_properties), axis=1),
        ),
    }
    class_count = len(class_names)
    probes: dict[str, Any] = {}
    for name, (train_features, test_features) in feature_sets.items():
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=int(max_iter),
                random_state=int(seed),
            ),
        )
        estimator.fit(train_features, train_labels)
        probabilities = np.zeros(
            (test_labels.size, class_count), dtype=np.float64
        )
        classifier = estimator.named_steps["logisticregression"]
        probabilities[:, classifier.classes_.astype(np.int64)] = (
            estimator.predict_proba(test_features)
        )
        probes[name] = {
            "feature_dimension": int(train_features.shape[1]),
            **_metric_payload(test_labels, probabilities, class_names),
        }

    counts = np.bincount(train_labels, minlength=class_count).astype(np.float64)
    prior = counts / counts.sum()
    prior_probabilities = np.broadcast_to(
        prior, (test_labels.size, class_count)
    ).copy()
    probes["train_class_prior"] = {
        "feature_dimension": 0,
        **_metric_payload(test_labels, prior_probabilities, class_names),
    }
    return {
        "method": {
            "estimator": "StandardScaler + LogisticRegression",
            "seed": int(seed),
            "max_iter": int(max_iter),
            "hyperparameter_search": False,
            "train_split_only": True,
        },
        "train_sample_count": int(train_labels.size),
        "evaluation_sample_count": int(test_labels.size),
        "property_names": list(SENSOR_PROPERTY_NAMES),
        "probes": probes,
    }


def write_probe_artifacts(output: Path, result: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "sensor_linear_probe.json", result)
    metric_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "negative_log_likelihood",
        "multiclass_brier_score",
        "ece",
    )
    with (output / "sensor_linear_probe.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("probe", "feature_dimension", *metric_names),
        )
        writer.writeheader()
        for name, probe in result["probes"].items():
            writer.writerow(
                {
                    "probe": name,
                    "feature_dimension": probe["feature_dimension"],
                    **{
                        metric: probe["metrics"][metric]
                        for metric in metric_names
                    },
                }
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    sensor_list = read_jsonl(
        resolve_project_path(data_config["sensor_manifest"])
    )
    sensor_records: dict[str, dict[str, Any]] = {}
    for record in sensor_list:
        sample_id = str(record["sample_id"])
        if sample_id in sensor_records:
            raise ValueError(f"duplicate sensor sample_id: {sample_id}")
        sensor_records[sample_id] = record
    train = load_sensor_feature_table(
        resolve_project_path(data_config["train_manifest"]),
        sensor_records,
    )
    evaluation = load_sensor_feature_table(
        resolve_project_path(data_config[f"{args.split}_manifest"]),
        sensor_records,
    )
    class_names = [
        item["name"] if isinstance(item, dict) else str(item)
        for item in config["classes"]
    ]
    result = fit_linear_sensor_probes(
        train,
        evaluation,
        class_names,
        seed=args.seed,
        max_iter=args.max_iter,
    )
    result["evaluation_split"] = args.split
    output = resolve_project_path(args.output)
    write_probe_artifacts(output, result)
    summary = {
        name: {
            "accuracy": probe["metrics"]["accuracy"],
            "macro_f1": probe["metrics"]["macro_f1"],
        }
        for name, probe in result["probes"].items()
    }
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
