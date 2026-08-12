from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from mmfc.evaluation import (
    compute_classification_metrics,
    evaluate_model,
    expected_calibration_error,
    grouped_bootstrap_classification_intervals,
    grouped_paired_bootstrap_differences,
    reliability_quality_metrics,
    write_evaluation_artifacts,
)
from evaluate_robustness import _resolve_seeds
from evaluate import apply_evaluation_interventions
from mmfc.model import MultimodalFusionClassifier
from mmfc.robustness import (
    aggregate_robustness_runs,
    summarize_robustness,
    write_multiseed_robustness_artifacts,
)


def _known_predictions() -> tuple[np.ndarray, np.ndarray]:
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.8, 0.2],
        ],
        dtype=np.float64,
    )
    return targets, probabilities


def test_classification_metrics_match_known_binary_example() -> None:
    targets, probabilities = _known_predictions()

    result = compute_classification_metrics(
        targets,
        probabilities,
        ("recyclable", "hazardous"),
        ece_bins=2,
    )

    metrics = result["metrics"]
    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_precision"] == pytest.approx(5 / 6)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(11 / 15)
    assert metrics["weighted_f1"] == pytest.approx(11 / 15)
    assert metrics["top3_accuracy"] == pytest.approx(1.0)
    assert metrics["ece"] == pytest.approx(0.025)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["per_class"][0] == {
        "label_id": 0,
        "label": "recyclable",
        "support": 2,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(0.8),
    }


def test_ece_rejects_invalid_shapes_and_bins() -> None:
    targets, probabilities = _known_predictions()

    with pytest.raises(ValueError, match="样本数一致"):
        expected_calibration_error(probabilities[:2], targets, bins=2)
    with pytest.raises(ValueError, match="大于等于 2"):
        expected_calibration_error(probabilities, targets, bins=1)
    with pytest.raises(ValueError, match="每行和为 1"):
        compute_classification_metrics(
            targets,
            probabilities * 0.5,
            ("recyclable", "hazardous"),
        )


def test_reliability_quality_metrics_report_rank_and_pairwise_ordering() -> None:
    metrics = reliability_quality_metrics(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.2, 0.6, 0.9]),
    )

    assert metrics["mae"] == pytest.approx(0.4 / 3.0)
    assert metrics["pearson_correlation"] is not None
    assert metrics["spearman_correlation"] == pytest.approx(1.0)
    assert metrics["pairwise_ordering_accuracy"] == pytest.approx(1.0)
    assert metrics["pair_count"] == 3

    constant = reliability_quality_metrics(
        np.ones(3),
        np.full(3, 0.8),
    )
    assert constant["pearson_correlation"] is None
    assert constant["spearman_correlation"] is None
    assert constant["pairwise_ordering_accuracy"] is None
    assert constant["pair_count"] == 0


def test_grouped_bootstrap_intervals_are_deterministic() -> None:
    targets, probabilities = _known_predictions()
    first = grouped_bootstrap_classification_intervals(
        targets,
        probabilities,
        ["image-a", "image-a", "image-b", "image-b"],
        ("recyclable", "hazardous"),
        repeats=100,
        seed=17,
    )
    second = grouped_bootstrap_classification_intervals(
        targets,
        probabilities,
        ["image-a", "image-a", "image-b", "image-b"],
        ("recyclable", "hazardous"),
        repeats=100,
        seed=17,
    )

    assert first == second
    assert first["cluster_count"] == 2
    assert first["sample_count"] == 4
    assert set(first["intervals"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "negative_log_likelihood",
        "multiclass_brier_score",
        "ece",
    }
    assert (
        first["intervals"]["accuracy"]["lower"]
        <= first["intervals"]["accuracy"]["upper"]
    )


def test_grouped_paired_bootstrap_reports_candidate_minus_reference() -> None:
    targets, reference = _known_predictions()
    candidate = reference.copy()
    candidate[-1] = [0.1, 0.9]
    result = grouped_paired_bootstrap_differences(
        targets,
        reference,
        candidate,
        ["image-a", "image-a", "image-b", "image-b"],
        ("recyclable", "hazardous"),
        repeats=100,
        seed=17,
    )

    assert result["differences"]["accuracy"]["point"] == pytest.approx(0.25)
    assert result["differences"]["macro_f1"]["point"] > 0.0
    assert result["differences"]["negative_log_likelihood"][
        "preferred_direction"
    ] == "lower"
    assert (
        result["differences"]["accuracy"]["probability_greater_than_zero"]
        > 0.0
    )


def test_apply_evaluation_interventions_bypasses_gate_after_loading() -> None:
    model = _AvailabilityDrivenModel()
    model.sensor_reliability_enabled = True

    metadata = apply_evaluation_interventions(
        model,
        disable_sensor_reliability=True,
    )

    assert model.sensor_reliability_enabled is False
    assert metadata["sensor_reliability_configured"] is True
    assert metadata["sensor_reliability_disabled"] is True
    assert metadata["sensor_reliability_active"] is False


def test_apply_evaluation_interventions_supports_fine_grained_paths() -> None:
    model = MultimodalFusionClassifier(
        num_classes=3,
        sensor_grid_quality_enabled=True,
        sensor_cross_modal_quality_enabled=True,
    )

    metadata = apply_evaluation_interventions(
        model,
        posthoc_modalities="visual_text",
        disabled_reliability_evidence=("uncertainty",),
        disable_film=True,
        disable_attention=True,
    )

    assert model.enabled_modalities == ("visual", "text")
    torch.testing.assert_close(
        model.modality_mask,
        torch.tensor([1.0, 1.0, 0.0]),
    )
    assert model.sensor_reliability_enabled is False
    assert model.sensor_grid_quality_enabled is False
    assert model.sensor_cross_modal_quality_enabled is False
    assert model.film_enabled is False
    assert model.fusion_mode == "mean"
    assert metadata["sensor_reliability_configured"] is True
    assert metadata["reliability_evidence_disabled"] == ["uncertainty"]


def test_write_evaluation_artifacts_creates_utf8_tables_and_plot(tmp_path) -> None:
    targets, probabilities = _known_predictions()
    result = compute_classification_metrics(
        targets,
        probabilities,
        ("可回收物", "有害垃圾"),
        ece_bins=2,
    )
    predictions = result.pop("predictions")
    result["dataset"] = {"split": "test"}
    result["diagnostics"] = {"sensor_reliability": [0.9]}
    result["performance"] = {"device": "cpu"}
    result["prediction_records"] = [
        {
            "sample_id": f"sample-{index}",
            "target_id": int(target),
            "target": ("可回收物", "有害垃圾")[target],
            "predicted_id": int(predictions[index]),
            "predicted": ("可回收物", "有害垃圾")[predictions[index]],
            "confidence": float(probabilities[index, predictions[index]]),
            "correct": bool(predictions[index] == target),
            "probabilities": probabilities[index].tolist(),
        }
        for index, target in enumerate(targets)
    ]

    artifacts = write_evaluation_artifacts(
        result,
        tmp_path,
        class_names=("可回收物", "有害垃圾"),
        split="test",
    )

    assert set(artifacts) == {
        "evaluation_json",
        "classification_report_csv",
        "confusion_matrix_csv",
        "normalized_confusion_matrix_csv",
        "predictions_jsonl",
        "confusion_matrix_png",
    }
    for path in artifacts.values():
        assert __import__("pathlib").Path(path).is_file()
    summary = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["accuracy"] == pytest.approx(0.75)
    with (tmp_path / "classification_report.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["可回收物", "有害垃圾"]
    prediction_lines = (tmp_path / "predictions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(prediction_lines) == 4
    assert json.loads(prediction_lines[0])["target"] == "可回收物"
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 1000

class _AvailabilityDrivenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.sensor_encoder = nn.Flatten(start_dim=1)
        self.sensor_cross_modal_quality_enabled = True

    def forward(
        self,
        images,
        sensors,
        *,
        sensor_available,
        return_aux,
    ):
        batch_size = images.shape[0]
        logits = torch.cat(
            (2.0 * sensor_available, 1.0 - sensor_available), dim=1
        ) + self.anchor * 0.0
        reliability = 0.9 * sensor_available
        aux = {
            "sensor_reliability": reliability,
            "sensor_consistency": torch.full_like(sensor_available, 0.8),
            "sensor_discrepancy": 1.0 - sensor_available,
            "sensor_uncertainty": torch.full_like(sensor_available, 0.2),
            "sensor_available": sensor_available,
            "modality_weights": torch.tensor(
                [0.4, 0.3, 0.3], device=images.device
            ).expand(batch_size, -1),
            "scale_weights": torch.tensor(
                [0.3, 0.4, 0.3], device=images.device
            ).expand(batch_size, -1),
            "predicted_sensor": torch.full(
                (batch_size, 16), 0.25, device=images.device
            ),
            "predicted_sensor_grid": torch.full_like(sensors, 0.25),
        }
        return (logits, aux) if return_aux else logits


def test_evaluate_model_applies_missing_condition_and_reports_drop() -> None:
    batches = [
        {
            "sample_id": f"sample-{index}",
            "image": torch.zeros(3, 2, 2),
            "sensor": torch.full((1, 4, 4), 0.5),
            "label": torch.tensor(0),
        }
        for index in range(4)
    ]
    loader = DataLoader(batches, batch_size=2, shuffle=False)
    model = _AvailabilityDrivenModel()
    common = {
        "device": torch.device("cpu"),
        "class_names": ("correct", "fallback"),
        "warmup_batches": 0,
    }

    clean = evaluate_model(model, loader, **common)
    missing = evaluate_model(
        model,
        loader,
        **common,
        sensor_degradation={"enabled": True, "missing_probability": 1.0},
        degradation_seed=7,
    )
    rows = summarize_robustness({"clean": clean, "missing_100": missing})

    assert clean["metrics"]["accuracy"] == pytest.approx(1.0)
    assert clean["sensor_condition"]["availability_mean"] == pytest.approx(1.0)
    assert missing["metrics"]["accuracy"] == pytest.approx(0.0)
    assert missing["sensor_condition"]["availability_mean"] == pytest.approx(0.0)
    assert missing["sensor_condition"]["mean_absolute_grid_change"] == pytest.approx(
        0.5
    )
    assert missing["diagnostics"]["sensor_reliability"] == pytest.approx([0.0])
    assert missing["sensor_reliability_calibration"]["mae"] == pytest.approx(0.0)
    assert clean["sensor_reliability_calibration"]["mae"] == pytest.approx(0.1)
    assert missing["fallback_reconstruction"]["latent_mse"] == pytest.approx(
        0.0625
    )
    assert missing["fallback_reconstruction"]["grid_mae"] == pytest.approx(0.25)
    assert missing["fallback_reconstruction"]["grid_mse"] == pytest.approx(
        0.0625
    )
    assert rows[1]["reliability_mae"] == pytest.approx(0.0)
    assert rows[1]["accuracy_drop_pp"] == pytest.approx(100.0)


def _minimal_robustness_result(
    accuracy: float,
    macro_f1: float,
    pearson: float | None,
) -> dict:
    return {
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": macro_f1,
            "ece": 0.05,
        },
        "sensor_condition": {},
        "sensor_reliability_calibration": {
            "pearson_correlation": pearson,
        },
        "diagnostics": {},
    }


def test_multiseed_robustness_uses_sample_std_and_writes_artifacts(
    tmp_path,
) -> None:
    seed_results = {
        11: {
            "clean": _minimal_robustness_result(0.90, 0.80, None),
            "missing_50": _minimal_robustness_result(0.80, 0.70, 0.20),
        },
        12: {
            "clean": _minimal_robustness_result(0.80, 0.70, None),
            "missing_50": _minimal_robustness_result(0.75, 0.65, 0.40),
        },
    }

    rows = aggregate_robustness_runs(seed_results)
    degraded = rows[1]

    assert degraded["accuracy_mean"] == pytest.approx(0.775)
    assert degraded["accuracy_std"] == pytest.approx(0.0353553391)
    assert degraded["accuracy_drop_pp_mean"] == pytest.approx(7.5)
    assert degraded["accuracy_drop_pp_std"] == pytest.approx(3.53553391)
    assert degraded["reliability_pearson_mean"] == pytest.approx(0.3)
    assert degraded["reliability_pearson_std"] == pytest.approx(0.141421356)
    assert degraded["reliability_pearson_valid_seed_count"] == 2
    assert rows[0]["reliability_pearson_mean"] is None
    assert rows[0]["reliability_pearson_valid_seed_count"] == 0

    artifacts = write_multiseed_robustness_artifacts(
        seed_results, tmp_path
    )
    assert set(artifacts) == {
        "summary_csv",
        "summary_json",
        "curves_png",
    }
    for path in artifacts.values():
        assert __import__("pathlib").Path(path).is_file()
    payload = json.loads(
        (tmp_path / "robustness_multiseed_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["seeds"] == [11, 12]
    assert payload["seed_count"] == 2


def test_resolve_robustness_seeds_is_backward_compatible() -> None:
    assert _resolve_seeds(None, None) == [2026]
    assert _resolve_seeds(7, None) == [7]
    assert _resolve_seeds(None, [3, 3, 5]) == [3, 5]
    with pytest.raises(ValueError, match="cannot be used together"):
        _resolve_seeds(7, [8])
