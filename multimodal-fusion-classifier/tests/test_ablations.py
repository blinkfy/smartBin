from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mmfc.ablations import (
    ablation_names,
    build_ablation_config,
    load_ablation_reports,
    summarize_ablation_reports,
    write_ablation_artifacts,
    write_ablation_configs,
)
from mmfc.model import MultimodalFusionClassifier


def _feature_inputs(
    batch_size: int = 3, num_classes: int = 4
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(73)
    scales = [
        torch.randn(batch_size, 512, generator=generator) for _ in range(3)
    ]
    sensor = torch.randn(batch_size, 512, generator=generator)
    prototypes = torch.randn(num_classes, 512, generator=generator)
    return scales, sensor, prototypes


def test_visual_only_ablation_masks_inputs_and_freezes_inactive_parameters() -> None:
    model = MultimodalFusionClassifier(
        num_classes=4,
        enabled_modalities=("visual",),
        fusion_mode="mean",
        film_enabled=False,
        sensor_reliability_enabled=False,
    )
    model.eval()
    scales, sensor, prototypes = _feature_inputs()

    first, aux = model.forward_from_features(
        scales, sensor, prototypes, return_aux=True
    )
    second = model.forward_from_features(
        scales,
        sensor + 1000.0,
        prototypes - 1000.0,
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        aux["modality_weights"],
        torch.tensor([[1.0, 0.0, 0.0]]).expand(3, -1),
    )
    torch.testing.assert_close(
        aux["modality_mask"],
        torch.tensor([[1.0, 0.0, 0.0]]).expand(3, -1),
    )
    torch.testing.assert_close(aux["prototype_weights"], torch.zeros(3, 4))
    assert all(
        not parameter.requires_grad
        for parameter in model.text_adapter.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.sensor_adapter.parameters()
    )
    assert sum(parameter.numel() for parameter in model.active_parameters()) < sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_vision_text_ablation_is_invariant_to_sensor_features() -> None:
    model = MultimodalFusionClassifier(
        num_classes=4,
        enabled_modalities=("visual", "text"),
        fusion_mode="mean",
        film_enabled=False,
        sensor_reliability_enabled=False,
    )
    model.eval()
    scales, sensor, prototypes = _feature_inputs()

    first, aux = model.forward_from_features(
        scales, sensor, prototypes, return_aux=True
    )
    second = model.forward_from_features(
        scales, -500.0 * sensor, prototypes
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        aux["modality_weights"],
        torch.tensor([[0.5, 0.5, 0.0]]).expand(3, -1),
    )
    assert torch.count_nonzero(aux["prototype_weights"]) > 0


def test_sensor_only_ablation_is_invariant_to_visual_and_text_features() -> None:
    model = MultimodalFusionClassifier(
        num_classes=4,
        enabled_modalities=("sensor",),
        fusion_mode="mean",
        film_enabled=False,
        sensor_reliability_enabled=False,
    )
    model.eval()
    scales, sensor, prototypes = _feature_inputs()

    first, aux = model.forward_from_features(
        scales, sensor, prototypes, return_aux=True
    )
    second = model.forward_from_features(
        [item + 1000.0 for item in scales], sensor, prototypes - 1000.0
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        aux["modality_weights"],
        torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1),
    )
    assert torch.count_nonzero(aux["prototype_weights"]) == 0
    assert all(
        not parameter.requires_grad
        for parameter in model.visual_adapter.parameters()
    )


def test_text_sensor_ablation_uses_sensor_conditioned_prototypes() -> None:
    model = MultimodalFusionClassifier(
        num_classes=4,
        enabled_modalities=("text", "sensor"),
        fusion_mode="mean",
        film_enabled=False,
        sensor_reliability_enabled=False,
    )
    model.eval()
    scales, sensor, prototypes = _feature_inputs()

    first, aux = model.forward_from_features(
        scales, sensor, prototypes, return_aux=True
    )
    second = model.forward_from_features(
        [item + 1000.0 for item in scales], sensor, prototypes
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        aux["modality_weights"],
        torch.tensor([[0.0, 0.5, 0.5]]).expand(3, -1),
    )
    assert torch.count_nonzero(aux["prototype_weights"]) > 0


def test_ablation_model_options_reject_ambiguous_variants() -> None:
    with pytest.raises(ValueError, match="sample-observation modality"):
        MultimodalFusionClassifier(
            num_classes=4, enabled_modalities=("text",)
        )
    with pytest.raises(ValueError, match="requires visual, text, and sensor"):
        MultimodalFusionClassifier(
            num_classes=4,
            enabled_modalities=("visual", "sensor"),
            sensor_reliability_enabled=True,
        )
    with pytest.raises(ValueError, match="fusion_mode"):
        MultimodalFusionClassifier(num_classes=4, fusion_mode="concatenate")


def _base_config() -> dict:
    return {
        "_config_path": "base.json",
        "classes": [{"name": "甲"}, {"name": "乙"}],
        "model": {"dropout": 0.3},
        "loss": {
            "reconstruction_weight": 0.2,
            "calibration_weight": 0.1,
        },
        "robustness": {"sensor_degradation": {"enabled": True}},
        "training": {"epochs": 100, "seed": 42},
    }


def test_ablation_configs_are_standalone_and_preserve_base_input(tmp_path) -> None:
    base = _base_config()
    configs = {
        name: build_ablation_config(base, name)
        for name in ablation_names()
    }

    assert base["_config_path"] == "base.json"
    assert configs["sensor_only"]["model"]["enabled_modalities"] == [
        "sensor"
    ]
    assert configs["vision_only"]["model"]["enabled_modalities"] == ["visual"]
    assert configs["vision_only"]["loss"]["reconstruction_weight"] == 0.0
    assert configs["text_sensor"]["model"]["enabled_modalities"] == [
        "text",
        "sensor",
    ]
    assert (
        configs["vision_only"]["robustness"]["sensor_degradation"]["enabled"]
        is False
    )
    assert (
        configs["full_no_degradation"]["robustness"]["sensor_degradation"][
            "enabled"
        ]
        is False
    )
    assert configs["full_no_reconstruction"]["loss"][
        "reconstruction_weight"
    ] == 0.0
    assert configs["full_no_calibration"]["loss"]["calibration_weight"] == 0.0
    quality_aware = configs["full_quality_aware"]
    assert quality_aware["model"]["sensor_grid_quality_enabled"] is True
    assert quality_aware["model"]["sensor_log_variance_min"] == -3.0
    assert quality_aware["loss"]["reconstruction_weight"] == 0.1
    assert quality_aware["loss"]["calibration_weight"] == 0.3
    assert quality_aware["robustness"]["sensor_degradation"][
        "noise_std_range"
    ] == [0.02, 0.15]
    quality_prior = configs["full_quality_aware_prior"]
    assert quality_prior["loss"]["variance_prior_weight"] == 0.1
    assert quality_prior["loss"]["variance_prior_target"] == -2.0
    assert quality_prior["loss"]["calibration_clean_weight"] == 3.0
    cross_modal = configs["full_cross_modal_quality"]
    assert cross_modal["model"]["sensor_cross_modal_quality_enabled"] is True
    assert cross_modal["loss"]["sensor_grid_reconstruction_weight"] == 0.5
    assert cross_modal["loss"]["calibration_ranking_weight"] == 0.05
    assert configs["full"]["experiment"]["reference_checkpoint"].endswith(
        "qwen-material/best.pt"
    )

    artifacts = write_ablation_configs(
        base,
        tmp_path,
        experiments=("vision_only", "full"),
        epochs=3,
        seed=9,
        max_train_samples=12,
        max_val_samples=6,
        save_checkpoints=False,
    )
    assert set(artifacts) == {"vision_only", "full"}
    written = json.loads(
        (tmp_path / "vision_only.json").read_text(encoding="utf-8")
    )
    assert written["classes"][0]["name"] == "甲"
    assert written["training"]["epochs"] == 3
    assert written["training"]["seed"] == 9
    assert written["training"]["max_train_samples"] == 12
    assert written["training"]["max_val_samples"] == 6
    assert written["training"]["save_checkpoints"] is False


def _evaluation_report(
    accuracy: float,
    macro_f1: float,
    *,
    checkpoint: str,
) -> dict:
    return {
        "metrics": {
            "sample_count": 20,
            "accuracy": accuracy,
            "macro_precision": macro_f1 - 0.01,
            "macro_recall": macro_f1 + 0.01,
            "macro_f1": macro_f1,
            "weighted_f1": accuracy - 0.01,
            "ece": 0.04,
            "negative_log_likelihood": 0.3,
            "multiclass_brier_score": 0.15,
        },
        "performance": {
            "parameter_count": 100,
            "trainable_parameter_count": 80,
            "active_parameter_count": 90,
            "active_trainable_parameter_count": 70,
            "amortized_forward_ms_per_sample": 1.2,
        },
        "checkpoint": {
            "epoch": 5,
            "path": checkpoint,
        },
    }


def test_ablation_summary_writes_deltas_and_protocol_warning(tmp_path) -> None:
    reports = {
        "vision_only": _evaluation_report(
            0.70, 0.60, checkpoint="vision.pt"
        ),
        "full": _evaluation_report(0.80, 0.75, checkpoint="full.pt"),
    }

    rows = summarize_ablation_reports(reports)
    assert rows[0]["accuracy_delta_pp"] == pytest.approx(-10.0)
    assert rows[0]["macro_f1_delta_pp"] == pytest.approx(-15.0)
    assert rows[1]["accuracy_delta_pp"] == pytest.approx(0.0)

    for name, report in reports.items():
        report_dir = tmp_path / "inputs" / name
        report_dir.mkdir(parents=True)
        (report_dir / "evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False),
            encoding="utf-8",
        )
    loaded = load_ablation_reports(
        tmp_path / "inputs",
        ("vision_only", "full"),
    )
    artifacts = write_ablation_artifacts(
        loaded,
        tmp_path / "output",
        protocol="posthoc_intervention",
    )

    assert set(artifacts) == {
        "summary_csv",
        "summary_json",
        "summary_markdown",
        "scores_png",
    }
    for value in artifacts.values():
        assert Path(value).is_file()
    markdown = (tmp_path / "output" / "ABLATION_RESULTS.md").read_text(
        encoding="utf-8"
    )
    assert "不能替代独立重训消融" in markdown
    assert (tmp_path / "output" / "ablation_scores.png").stat().st_size > 1000
