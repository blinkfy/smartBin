from __future__ import annotations

import inspect
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from mmfc.losses import JointClassificationLoss
import mmfc.model as model_module
from mmfc.model import (
    MultimodalFusionClassifier,
    SensorEncoder,
    _import_external_ultralytics,
    build_production_model,
)
from mmfc.training import (
    ManifestDataset,
    degrade_sensor_grid,
    _run_epoch,
    load_cropped_image,
    load_sensor_grid,
)
from predict import (
    _load_cli_grid,
    _load_model_state_compatible,
    _torch_load,
    _validated_checkpoint_class_names,
)


def _feature_inputs(batch_size: int = 4, num_classes: int = 6):
    generator = torch.Generator().manual_seed(7)
    scales = [torch.randn(batch_size, 512, generator=generator) for _ in range(3)]
    sensor = torch.randn(batch_size, 512, generator=generator)
    prototypes = torch.randn(num_classes, 512, generator=generator)
    return scales, sensor, prototypes


def test_forward_from_features_shapes_and_normalized_weights():
    model = MultimodalFusionClassifier(num_classes=6)
    model.eval()
    scales, sensor, prototypes = _feature_inputs()
    logits, aux = model.forward_from_features(
        scales, sensor, prototypes, return_aux=True
    )

    assert logits.shape == (4, 6)
    assert aux["prototype_weights"].shape == (4, 6)
    assert aux["modality_weights"].shape == (4, 3)
    assert aux["scale_weights"].shape == (4, 3)
    assert aux["attention_weights"].shape == (4, 8, 3, 3)
    assert aux["sensor_reliability"].shape == (4, 1)
    assert aux["predicted_sensor"].shape == (4, 512)
    assert aux["sensor_log_variance"].shape == (4, 512)
    assert aux["sensor_available"].shape == (4, 1)
    assert aux["sensor_consistency"].shape == (4, 1)
    torch.testing.assert_close(aux["sensor_available"], torch.ones(4, 1))
    assert torch.all((aux["sensor_reliability"] >= 0.0) & (aux["sensor_reliability"] <= 1.0))
    assert torch.all((aux["sensor_consistency"] >= 0.0) & (aux["sensor_consistency"] <= 1.0))
    for key in ("prototype_weights", "modality_weights", "scale_weights"):
        torch.testing.assert_close(
            aux[key].sum(dim=-1), torch.ones(4), atol=1e-6, rtol=1e-6
        )


def test_dynamic_scale_prior_and_film_initialization():
    model = MultimodalFusionClassifier(num_classes=6)
    model.eval()
    scales, sensor, prototypes = _feature_inputs()
    _, aux = model.forward_from_features(scales, sensor, prototypes, return_aux=True)

    expected = torch.tensor([0.3, 0.5, 0.2]).expand(4, -1)
    torch.testing.assert_close(aux["scale_weights"], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model.film_gamma, torch.ones_like(model.film_gamma))
    torch.testing.assert_close(
        model.film_beta, torch.full_like(model.film_beta, 0.1)
    )
    for adapter in (
        model.visual_adapter,
        model.text_adapter,
        model.sensor_adapter,
    ):
        torch.testing.assert_close(adapter[0].bias, torch.zeros_like(adapter[0].bias))
    torch.testing.assert_close(
        model.down_projection.bias, torch.zeros_like(model.down_projection.bias)
    )
    torch.testing.assert_close(
        model.up_projection.bias, torch.zeros_like(model.up_projection.bias)
    )


def test_configurable_film_initialization_and_validation():
    model = MultimodalFusionClassifier(
        num_classes=6,
        film_gamma_init=0.8,
        film_beta_init=-0.05,
    )

    torch.testing.assert_close(model.film_gamma, torch.full_like(model.film_gamma, 0.8))
    torch.testing.assert_close(model.film_beta, torch.full_like(model.film_beta, -0.05))
    with pytest.raises(ValueError, match="film_gamma_init"):
        MultimodalFusionClassifier(num_classes=6, film_gamma_init=0.0)
    with pytest.raises(ValueError, match="film_beta_init"):
        MultimodalFusionClassifier(num_classes=6, film_beta_init=float("nan"))


def test_forward_from_features_uses_predicted_sensor_when_missing():
    model = MultimodalFusionClassifier(num_classes=6)
    model.eval()
    scales, sensor, prototypes = _feature_inputs(batch_size=3)
    logits, aux = model.forward_from_features(
        scales,
        sensor,
        prototypes,
        sensor_available=torch.zeros(3, 1),
        return_aux=True,
    )

    assert logits.shape == (3, 6)
    torch.testing.assert_close(aux["sensor_available"], torch.zeros(3, 1))
    torch.testing.assert_close(aux["sensor_reliability"], torch.zeros(3, 1))
    torch.testing.assert_close(aux["effective_sensor"], aux["predicted_sensor"])


def test_joint_loss_backward_reaches_model_parameters():
    model = MultimodalFusionClassifier(num_classes=6)
    scales, sensor, prototypes = _feature_inputs()
    logits = model.forward_from_features(scales, sensor, prototypes)
    targets = torch.tensor([0, 2, 1, 5])
    criterion = JointClassificationLoss(
        ce_weight=1.0, focal_weight=0.5, alpha=0.25, gamma=2.0
    )
    loss, components = criterion(logits, targets, return_components=True)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(components["cross_entropy"])
    assert torch.isfinite(components["focal"])
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_auxiliary_sensor_losses_backpropagate_to_reliability_heads():
    model = MultimodalFusionClassifier(num_classes=6)
    scales, sensor, prototypes = _feature_inputs()
    logits, aux = model.forward_from_features(
        scales,
        sensor,
        prototypes,
        sensor_available=torch.tensor([[1.0], [0.3], [0.0], [1.0]]),
        return_aux=True,
    )
    targets = torch.tensor([0, 2, 1, 5])
    sensor_target = sensor.detach() + 0.05
    quality_target = torch.tensor([[1.0], [0.4], [0.0], [0.8]])
    criterion = JointClassificationLoss(
        ce_weight=1.0,
        focal_weight=0.5,
        alpha=0.25,
        gamma=2.0,
        reconstruction_weight=0.2,
        calibration_weight=0.1,
    )
    loss, components = criterion(
        logits,
        targets,
        aux=aux,
        sensor_target=sensor_target,
        quality_target=quality_target,
        return_components=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(components["sensor_reconstruction"])
    assert torch.isfinite(components["reliability_calibration"])
    assert model.sensor_mean_predictor[-1].weight.grad is not None
    assert model.sensor_reliability_predictor[-1].bias.grad is not None


def test_reconstruction_loss_masked_nan_sample_does_not_pollute_gradients():
    logits = torch.tensor([[1.0, -0.2], [-0.1, 0.8]], requires_grad=True)
    targets = torch.tensor([0, 1])
    predicted = torch.tensor(
        [[0.2, -0.1], [float("nan"), 0.4]],
        requires_grad=True,
    )
    log_variance = torch.zeros(2, 2, requires_grad=True)
    aux = {
        "predicted_sensor": predicted,
        "sensor_log_variance": log_variance,
    }
    sensor_target = torch.zeros(2, 2)
    criterion = JointClassificationLoss(
        ce_weight=1.0,
        focal_weight=0.0,
        reconstruction_weight=1.0,
    )

    loss, components = criterion(
        logits,
        targets,
        aux=aux,
        sensor_target=sensor_target,
        reconstruction_mask=torch.tensor([[1.0], [0.0]]),
        return_components=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(components["sensor_reconstruction"])
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad[0]).all()
    assert predicted.grad[0].abs().sum() > 0
    torch.testing.assert_close(predicted.grad[1], torch.zeros_like(predicted.grad[1]))


def test_forward_signature_has_no_label_or_target_parameter():
    parameters = inspect.signature(MultimodalFusionClassifier.forward).parameters
    assert "label" not in parameters
    assert "labels" not in parameters
    assert "target" not in parameters
    assert "targets" not in parameters


class _VisualStub(nn.Module):
    def forward(self, images):
        pooled = images.mean(dim=(2, 3))
        base = torch.nn.functional.pad(pooled, (0, 509))
        return [base, base + 0.1, base + 0.2]


class _TextStub(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.seen_descriptions = None

    def forward(self, descriptions, device):
        self.seen_descriptions = tuple(descriptions)
        values = torch.arange(
            self.num_classes * 512, dtype=torch.float32, device=device
        )
        return values.reshape(self.num_classes, 512)


def test_injected_forward_encodes_all_candidates_without_target_text():
    descriptions = [f"候选类别 {index}" for index in range(6)]
    text_encoder = _TextStub(6)
    model = MultimodalFusionClassifier(
        num_classes=6,
        class_descriptions=descriptions,
        visual_encoder=_VisualStub(),
        text_encoder=text_encoder,
        sensor_encoder=SensorEncoder(),
    )
    model.eval()
    logits = model(torch.rand(2, 3, 24, 24), torch.rand(2, 1, 4, 4))
    assert logits.shape == (2, 6)
    assert text_encoder.seen_descriptions == tuple(descriptions)


def test_sensor_encoder_shape_validation_and_output():
    encoder = SensorEncoder()
    torch.testing.assert_close(
        encoder.projection.bias, torch.zeros_like(encoder.projection.bias)
    )
    output = encoder(torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4))
    assert output.shape == (2, 512)
    try:
        encoder(torch.zeros(2, 4, 4))
    except ValueError as exc:
        assert "[B,1,4,4]" in str(exc)
    else:
        raise AssertionError("非法传感器形状应触发 ValueError")


def test_sensor_degradation_is_seeded_and_bounded():
    sensors = torch.linspace(0.0, 1.0, steps=64).reshape(4, 1, 4, 4)
    config = {
        "sensor_degradation": {
            "enabled": True,
            "missing_probability": 0.25,
            "noise_probability": 1.0,
            "noise_std": 0.05,
            "bias_probability": 1.0,
            "bias_strength": 0.1,
            "stuck_probability": 0.25,
            "quality_scale": 0.2,
        }
    }
    first = degrade_sensor_grid(
        sensors, config, generator=torch.Generator().manual_seed(123)
    )
    second = degrade_sensor_grid(
        sensors, config, generator=torch.Generator().manual_seed(123)
    )

    for key in ("degraded_grid", "availability", "quality_target"):
        torch.testing.assert_close(first[key], second[key])
        assert torch.isfinite(first[key]).all()
    assert first["degraded_grid"].shape == sensors.shape
    assert first["availability"].shape == (4, 1)
    assert first["quality_target"].shape == (4, 1)
    assert torch.all((first["degraded_grid"] >= 0.0) & (first["degraded_grid"] <= 1.0))
    assert torch.all((first["availability"] >= 0.0) & (first["availability"] <= 1.0))
    assert torch.all((first["quality_target"] >= 0.0) & (first["quality_target"] <= 1.0))


def test_sensor_degradation_missing_zeroes_grid_and_quality():
    sensors = torch.ones(2, 1, 4, 4)
    degraded = degrade_sensor_grid(
        sensors,
        {"enabled": True, "missing_probability": 1.0},
        generator=torch.Generator().manual_seed(5),
    )

    torch.testing.assert_close(degraded["degraded_grid"], torch.zeros_like(sensors))
    torch.testing.assert_close(degraded["availability"], torch.zeros(2, 1))
    torch.testing.assert_close(degraded["quality_target"], torch.zeros(2, 1))


def _structured_sensor_batch() -> torch.Tensor:
    base = torch.tensor(
        [
            [0.02, 0.18, 0.64, 0.91],
            [0.11, 0.37, 0.49, 0.83],
            [0.26, 0.58, 0.73, 0.35],
            [0.07, 0.44, 0.88, 0.52],
        ],
        dtype=torch.float32,
    )
    return torch.stack((base, base.flip(0)), dim=0).unsqueeze(1)


def test_sensor_degradation_noise_only_changes_values_and_quality():
    sensors = torch.full((2, 1, 4, 4), 0.5)
    degraded = degrade_sensor_grid(
        sensors,
        {
            "enabled": True,
            "missing_probability": 0.0,
            "noise_probability": 1.0,
            "noise_std": 0.08,
            "bias_probability": 0.0,
            "stuck_probability": 0.0,
            "quality_scale": 0.2,
        },
        generator=torch.Generator().manual_seed(11),
    )

    torch.testing.assert_close(degraded["availability"], torch.ones(2, 1))
    assert not torch.allclose(degraded["degraded_grid"], sensors)
    assert torch.all((degraded["degraded_grid"] >= 0.0) & (degraded["degraded_grid"] <= 1.0))
    assert torch.all(degraded["quality_target"] < 1.0)


def test_sensor_degradation_bias_drift_is_spatial_and_encoder_visible():
    sensors = _structured_sensor_batch()
    config = {
        "enabled": True,
        "missing_probability": 0.0,
        "noise_probability": 0.0,
        "bias_probability": 1.0,
        "bias_strength": 0.3,
        "bias_min_strength": 0.3,
        "stuck_probability": 0.0,
        "quality_scale": 0.3,
    }
    first = degrade_sensor_grid(
        sensors, config, generator=torch.Generator().manual_seed(23)
    )
    second = degrade_sensor_grid(
        sensors, config, generator=torch.Generator().manual_seed(23)
    )

    torch.testing.assert_close(first["degraded_grid"], second["degraded_grid"])
    torch.testing.assert_close(first["availability"], torch.ones(2, 1))
    assert not torch.allclose(first["degraded_grid"], sensors)
    drift = first["degraded_grid"] - sensors
    assert drift.flatten(start_dim=1).std(dim=1).min() > 0.01
    assert torch.all((first["quality_target"] >= 0.0) & (first["quality_target"] < 1.0))

    torch.manual_seed(31)
    encoder = SensorEncoder()
    clean_latent = encoder(sensors)
    drift_latent = encoder(first["degraded_grid"])
    assert (drift_latent - clean_latent).abs().mean() > 1e-4


def test_sensor_degradation_stuck_only_replaces_with_constant_reading():
    sensors = _structured_sensor_batch()
    degraded = degrade_sensor_grid(
        sensors,
        {
            "enabled": True,
            "missing_probability": 0.0,
            "noise_probability": 0.0,
            "bias_probability": 0.0,
            "stuck_probability": 1.0,
            "stuck_value": 0.25,
            "quality_scale": 0.3,
        },
        generator=torch.Generator().manual_seed(41),
    )

    torch.testing.assert_close(degraded["degraded_grid"], torch.full_like(sensors, 0.25))
    torch.testing.assert_close(degraded["availability"], torch.ones(2, 1))
    assert torch.all(degraded["quality_target"] < 1.0)


def test_actual_manifest_schema_with_sensor_sidecar(tmp_path):
    image_path = tmp_path / "原图.png"
    pixels = np.zeros((12, 16, 3), dtype=np.uint8)
    pixels[2:10, 3:13] = np.array([120, 80, 40], dtype=np.uint8)
    Image.fromarray(pixels).save(image_path)

    manifest_path = tmp_path / "train.jsonl"
    sample = {
        "sample_id": "sample-001",
        "image_path": str(image_path),
        "bbox": [3, 2, 13, 10],
        "label": "battery",
    }
    manifest_path.write_text(
        json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sensor_path = tmp_path / "sensors.jsonl"
    sensor_record = {
        "sample_id": "sample-001",
        "grid": (100 + np.arange(16, dtype=np.float32)).reshape(4, 4).tolist(),
        "normalized_grid": (
            np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
        ).tolist(),
    }
    sensor_path.write_text(
        json.dumps(sensor_record, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    files_before_loading = {path.name for path in tmp_path.iterdir()}
    dataset = ManifestDataset(
        manifest_path,
        class_names=["paper", "battery"],
        image_size=20,
        sensor_manifest=sensor_path,
    )
    item = dataset[0]
    assert item["sample_id"] == "sample-001"
    assert item["image"].shape == (3, 20, 20)
    assert item["sensor"].shape == (1, 4, 4)
    torch.testing.assert_close(
        item["sensor"],
        torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 15.0,
    )
    assert item["label"].item() == 1

    enabled_dataset = ManifestDataset(
        manifest_path,
        class_names=["paper", "battery"],
        image_size=18,
        sensor_manifest=sensor_path,
        preprocess={
            "enabled": True,
            "bilateral": {
                "diameter": 3,
                "sigma_space": 1.0,
                "sigma_color": 0.8,
            },
            "clahe": {
                "enabled": True,
                "brightness_threshold": 1.0,
                "clip_limit": 2.0,
                "tile_grid_size": [8, 8],
            },
            "laplacian": {"enabled": True, "kernel_size": 3, "strength": 1.0},
            "affine_matrix": [[1, 0, 0], [0, 1, 0]],
        },
    )
    enabled_item = enabled_dataset[0]
    assert enabled_item["image"].shape == (3, 18, 18)
    assert {path.name for path in tmp_path.iterdir()} == files_before_loading


def test_manifest_rejects_duplicate_sensor_sidecar_sample_id_aliases(tmp_path):
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": "duplicate-001",
                "image_path": "unused.png",
                "bbox": [0, 0, 1, 1],
                "label": "paper",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    grid = np.zeros((4, 4), dtype=np.float32).tolist()
    sensor_path = tmp_path / "sensors.jsonl"
    sensor_path.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "duplicate-001", "normalized_grid": grid}),
                json.dumps({"id": "duplicate-001", "normalized_grid": grid}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"重复 sample_id/id.*duplicate-001"):
        ManifestDataset(
            manifest_path,
            class_names=["paper"],
            sensor_manifest=sensor_path,
        )


def test_run_epoch_one_epoch_train_and_eval_smoke():
    class TinyFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(28, 3)

        def forward(self, images, sensors):
            features = torch.cat(
                [images.flatten(start_dim=1), sensors.flatten(start_dim=1)],
                dim=1,
            )
            return self.classifier(features)

    generator = torch.Generator().manual_seed(17)
    batch = {
        "image": torch.randn(6, 3, 2, 2, generator=generator),
        "sensor": torch.randn(6, 1, 4, 4, generator=generator),
        "label": torch.tensor([0, 1, 2, 0, 1, 2]),
    }
    model = TinyFusionModel()
    criterion = JointClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    weights_before = model.classifier.weight.detach().clone()

    train_metrics = _run_epoch(
        model, [batch], criterion, torch.device("cpu"), optimizer
    )
    weights_after_train = model.classifier.weight.detach().clone()
    eval_metrics = _run_epoch(
        model, [batch], criterion, torch.device("cpu"), None
    )

    assert not torch.equal(weights_before, weights_after_train)
    assert all(np.isfinite(value) for value in (*train_metrics, *eval_metrics))
    assert 0.0 <= train_metrics[1] <= 1.0
    assert 0.0 <= eval_metrics[1] <= 1.0
    torch.testing.assert_close(model.classifier.weight, weights_after_train)


def test_run_epoch_robust_path_passes_degradation_targets_and_metrics():
    class RecordingSensorEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(16, 512)
            self.outputs = []

        def forward(self, sensors):
            output = self.projection(sensors.flatten(start_dim=1))
            self.outputs.append(output.detach().clone())
            return output

    class RobustPathModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sensor_encoder = RecordingSensorEncoder()
            self.classifier = nn.Linear(28, 3)
            self.reliability = nn.Linear(28, 1)
            self.received_sensors = None
            self.received_availability = None

        def forward(self, images, sensors, *, sensor_available=None, return_aux=False):
            self.received_sensors = sensors.detach().clone()
            self.received_availability = sensor_available.detach().clone()
            features = torch.cat(
                [images.flatten(start_dim=1), sensors.flatten(start_dim=1)],
                dim=1,
            )
            logits = self.classifier(features)
            aux = {
                "sensor_reliability": torch.sigmoid(self.reliability(features)),
                "sensor_consistency": torch.ones(
                    sensors.shape[0], 1, device=sensors.device
                )
                * 0.5,
            }
            return (logits, aux) if return_aux else logits

    class RecordingCriterion(nn.Module):
        def __init__(self):
            super().__init__()
            self.sensor_target = None
            self.quality_target = None
            self.reconstruction_mask = None
            self.aux = None

        def forward(
            self,
            logits,
            targets,
            *,
            aux=None,
            sensor_target=None,
            quality_target=None,
            reconstruction_mask=None,
            return_components=False,
        ):
            self.sensor_target = sensor_target
            self.quality_target = quality_target
            self.reconstruction_mask = reconstruction_mask
            self.aux = aux
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss = loss + 0.01 * aux["sensor_reliability"].mean()
            components = {
                "cross_entropy": loss,
                "sensor_reconstruction": sensor_target.pow(2).mean(),
                "reliability_calibration": quality_target.mean(),
            }
            return (loss, components) if return_components else loss

    generator = torch.Generator().manual_seed(19)
    batch = {
        "image": torch.randn(4, 3, 2, 2, generator=generator),
        "sensor": torch.rand(4, 1, 4, 4, generator=generator),
        "label": torch.tensor([0, 1, 2, 1]),
    }
    model = RobustPathModel()
    criterion = RecordingCriterion()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    weights_before = model.classifier.weight.detach().clone()

    loss, accuracy, metrics = _run_epoch(
        model,
        [batch],
        criterion,
        torch.device("cpu"),
        optimizer,
        robustness_config={"enabled": True, "missing_probability": 1.0},
        return_metrics=True,
    )

    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
    assert not torch.equal(weights_before, model.classifier.weight.detach())
    torch.testing.assert_close(model.received_sensors, torch.zeros_like(batch["sensor"]))
    torch.testing.assert_close(model.received_availability, torch.zeros(4, 1))
    assert criterion.sensor_target is not None
    assert criterion.sensor_target.requires_grad is False
    torch.testing.assert_close(criterion.sensor_target, model.sensor_encoder.outputs[0])
    torch.testing.assert_close(criterion.quality_target, torch.zeros(4, 1))
    torch.testing.assert_close(criterion.reconstruction_mask, torch.ones(4, 1))
    assert criterion.aux is not None
    assert "sensor_reliability" in metrics
    assert "sensor_consistency" in metrics
    assert "sensor_reconstruction" in metrics
    assert "reliability_calibration" in metrics


def test_external_ultralytics_import_rejects_preloaded_local_without_mutation(
    monkeypatch,
):
    local_module = SimpleNamespace(
        __file__=str(model_module.LOCAL_ULTRALYTICS_ROOT / "__init__.py"),
        __version__="8.4.70",
    )
    monkeypatch.setitem(sys.modules, "ultralytics", local_module)
    sys_path_before = list(sys.path)

    with pytest.raises(RuntimeError, match="smartBin/ultralytics"):
        _import_external_ultralytics()

    assert sys.modules["ultralytics"] is local_module
    assert sys.path == sys_path_before


def test_external_ultralytics_import_temporarily_excludes_cwd_and_smartbin(
    tmp_path, monkeypatch
):
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    monkeypatch.chdir(model_module.SMARTBIN_ROOT)
    external_origin = tmp_path / "site-packages" / "ultralytics" / "__init__.py"
    external_origin.parent.mkdir(parents=True)
    external_origin.write_text("", encoding="utf-8")
    external_module = SimpleNamespace(
        __file__=str(external_origin),
        __version__="8.4.70",
    )
    original_path = ["", str(model_module.SMARTBIN_ROOT), str(tmp_path)]
    monkeypatch.setattr(sys, "path", list(original_path))
    observed_path = None

    def fake_import(name):
        nonlocal observed_path
        assert name == "ultralytics"
        observed_path = list(sys.path)
        return external_module

    monkeypatch.setattr(model_module.importlib, "import_module", fake_import)
    imported = _import_external_ultralytics()

    assert imported is external_module
    assert observed_path == [str(tmp_path)]
    assert sys.path == original_path


def test_external_ultralytics_import_rejects_incompatible_version(
    tmp_path, monkeypatch
):
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    external_module = SimpleNamespace(
        __file__=str(tmp_path / "site-packages" / "ultralytics" / "__init__.py"),
        __version__="9.0.0",
    )
    monkeypatch.setattr(
        model_module.importlib,
        "import_module",
        lambda name: external_module,
    )
    with pytest.raises(RuntimeError, match=r">=8\.3,<9"):
        _import_external_ultralytics()


def test_build_production_model_passes_configured_freeze_counts(monkeypatch):
    observed = {}

    class VisualStub(nn.Module):
        def __init__(self, weights_path, output_dim=512, freeze_layers=5):
            super().__init__()
            observed["yolo"] = (weights_path, output_dim, freeze_layers)

    class TextStub(nn.Module):
        def __init__(
            self,
            model_name,
            freeze_layers=3,
            local_files_only=False,
        ):
            super().__init__()
            observed["clip"] = (model_name, freeze_layers, local_files_only)

    monkeypatch.setattr(model_module, "YOLOMultiScaleEncoder", VisualStub)
    monkeypatch.setattr(model_module, "CLIPPrototypeEncoder", TextStub)
    config = {
        "classes": [
            {"name": "a", "description": "A"},
            {"name": "b", "description": "B"},
        ],
        "model": {
            "yolo_weights": "unused.pt",
            "clip_model": "clip-stub",
            "clip_local_files_only": True,
            "freeze_yolo_layers": 7,
            "freeze_clip_text_layers": 4,
            "film_gamma_init": 0.75,
            "film_beta_init": -0.02,
        },
    }

    model = build_production_model(config)

    assert observed["yolo"][2] == 7
    assert observed["clip"] == ("clip-stub", 4, True)
    torch.testing.assert_close(model.film_gamma, torch.full_like(model.film_gamma, 0.75))
    torch.testing.assert_close(model.film_beta, torch.full_like(model.film_beta, -0.02))


def test_inline_normalized_sensor_grid_has_priority(tmp_path):
    normalized = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    sensor = load_sensor_grid(
        {
            "sample_id": "inline-001",
            "normalized_grid": normalized.tolist(),
            "sensor_grid": (100 + np.arange(16)).reshape(4, 4).tolist(),
        },
        manifest_dir=tmp_path,
    )
    torch.testing.assert_close(
        sensor,
        torch.from_numpy(normalized).unsqueeze(0),
    )


def test_manifest_rejects_label_id_inconsistent_with_config(tmp_path):
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": "bad-mapping",
                "image_path": "unused.png",
                "bbox": [0, 0, 1, 1],
                "label": "battery",
                "label_id": 0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ManifestDataset(
        manifest_path,
        class_names=["paper", "battery"],
    )

    with pytest.raises(ValueError, match="label/label_id.*不一致"):
        dataset[0]


def test_load_cropped_image_applies_exif_orientation_before_bbox(tmp_path):
    image_path = tmp_path / "oriented.png"
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[0, 0] = [255, 0, 0]
    pixels[1, 0] = [0, 255, 0]
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(pixels).save(image_path, exif=exif)

    crop = load_cropped_image(
        image_path,
        [0, 0, 1, 1],
        manifest_dir=tmp_path,
        image_size=1,
    )

    assert crop[1, 0, 0] > 0.9
    assert crop[0, 0, 0] < 0.1


def test_prediction_rejects_reordered_config_class_names():
    checkpoint = {
        "model_state": {},
        "class_names": ["potato", "battery"],
    }
    reordered_config = {
        "classes": [
            {"name": "battery"},
            {"name": "potato"},
        ]
    }

    with pytest.raises(ValueError, match="顺序不一致"):
        _validated_checkpoint_class_names(checkpoint, reordered_config)

    matching_config = {
        "classes": [
            {"name": "potato"},
            {"name": "battery"},
        ]
    }
    assert _validated_checkpoint_class_names(
        checkpoint, matching_config
    ) == ("potato", "battery")


def test_prediction_checkpoint_load_is_weights_only_and_preserves_metadata(
    tmp_path, monkeypatch
):
    checkpoint_path = tmp_path / "safe.pt"
    observed = []
    expected = {
        "model_state": {"classifier.weight": torch.ones(2, 2)},
        "class_names": ["potato", "battery"],
    }

    def fake_load(path, *, map_location, weights_only):
        observed.append((path, map_location, weights_only))
        return expected

    monkeypatch.setattr(torch, "load", fake_load)
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))

    assert observed == [
        (checkpoint_path, torch.device("cpu"), True),
    ]
    assert checkpoint is expected
    assert _validated_checkpoint_class_names(
        checkpoint,
        {"classes": [{"name": "potato"}, {"name": "battery"}]},
    ) == ("potato", "battery")


def test_prediction_checkpoint_refuses_pytorch_without_weights_only(
    tmp_path, monkeypatch
):
    calls = []

    def unsupported_load(path, *, map_location, weights_only):
        calls.append((path, map_location, weights_only))
        raise TypeError("unexpected keyword argument 'weights_only'")

    monkeypatch.setattr(torch, "load", unsupported_load)
    checkpoint_path = tmp_path / "unsafe.pt"
    with pytest.raises(RuntimeError, match=r"weights_only=True.*升级"):
        _torch_load(checkpoint_path, torch.device("cpu"))
    assert calls == [(checkpoint_path, torch.device("cpu"), True)]


_SENSOR_HEAD_PREFIXES = (
    "sensor_mean_predictor.",
    "sensor_logvar_predictor.",
    "sensor_reliability_predictor.",
)


def _legacy_state_without_sensor_heads(model):
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(_SENSOR_HEAD_PREFIXES)
    }


def test_legacy_checkpoint_available_sensor_passthrough_exact():
    model = MultimodalFusionClassifier(num_classes=3)
    _load_model_state_compatible(model, _legacy_state_without_sensor_heads(model))
    model.eval()
    scales, sensor, prototypes = _feature_inputs(batch_size=2, num_classes=3)

    _, aux = model.forward_from_features(
        scales,
        sensor,
        prototypes,
        sensor_available=torch.ones(2, 1),
        return_aux=True,
    )

    assert model.legacy_sensor_passthrough is True
    torch.testing.assert_close(aux["sensor_reliability"], torch.ones(2, 1))
    torch.testing.assert_close(aux["effective_sensor"], sensor)


def test_legacy_checkpoint_missing_sensor_is_rejected():
    model = MultimodalFusionClassifier(num_classes=3)
    _load_model_state_compatible(model, _legacy_state_without_sensor_heads(model))
    scales, sensor, prototypes = _feature_inputs(batch_size=2, num_classes=3)

    with pytest.raises(ValueError, match="legacy checkpoint requires sensor_available=1"):
        model.forward_from_features(
            scales,
            sensor,
            prototypes,
            sensor_available=torch.tensor([[1.0], [0.0]]),
            return_aux=True,
        )


def test_prediction_checkpoint_loader_rejects_partial_missing_sensor_heads():
    model = MultimodalFusionClassifier(num_classes=3)
    state = model.state_dict()
    partial_state = dict(state)
    partial_state.pop(next(key for key in state if key.startswith("sensor_mean_predictor.")))

    with pytest.raises(RuntimeError, match="partial sensor reliability"):
        _load_model_state_compatible(model, partial_state)


def test_prediction_checkpoint_loader_accepts_full_new_state_and_rejects_old_gaps():
    model = MultimodalFusionClassifier(num_classes=3)
    _load_model_state_compatible(model, model.state_dict())
    assert model.legacy_sensor_passthrough is False
    scales, sensor, prototypes = _feature_inputs(batch_size=2, num_classes=3)
    _, aux = model.forward_from_features(
        scales,
        sensor,
        prototypes,
        sensor_available=torch.zeros(2, 1),
        return_aux=True,
    )
    torch.testing.assert_close(aux["sensor_reliability"], torch.zeros(2, 1))

    old_state = _legacy_state_without_sensor_heads(model)
    _load_model_state_compatible(model, old_state)
    assert model.legacy_sensor_passthrough is True
    incompatible = dict(old_state)
    incompatible.pop("classifier.weight")
    with pytest.raises(RuntimeError, match="classifier.weight"):
        _load_model_state_compatible(model, incompatible)


@pytest.mark.parametrize("out_of_range", [-0.001, 1.001])
def test_prediction_cli_rejects_sensor_values_outside_unit_interval(
    out_of_range,
):
    values = np.zeros((4, 4), dtype=np.float32)
    values[0, 0] = out_of_range
    args = SimpleNamespace(
        sensor_json=None,
        sensor_grid=json.dumps(values.tolist()),
    )

    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _load_cli_grid(args)


def test_prediction_cli_sensor_missing_returns_zero_grid_and_unavailable():
    args = SimpleNamespace(
        sensor_missing=True,
        sensor_json=None,
        sensor_grid=None,
    )

    grid, availability = _load_cli_grid(args, return_availability=True)

    torch.testing.assert_close(grid, torch.zeros(1, 1, 4, 4))
    torch.testing.assert_close(availability, torch.zeros(1, 1))
    torch.testing.assert_close(_load_cli_grid(args), torch.zeros(1, 1, 4, 4))
