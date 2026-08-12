"""Reliability-aware external sensor conditioning for the joint Segment head."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .joint_seg import JointMultimodalSegmentationModel, SensorTextFeatureFusion
from .joint_seg_external import ExternalSceneSensorDataset
from .model import (
    SENSOR_CROSS_MODAL_QUALITY_DIM,
    cross_modal_sensor_grid_quality_features,
)
from .training import degrade_sensor_grid


DEFAULT_SENSOR_DEGRADATION: dict[str, Any] = {
    "enabled": True,
    "missing_probability": 0.10,
    "noise_probability": 0.50,
    "noise_std_range": [0.02, 0.15],
    "bias_probability": 0.25,
    "bias_strength": 0.15,
    "stuck_probability": 0.15,
    "quality_scale": 0.25,
}


def pack_reliable_sensor_channels(
    images: Tensor,
    grids: Tensor,
    availability: Tensor,
) -> Tensor:
    """Append grid and availability transport channels to uint8 RGB."""

    if (
        images.ndim != 4
        or images.shape[1] != 3
        or images.dtype != torch.uint8
    ):
        raise ValueError("images must be uint8 [B,3,H,W]")
    batch_size = int(images.shape[0])
    if grids.shape != (batch_size, 1, 4, 4):
        raise ValueError("grids must have shape [B,1,4,4]")
    if availability.ndim == 1:
        availability = availability.unsqueeze(1)
    if availability.shape != (batch_size, 1):
        raise ValueError("availability must have shape [B,1] or [B]")
    if not torch.isfinite(grids).all() or not torch.isfinite(availability).all():
        raise ValueError("sensor transport values must be finite")
    grid_raster = F.interpolate(
        grids.float().clamp(0.0, 1.0),
        size=images.shape[-2:],
        mode="nearest",
    )
    availability_raster = availability.float().clamp(0.0, 1.0).view(
        batch_size, 1, 1, 1
    ).expand(batch_size, 1, *images.shape[-2:])
    return torch.cat(
        [
            images,
            (grid_raster * 255.0).round().to(torch.uint8),
            (availability_raster * 255.0).round().to(torch.uint8),
        ],
        dim=1,
    )


def split_reliable_sensor_channels(
    inputs: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Split RGB, 4x4 sensor grid, and per-sample availability."""

    if inputs.ndim != 4 or inputs.shape[1] not in {3, 5}:
        raise ValueError("model input must be [B,3,H,W] or [B,5,H,W]")
    rgb = inputs[:, :3]
    if inputs.shape[1] == 3:
        return rgb, None, None
    sensor = F.adaptive_avg_pool2d(inputs[:, 3:4], output_size=(4, 4))
    availability = F.adaptive_avg_pool2d(
        inputs[:, 4:5], output_size=1
    ).flatten(1)
    return rgb, sensor.clamp(0.0, 1.0), availability.clamp(0.0, 1.0)


class ReliableExternalSceneSensorDataset(ExternalSceneSensorDataset):
    """External scene sensor dataset with deterministic software degradation."""

    def __init__(
        self,
        *args: Any,
        degradation: Mapping[str, Any] | None = None,
        degradation_seed: int = 42,
        degrade: bool = False,
        **kwargs: Any,
    ) -> None:
        self.degradation = dict(
            DEFAULT_SENSOR_DEGRADATION if degradation is None else degradation
        )
        self.degradation_seed = int(degradation_seed)
        self.degrade = bool(degrade)
        super().__init__(*args, **kwargs)

    def _degrade_one(
        self,
        grid: Tensor,
        source_id: str,
    ) -> dict[str, Tensor]:
        if not self.degrade:
            ones = torch.ones(1, 1, dtype=grid.dtype)
            return {
                "degraded_grid": grid.unsqueeze(0).clone(),
                "availability": ones,
                "quality_target": ones.clone(),
            }
        digest = hashlib.sha256(
            f"{self.degradation_seed}:{source_id}".encode("utf-8")
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little"))
        return degrade_sensor_grid(
            grid.unsqueeze(0),
            self.degradation,
            generator=generator,
        )

    def collate_fn(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        clean = [
            sample.pop("scene_sensor_observation") for sample in batch
        ]
        source_ids = [
            Path(str(sample["im_file"])).stem for sample in batch
        ]
        result = YOLODataset.collate_fn(batch)
        clean_grids = torch.stack(clean, dim=0)
        degraded = [
            self._degrade_one(grid, source_id)
            for grid, source_id in zip(clean, source_ids)
        ]
        observed = torch.cat(
            [item["degraded_grid"] for item in degraded], dim=0
        )
        availability = torch.cat(
            [item["availability"] for item in degraded], dim=0
        )
        quality = torch.cat(
            [item["quality_target"] for item in degraded], dim=0
        )
        result["img"] = pack_reliable_sensor_channels(
            result["img"], observed, availability
        )
        result["scene_sensor_observation"] = observed
        result["sensor_available"] = availability
        result["sensor_quality_target"] = quality
        result["soft_sensor_target"] = clean_grids
        result["soft_sensor_coverage"] = torch.ones_like(clean_grids)
        return result


class ReliabilityAwareSensorTextFeatureFusion(SensorTextFeatureFusion):
    """Blend observed and RGB-predicted grids using a calibrated gate."""

    def __init__(
        self,
        channels: Sequence[int],
        text_prototypes: Tensor,
        *,
        semantic_dim: int = 64,
        sensor_hidden_dim: int = 256,
        temperature: float = 0.10,
        reliability_hidden_dim: int = 32,
        reliability_init: float = 0.85,
    ) -> None:
        super().__init__(
            channels,
            text_prototypes,
            semantic_dim=semantic_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            temperature=temperature,
            text_enabled=True,
            sensor_enabled=True,
        )
        if reliability_hidden_dim <= 0:
            raise ValueError("reliability_hidden_dim must be positive")
        if not 0.0 < reliability_init < 1.0:
            raise ValueError("reliability_init must be in (0,1)")
        feature_dim = 4 + SENSOR_CROSS_MODAL_QUALITY_DIM
        self.sensor_reliability_predictor = nn.Sequential(
            nn.Linear(feature_dim, int(reliability_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(reliability_hidden_dim), 1),
        )
        nn.init.zeros_(self.sensor_reliability_predictor[-1].weight)
        nn.init.constant_(
            self.sensor_reliability_predictor[-1].bias,
            math.log(reliability_init / (1.0 - reliability_init)),
        )
        self._transport_availability: Tensor | None = None

    def forward(
        self,
        features: Sequence[Tensor],
        observed_sensor: Tensor | None = None,
        sensor_available: Tensor | None = None,
    ) -> list[Tensor]:
        if len(features) != len(self.channels):
            raise ValueError("expected exactly three YOLO feature maps")
        reference = features[0]
        batch_size = int(reference.shape[0])
        for feature, channels in zip(features, self.channels):
            if (
                feature.ndim != 4
                or feature.shape[0] != batch_size
                or feature.shape[1] != channels
            ):
                raise ValueError("YOLO feature map shape does not match fusion contract")

        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(feature, 1).flatten(1)
                for feature in features
            ],
            dim=1,
        )
        sensor_hidden = self.sensor_predictor(pooled)
        predicted_sensor = torch.sigmoid(
            self.sensor_mean(sensor_hidden)
        ).view(batch_size, 1, 4, 4)
        uncertainty = (
            F.softplus(self.sensor_uncertainty(sensor_hidden)) + 1e-4
        ).view(batch_size, 1, 4, 4)

        availability_source = (
            sensor_available
            if sensor_available is not None
            else self._transport_availability
        )
        observation_present = observed_sensor is not None
        if not observation_present:
            observed = predicted_sensor
            availability = reference.new_zeros(batch_size, 1)
        else:
            if observed_sensor.shape != (batch_size, 1, 4, 4):
                raise ValueError("observed_sensor must have shape [B,1,4,4]")
            observed = observed_sensor.to(
                device=reference.device,
                dtype=reference.dtype,
            ).clamp(0.0, 1.0)
            if availability_source is None:
                availability = reference.new_ones(batch_size, 1)
            else:
                availability = availability_source.to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
                if availability.ndim == 1:
                    availability = availability.unsqueeze(1)
                if availability.shape != (batch_size, 1):
                    raise ValueError(
                        "sensor_available must have shape [B,1] or [B]"
                    )
                availability = availability.clamp(0.0, 1.0)
        if observation_present:
            observed_float = observed.float()
            predicted_float = predicted_sensor.float()
            uncertainty_float = uncertainty.float()
            difference = observed_float - predicted_float
            precision = uncertainty_float.reciprocal()
            discrepancy = (
                difference.square() * precision
            ).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
            cosine = F.cosine_similarity(
                observed_float.flatten(1),
                predicted_float.flatten(1),
                dim=1,
                eps=1e-6,
            ).unsqueeze(1)
            consistency = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
            uncertainty_mean = uncertainty_float.mean(
                dim=(1, 2, 3), keepdim=False
            ).unsqueeze(1)
            quality_features = cross_modal_sensor_grid_quality_features(
                predicted_float.detach(),
                observed_float,
                availability.float(),
            )
        else:
            discrepancy = reference.new_zeros(batch_size, 1).float()
            consistency = reference.new_ones(batch_size, 1).float()
            uncertainty_mean = torch.nan_to_num(
                uncertainty.float(),
                nan=0.0,
                posinf=1e6,
                neginf=0.0,
            ).mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
            quality_features = reference.new_zeros(
                batch_size,
                SENSOR_CROSS_MODAL_QUALITY_DIM,
            ).float()
        reliability_features = torch.cat(
            [
                torch.log1p(discrepancy.clamp(max=1e6)),
                consistency,
                torch.log1p(uncertainty_mean.clamp(max=1e6)),
                availability,
                quality_features,
            ],
            dim=1,
        ).to(device=reference.device, dtype=reference.dtype)
        reliability_logits = self.sensor_reliability_predictor(
            reliability_features
        )
        raw_reliability = torch.sigmoid(reliability_logits)
        reliability = raw_reliability * availability
        effective_sensor = (
            predicted_sensor
            if not observation_present
            else predicted_sensor
            + reliability.view(batch_size, 1, 1, 1)
            * (observed - predicted_sensor)
        )

        text_bank = F.normalize(
            self.text_projection(self.text_prototypes), dim=-1
        )
        sensor_base = self.sensor_grid_encoder(effective_sensor)
        fused: list[Tensor] = []
        entropy_values: list[Tensor] = []
        for index, feature in enumerate(features):
            visual_semantic = F.normalize(
                self.visual_semantic[index](feature), dim=1
            )
            logits = torch.einsum(
                "bdhw,cd->bchw", visual_semantic, text_bank
            ) / self.temperature
            attention = logits.softmax(dim=1)
            text_context = torch.einsum(
                "bchw,cd->bdhw", attention, text_bank
            )
            text_residual = self.text_output[index](text_context)
            sensor_residual = self.sensor_output[index](
                F.interpolate(
                    sensor_base,
                    size=feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            fused.append(
                feature
                + torch.tanh(self.text_scale[index]) * text_residual
                + torch.tanh(self.sensor_scale[index])
                * reliability.view(batch_size, 1, 1, 1)
                * sensor_residual
            )
            entropy_values.append(
                -(attention.clamp_min(1e-8).log() * attention)
                .sum(dim=1)
                .mean(dim=(1, 2))
            )
        self.last_aux = {
            "predicted_sensor_grid": predicted_sensor,
            "effective_sensor_grid": effective_sensor,
            "observed_sensor_grid": observed,
            "sensor_uncertainty": uncertainty,
            "sensor_reliability": reliability,
            "sensor_raw_reliability": raw_reliability,
            "sensor_reliability_logits": reliability_logits,
            "sensor_available": availability,
            "sensor_consistency": consistency,
            "sensor_discrepancy": discrepancy,
            "sensor_cross_modal_quality_features": quality_features,
            "external_sensor_used": availability,
            "text_attention_entropy": torch.stack(
                entropy_values, dim=1
            ).mean(dim=1, keepdim=True),
            "text_enabled": reference.new_ones(batch_size, 1),
            "sensor_enabled": reference.new_ones(batch_size, 1),
        }
        return fused


class ReliableExternalSensorJointSegmentationModel(
    JointMultimodalSegmentationModel
):
    """Joint Segment model with a cross-modal sensor reliability gate."""

    def __init__(
        self,
        cfg: dict[str, Any] | str,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        text_prototypes: Tensor,
        semantic_dim: int = 64,
        sensor_hidden_dim: int = 256,
        text_temperature: float = 0.10,
        sensor_reconstruction_weight: float = 0.25,
        sensor_calibration_weight: float = 0.05,
        reliability_calibration_weight: float = 0.30,
        reliability_hidden_dim: int = 32,
        reliability_init: float = 0.85,
    ) -> None:
        super().__init__(
            cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            text_prototypes=text_prototypes,
            semantic_dim=semantic_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            text_temperature=text_temperature,
            sensor_reconstruction_weight=sensor_reconstruction_weight,
            sensor_calibration_weight=sensor_calibration_weight,
            fusion_mode="full",
        )
        channels = self.joint_fusion.channels
        self.joint_fusion = ReliabilityAwareSensorTextFeatureFusion(
            channels,
            text_prototypes,
            semantic_dim=semantic_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            temperature=text_temperature,
            reliability_hidden_dim=reliability_hidden_dim,
            reliability_init=reliability_init,
        )
        self.reliability_calibration_weight = float(
            reliability_calibration_weight
        )
        if self.reliability_calibration_weight < 0:
            raise ValueError(
                "reliability_calibration_weight must be non-negative"
            )

    def predict(
        self,
        x: Tensor,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed: list[int] | None = None,
        *,
        sensor_grid: Tensor | None = None,
        sensor_available: Tensor | None = None,
    ):
        rgb, packed_sensor, packed_availability = (
            split_reliable_sensor_channels(x)
        )
        if not hasattr(self, "joint_fusion"):
            return super().predict(
                rgb,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
                sensor_grid=None,
            )
        if sensor_grid is not None and packed_sensor is not None:
            raise ValueError(
                "sensor_grid cannot be supplied with packed sensor channels"
            )
        effective_sensor = (
            sensor_grid if sensor_grid is not None else packed_sensor
        )
        effective_availability = (
            sensor_available
            if sensor_available is not None
            else packed_availability
        )
        previous = self.joint_fusion._transport_availability
        self.joint_fusion._transport_availability = effective_availability
        try:
            return super().predict(
                rgb,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
                sensor_grid=effective_sensor,
            )
        finally:
            self.joint_fusion._transport_availability = previous

    def loss(self, batch: dict[str, Any], preds=None):
        total, items = super().loss(batch, preds)
        quality = batch.get("sensor_quality_target")
        if quality is None or self.reliability_calibration_weight == 0.0:
            return total, items
        quality = quality.to(
            device=batch["img"].device,
            dtype=batch["img"].dtype,
        )
        if quality.ndim == 1:
            quality = quality.unsqueeze(1)
        reliability_logits = self.joint_fusion.last_aux[
            "sensor_reliability_logits"
        ]
        if quality.shape != reliability_logits.shape:
            raise ValueError(
                "sensor quality target and reliability shape differ"
            )
        reliability_loss = F.binary_cross_entropy_with_logits(
            reliability_logits.float(),
            quality.float().clamp(0.0, 1.0),
        )
        weighted = self.reliability_calibration_weight * reliability_loss
        batch_size = int(batch["img"].shape[0])
        total = total.clone()
        items = items.clone()
        total[-1] = total[-1] + weighted * batch_size
        items[-1] = items[-1] + weighted.detach()
        return total, items


class ReliableExternalSensorSegmentationValidator(SegmentationValidator):
    """Validator for clean or deterministically degraded reliable sidecars."""

    def __init__(
        self,
        *args: Any,
        sensor_observations: Mapping[str, Mapping[str, Any]],
        degradation: Mapping[str, Any] | None = None,
        degradation_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.sensor_observation_index = dict(sensor_observations)
        self.sensor_degradation = dict(degradation or {"enabled": False})
        self.degradation_seed = int(degradation_seed)
        self.external_sensor_batch_count = 0
        self.external_sensor_sample_count = 0
        super().__init__(*args, **kwargs)

    def build_dataset(
        self,
        img_path: str,
        mode: str = "val",
        batch: int | None = None,
    ):
        return ReliableExternalSceneSensorDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=self.stride,
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
            sensor_observations=self.sensor_observation_index,
            degradation=self.sensor_degradation,
            degradation_seed=self.degradation_seed,
            degrade=bool(self.sensor_degradation.get("enabled", False)),
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        images = batch.get("img")
        if not isinstance(images, Tensor) or images.shape[1] != 5:
            raise ValueError(
                "reliable external validation requires five packed channels"
            )
        self.external_sensor_batch_count += 1
        self.external_sensor_sample_count += int(images.shape[0])
        return super().preprocess(batch)


def build_reliable_external_sensor_dataset(
    *,
    trainer: Any,
    img_path: str,
    mode: str,
    batch: int | None,
    sensor_observations: Mapping[str, Mapping[str, Any]],
    degradation: Mapping[str, Any],
    degradation_seed: int,
) -> ReliableExternalSceneSensorDataset:
    """Build the training/validation dataset for the reliable v3 head."""

    stride = max(int(unwrap_model(trainer.model).stride.max()), 32)
    return ReliableExternalSceneSensorDataset(
        img_path=img_path,
        imgsz=trainer.args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=trainer.args,
        rect=trainer.args.rect or mode == "val",
        cache=trainer.args.cache or None,
        single_cls=trainer.args.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=trainer.args.task,
        classes=trainer.args.classes,
        data=trainer.data,
        fraction=(trainer.args.fraction if mode == "train" else 1.0),
        sensor_observations=sensor_observations,
        degradation=degradation,
        degradation_seed=degradation_seed,
        degrade=mode == "train",
    )
