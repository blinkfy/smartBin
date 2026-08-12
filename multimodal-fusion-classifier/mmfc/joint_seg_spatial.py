"""Spatial RGB-to-sensor prediction for reliability-aware joint segmentation."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .joint_seg_reliable import (
    ReliableExternalSensorJointSegmentationModel,
    ReliabilityAwareSensorTextFeatureFusion,
)
from .model import (
    SENSOR_CROSS_MODAL_QUALITY_DIM,
    cross_modal_sensor_grid_quality_features,
)


class SpatialReliabilityAwareSensorTextFeatureFusion(
    ReliabilityAwareSensorTextFeatureFusion
):
    """Predict a 4x4 sensor grid without discarding feature-map layout."""

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
        spatial_hidden_dim: int = 64,
    ) -> None:
        super().__init__(
            channels,
            text_prototypes,
            semantic_dim=semantic_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            temperature=temperature,
            reliability_hidden_dim=reliability_hidden_dim,
            reliability_init=reliability_init,
        )
        if spatial_hidden_dim <= 0 or spatial_hidden_dim % 8:
            raise ValueError(
                "spatial_hidden_dim must be positive and divisible by 8"
            )
        self.sensor_spatial_predictor = nn.Sequential(
            nn.Conv2d(sum(self.channels), spatial_hidden_dim, kernel_size=1),
            nn.GroupNorm(8, spatial_hidden_dim),
            nn.SiLU(),
            nn.Conv2d(
                spatial_hidden_dim,
                spatial_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
        )
        self.sensor_spatial_mean = nn.Conv2d(
            spatial_hidden_dim, 1, kernel_size=1
        )
        self.sensor_spatial_uncertainty = nn.Conv2d(
            spatial_hidden_dim, 1, kernel_size=1
        )
        nn.init.zeros_(self.sensor_spatial_mean.bias)
        nn.init.zeros_(self.sensor_spatial_uncertainty.weight)
        nn.init.constant_(self.sensor_spatial_uncertainty.bias, -2.25)

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
                raise ValueError(
                    "YOLO feature map shape does not match fusion contract"
                )

        spatial_features = torch.cat(
            [
                F.adaptive_avg_pool2d(feature, output_size=(4, 4))
                for feature in features
            ],
            dim=1,
        )
        sensor_hidden = self.sensor_spatial_predictor(spatial_features)
        predicted_sensor = torch.sigmoid(
            self.sensor_spatial_mean(sensor_hidden)
        )
        uncertainty = (
            F.softplus(self.sensor_spatial_uncertainty(sensor_hidden)) + 1e-4
        )

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
                raise ValueError(
                    "observed_sensor must have shape [B,1,4,4]"
                )
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
            discrepancy = (
                difference.square() * uncertainty_float.reciprocal()
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


class SpatialReliableExternalSensorJointSegmentationModel(
    ReliableExternalSensorJointSegmentationModel
):
    """Reliable external-sensor model with spatial RGB sensor prediction."""

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
        reliability_calibration_weight: float = 0.50,
        reliability_hidden_dim: int = 32,
        reliability_init: float = 0.85,
        spatial_hidden_dim: int = 64,
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
            reliability_calibration_weight=reliability_calibration_weight,
            reliability_hidden_dim=reliability_hidden_dim,
            reliability_init=reliability_init,
        )
        channels = self.joint_fusion.channels
        self.joint_fusion = SpatialReliabilityAwareSensorTextFeatureFusion(
            channels,
            text_prototypes,
            semantic_dim=semantic_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            temperature=text_temperature,
            reliability_hidden_dim=reliability_hidden_dim,
            reliability_init=reliability_init,
            spatial_hidden_dim=spatial_hidden_dim,
        )
