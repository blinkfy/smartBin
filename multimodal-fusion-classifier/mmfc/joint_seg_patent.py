"""Patent-aligned multimodal fusion stack for the native Segment head."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .joint_seg_spatial import (
    SpatialReliabilityAwareSensorTextFeatureFusion,
    SpatialReliableExternalSensorJointSegmentationModel,
)
from .model import ModalityAdapter


class PatentAlignedReliableFeatureFusion(
    SpatialReliabilityAwareSensorTextFeatureFusion
):
    """Apply Adapter, FiLM, cosine weighting, interaction, and scale fusion."""

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
        patent_feature_dim: int = 512,
        patent_attention_dim: int = 128,
        patent_attention_heads: int = 8,
        patent_dropout: float = 0.30,
        scale_prior: Sequence[float] = (0.3, 0.5, 0.2),
    ) -> None:
        super().__init__(
            channels,
            text_prototypes,
            semantic_dim=semantic_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            temperature=temperature,
            reliability_hidden_dim=reliability_hidden_dim,
            reliability_init=reliability_init,
            spatial_hidden_dim=spatial_hidden_dim,
        )
        if patent_feature_dim != 512:
            raise ValueError("the patent-aligned feature dimension must be 512")
        if (
            patent_attention_dim <= 0
            or patent_attention_dim % patent_attention_heads
        ):
            raise ValueError("invalid patent attention dimensions")
        if len(scale_prior) != 3 or any(value <= 0 for value in scale_prior):
            raise ValueError("scale_prior must contain three positive values")

        self.patent_feature_dim = int(patent_feature_dim)
        self.patent_attention_dim = int(patent_attention_dim)
        self.patent_attention_heads = int(patent_attention_heads)
        self.patent_dropout = float(patent_dropout)
        self.patent_scale_projections = nn.ModuleList(
            nn.Linear(channel, patent_feature_dim)
            for channel in self.channels
        )
        for projection in self.patent_scale_projections:
            nn.init.xavier_normal_(projection.weight)
            nn.init.zeros_(projection.bias)

        self.patent_sensor_projection = nn.Conv2d(
            1, patent_feature_dim, kernel_size=1
        )
        nn.init.kaiming_normal_(
            self.patent_sensor_projection.weight,
            a=0.1,
            mode="fan_out",
            nonlinearity="leaky_relu",
        )
        nn.init.zeros_(self.patent_sensor_projection.bias)
        self.patent_sensor_activation = nn.LeakyReLU(negative_slope=0.1)

        self.patent_adapters = nn.ModuleList(
            ModalityAdapter(patent_feature_dim, patent_dropout)
            for _ in range(3)
        )
        self.patent_film_gamma = nn.Parameter(
            torch.ones(3, patent_feature_dim)
        )
        self.patent_film_beta = nn.Parameter(
            torch.full((3, patent_feature_dim), 0.1)
        )
        self.patent_down_projection = nn.Linear(
            patent_feature_dim, patent_attention_dim
        )
        self.patent_interaction = nn.MultiheadAttention(
            patent_attention_dim,
            patent_attention_heads,
            dropout=patent_dropout,
            batch_first=True,
        )
        self.patent_up_projection = nn.Linear(
            patent_attention_dim, patent_feature_dim
        )
        for projection in (
            self.patent_down_projection,
            self.patent_up_projection,
        ):
            nn.init.xavier_normal_(projection.weight)
            nn.init.zeros_(projection.bias)

        prior = torch.as_tensor(scale_prior, dtype=torch.float32)
        prior = prior / prior.sum()
        self.patent_scale_prior_logits = nn.Parameter(prior.log())
        self.patent_dynamic_scale_gate = nn.Linear(
            patent_feature_dim, 3
        )
        nn.init.zeros_(self.patent_dynamic_scale_gate.weight)
        nn.init.zeros_(self.patent_dynamic_scale_gate.bias)
        self.patent_output_norm = nn.LayerNorm(patent_feature_dim)
        self.patent_scale_modulators = nn.ModuleList(
            nn.Linear(patent_feature_dim, 2 * channel)
            for channel in self.channels
        )
        for modulator in self.patent_scale_modulators:
            nn.init.zeros_(modulator.weight)
            nn.init.zeros_(modulator.bias)

    @staticmethod
    def _modality_weights(tokens: Tensor) -> Tensor:
        normalized = F.normalize(tokens.float(), dim=-1)
        pairwise = normalized @ normalized.transpose(1, 2)
        self_similarity = pairwise.diagonal(dim1=1, dim2=2)
        scores = (pairwise.sum(dim=-1) - self_similarity) / 2.0
        return torch.softmax(scores, dim=-1).to(dtype=tokens.dtype)

    def forward(
        self,
        features: Sequence[Tensor],
        observed_sensor: Tensor | None = None,
        sensor_available: Tensor | None = None,
    ) -> list[Tensor]:
        if len(features) != 3:
            raise ValueError("expected exactly three YOLO feature maps")

        # Reuse the v5 spatial predictor and reliability path. Its feature-map
        # residual output is intentionally discarded; v6 applies the full
        # patent-aligned stack to the original P3/P4/P5 tensors below.
        _ = super().forward(
            features,
            observed_sensor=observed_sensor,
            sensor_available=sensor_available,
        )
        reliability_aux = dict(self.last_aux)
        effective_sensor = reliability_aux["effective_sensor_grid"]
        batch_size = int(features[0].shape[0])

        scale_tokens = torch.stack(
            [
                projection(
                    F.adaptive_avg_pool2d(feature, 1).flatten(1)
                )
                for feature, projection in zip(
                    features, self.patent_scale_projections
                )
            ],
            dim=1,
        )
        prior = torch.softmax(
            self.patent_scale_prior_logits, dim=0
        ).to(device=scale_tokens.device, dtype=scale_tokens.dtype)
        visual_token = (
            scale_tokens * prior.view(1, 3, 1)
        ).sum(dim=1)

        sensor_16 = F.interpolate(
            effective_sensor,
            size=(16, 16),
            mode="bilinear",
            align_corners=False,
        )
        sensor_token = F.adaptive_avg_pool2d(
            self.patent_sensor_activation(
                self.patent_sensor_projection(sensor_16)
            ),
            output_size=1,
        ).flatten(1)

        prototype_bank = F.normalize(
            self.text_prototypes.to(
                device=visual_token.device,
                dtype=visual_token.dtype,
            ),
            dim=-1,
        )
        text_query = F.normalize(visual_token + sensor_token, dim=-1)
        text_logits = (
            text_query @ prototype_bank.transpose(0, 1)
        ) / self.temperature
        text_weights = text_logits.softmax(dim=-1)
        text_token = text_weights @ prototype_bank

        raw_tokens = (visual_token, text_token, sensor_token)
        adapted = torch.stack(
            [
                adapter(token)
                for adapter, token in zip(
                    self.patent_adapters, raw_tokens
                )
            ],
            dim=1,
        )
        film_tokens = (
            adapted * self.patent_film_gamma.unsqueeze(0)
            + self.patent_film_beta.unsqueeze(0)
        )
        modality_weights = self._modality_weights(film_tokens)
        compressed = self.patent_down_projection(film_tokens)
        interacted, interaction_weights = self.patent_interaction(
            compressed,
            compressed,
            compressed,
            need_weights=True,
            average_attn_weights=False,
        )
        restored = self.patent_up_projection(interacted)
        residual_tokens = film_tokens + restored
        fused_token = (
            residual_tokens * modality_weights.unsqueeze(-1)
        ).sum(dim=1)

        scale_weights = torch.softmax(
            self.patent_scale_prior_logits.unsqueeze(0)
            + self.patent_dynamic_scale_gate(fused_token),
            dim=-1,
        )
        refined_visual = (
            scale_tokens * scale_weights.unsqueeze(-1)
        ).sum(dim=1)
        final_token = self.patent_output_norm(
            fused_token + refined_visual
        )

        fused_features: list[Tensor] = []
        for feature, modulator in zip(
            features, self.patent_scale_modulators
        ):
            gamma, beta = modulator(final_token).chunk(2, dim=-1)
            gamma = torch.tanh(gamma).view(
                batch_size, feature.shape[1], 1, 1
            )
            beta = beta.view(batch_size, feature.shape[1], 1, 1)
            fused_features.append(feature * (1.0 + gamma) + beta)

        text_entropy = -(
            text_weights.clamp_min(1e-8).log() * text_weights
        ).sum(dim=-1, keepdim=True)
        self.last_aux = {
            **reliability_aux,
            "text_attention_entropy": text_entropy,
            "patent_modality_weights": modality_weights,
            "patent_scale_weights": scale_weights,
            "patent_interaction_weights": interaction_weights,
            "patent_final_fusion_feature": final_token,
            "patent_sensor_grid_16x16": sensor_16,
            "patent_stack_enabled": final_token.new_ones(
                batch_size, 1
            ),
        }
        return fused_features


class PatentAlignedReliableJointSegmentationModel(
    SpatialReliableExternalSensorJointSegmentationModel
):
    """Native Segment model conditioned by the complete patent-style stack."""

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
        patent_feature_dim: int = 512,
        patent_attention_dim: int = 128,
        patent_attention_heads: int = 8,
        patent_dropout: float = 0.30,
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
            spatial_hidden_dim=spatial_hidden_dim,
        )
        channels = self.joint_fusion.channels
        self.joint_fusion = PatentAlignedReliableFeatureFusion(
            channels,
            text_prototypes,
            semantic_dim=semantic_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            temperature=text_temperature,
            reliability_hidden_dim=reliability_hidden_dim,
            reliability_init=reliability_init,
            spatial_hidden_dim=spatial_hidden_dim,
            patent_feature_dim=patent_feature_dim,
            patent_attention_dim=patent_attention_dim,
            patent_attention_heads=patent_attention_heads,
            patent_dropout=patent_dropout,
        )
