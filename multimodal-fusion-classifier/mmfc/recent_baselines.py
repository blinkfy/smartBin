"""Same-input adaptations of established multimodal fusion baselines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import (
    CLIPPrototypeEncoder,
    ModalityAdapter,
    PROJECT_ROOT,
    SensorEncoder,
    YOLOMultiScaleEncoder,
)


BASELINE_METHODS = ("qmf", "tmc", "concat_mlp", "gmu", "lmf")
BRANCH_NAMES = ("visual", "visual_text", "sensor")


def dirichlet_kl(alpha: Tensor) -> Tensor:
    """KL(Dir(alpha) || Dir(1)) for each sample."""

    if alpha.ndim != 2:
        raise ValueError("alpha must have shape [B,C]")
    classes = alpha.shape[1]
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)
    log_beta_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(
        dim=1, keepdim=True
    )
    log_beta_uniform = torch.lgamma(beta).sum(
        dim=1, keepdim=True
    ) - torch.lgamma(sum_beta)
    digamma_delta = torch.digamma(alpha) - torch.digamma(sum_alpha)
    value = ((alpha - beta) * digamma_delta).sum(
        dim=1, keepdim=True
    )
    value = value + log_beta_alpha + log_beta_uniform
    if classes < 2 or not torch.isfinite(value).all():
        raise ValueError("invalid Dirichlet KL result")
    return value.squeeze(1)


def tmc_evidential_loss(
    labels: Tensor,
    alpha: Tensor,
    *,
    epoch: int,
    annealing_epochs: int,
) -> Tensor:
    """Expected CE plus annealed evidence KL from the TMC implementation."""

    if labels.ndim != 1 or alpha.ndim != 2:
        raise ValueError("labels/alpha must have shapes [B] and [B,C]")
    if labels.shape[0] != alpha.shape[0]:
        raise ValueError("labels and alpha batch sizes differ")
    classes = alpha.shape[1]
    one_hot = F.one_hot(labels, num_classes=classes).to(dtype=alpha.dtype)
    strength = alpha.sum(dim=1, keepdim=True)
    expected_ce = (
        one_hot * (torch.digamma(strength) - torch.digamma(alpha))
    ).sum(dim=1)
    evidence = alpha - 1.0
    non_target_alpha = evidence * (1.0 - one_hot) + 1.0
    annealing = min(1.0, max(0.0, float(epoch) / max(1, annealing_epochs)))
    return expected_ce + annealing * dirichlet_kl(non_target_alpha)


def dempster_shafer_combine(alpha_a: Tensor, alpha_b: Tensor) -> Tensor:
    """Combine two Dirichlet opinions using the TMC DS rule."""

    if alpha_a.shape != alpha_b.shape or alpha_a.ndim != 2:
        raise ValueError("Dirichlet opinions must share shape [B,C]")
    classes = alpha_a.shape[1]
    if classes < 2:
        raise ValueError("at least two classes are required")
    strength_a = alpha_a.sum(dim=1, keepdim=True)
    strength_b = alpha_b.sum(dim=1, keepdim=True)
    evidence_a = alpha_a - 1.0
    evidence_b = alpha_b - 1.0
    belief_a = evidence_a / strength_a.clamp_min(1e-8)
    belief_b = evidence_b / strength_b.clamp_min(1e-8)
    uncertainty_a = classes / strength_a.clamp_min(1e-8)
    uncertainty_b = classes / strength_b.clamp_min(1e-8)

    pairwise = torch.bmm(belief_a.unsqueeze(2), belief_b.unsqueeze(1))
    conflict = pairwise.sum(dim=(1, 2)) - pairwise.diagonal(
        dim1=1, dim2=2
    ).sum(dim=1)
    denominator = (1.0 - conflict).clamp_min(1e-6).unsqueeze(1)
    combined_belief = (
        belief_a * belief_b
        + belief_a * uncertainty_b
        + belief_b * uncertainty_a
    ) / denominator
    combined_uncertainty = (
        uncertainty_a * uncertainty_b
    ) / denominator
    combined_strength = classes / combined_uncertainty.clamp_min(1e-8)
    combined_alpha = combined_belief * combined_strength + 1.0
    if not torch.isfinite(combined_alpha).all():
        raise FloatingPointError("non-finite DS-combined Dirichlet opinion")
    return combined_alpha


class RecentFusionBaseline(nn.Module):
    """Matched fusion baselines on the project's three input branches.

    The text side is a candidate-prototype branch queried only by visual
    features. It is therefore named ``visual_text`` rather than pretending
    that per-object free text is independently observed.
    """

    def __init__(
        self,
        num_classes: int,
        class_descriptions: Sequence[str] | None = None,
        *,
        method: str,
        visual_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        sensor_encoder: nn.Module | None = None,
        feature_dim: int = 512,
        dropout: float = 0.3,
        scale_prior: Sequence[float] = (0.3, 0.5, 0.2),
        text_temperature: float = 0.07,
        qmf_energy_temperature: float = 10.0,
        fusion_hidden_dim: int = 512,
        lmf_rank: int = 4,
    ) -> None:
        super().__init__()
        method = str(method).strip().lower()
        if method not in BASELINE_METHODS:
            raise ValueError(f"method must be one of {list(BASELINE_METHODS)}")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if feature_dim != 512:
            raise ValueError("the matched baseline requires feature_dim=512")
        if len(scale_prior) != 3 or any(float(value) <= 0 for value in scale_prior):
            raise ValueError("scale_prior must contain three positive values")
        if text_temperature <= 0.0 or qmf_energy_temperature <= 0.0:
            raise ValueError("temperatures must be positive")
        if fusion_hidden_dim <= 0 or lmf_rank <= 0:
            raise ValueError("fusion_hidden_dim and lmf_rank must be positive")

        self.method = method
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.class_descriptions = (
            tuple(str(value) for value in class_descriptions)
            if class_descriptions is not None
            else None
        )
        if (
            self.class_descriptions is not None
            and len(self.class_descriptions) != self.num_classes
        ):
            raise ValueError("class description count must match num_classes")
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.sensor_encoder = sensor_encoder
        self.text_temperature = float(text_temperature)
        self.qmf_energy_temperature = float(qmf_energy_temperature)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.lmf_rank = int(lmf_rank)

        self.scale_projections = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(3)]
        )
        for projection in self.scale_projections:
            nn.init.eye_(projection.weight)
            nn.init.zeros_(projection.bias)
        prior = torch.as_tensor(scale_prior, dtype=torch.float32)
        self.scale_prior_logits = nn.Parameter((prior / prior.sum()).log())

        self.branch_adapters = nn.ModuleDict(
            {
                name: ModalityAdapter(feature_dim, dropout)
                for name in BRANCH_NAMES
            }
        )
        self.branch_heads = nn.ModuleDict()
        if method in {"qmf", "tmc"}:
            self.branch_heads.update(
                {
                    name: nn.Linear(feature_dim, num_classes)
                    for name in BRANCH_NAMES
                }
            )
        for head in self.branch_heads.values():
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)

        self.concat_head: nn.Module | None = None
        self.gmu_candidates: nn.ModuleDict | None = None
        self.gmu_gate: nn.Linear | None = None
        self.gmu_head: nn.Linear | None = None
        self.lmf_factors: nn.ParameterList | None = None
        self.lmf_bias: nn.Parameter | None = None
        if method == "concat_mlp":
            self.concat_head = nn.Sequential(
                nn.Linear(len(BRANCH_NAMES) * feature_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, num_classes),
            )
        elif method == "gmu":
            self.gmu_candidates = nn.ModuleDict(
                {
                    name: nn.Linear(feature_dim, fusion_hidden_dim)
                    for name in BRANCH_NAMES
                }
            )
            self.gmu_gate = nn.Linear(
                len(BRANCH_NAMES) * feature_dim,
                len(BRANCH_NAMES) * fusion_hidden_dim,
            )
            self.gmu_head = nn.Linear(fusion_hidden_dim, num_classes)
        elif method == "lmf":
            self.lmf_factors = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(lmf_rank, feature_dim + 1, num_classes)
                    )
                    for _ in BRANCH_NAMES
                ]
            )
            self.lmf_bias = nn.Parameter(torch.zeros(num_classes))

        for module in (self.concat_head, self.gmu_candidates):
            if module is None:
                continue
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for layer in (self.gmu_gate, self.gmu_head):
            if layer is not None:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        if self.lmf_factors is not None:
            for factor in self.lmf_factors:
                nn.init.xavier_normal_(factor)
        self.class_names: tuple[str, ...] = tuple()

    @classmethod
    def create_production(
        cls,
        num_classes: int,
        class_descriptions: Sequence[str],
        *,
        method: str,
        yolo_weights: str | Path,
        clip_model: str,
        clip_local_files_only: bool,
        freeze_yolo_layers: int,
        freeze_clip_text_layers: int,
        sensor_standardize: bool = True,
        **kwargs: Any,
    ) -> "RecentFusionBaseline":
        visual = YOLOMultiScaleEncoder(
            yolo_weights,
            output_dim=512,
            freeze_layers=freeze_yolo_layers,
        )
        text = CLIPPrototypeEncoder(
            clip_model,
            freeze_layers=freeze_clip_text_layers,
            local_files_only=clip_local_files_only,
        )
        sensor = SensorEncoder(output_dim=512, normalize=sensor_standardize)
        return cls(
            num_classes,
            class_descriptions,
            method=method,
            visual_encoder=visual,
            text_encoder=text,
            sensor_encoder=sensor,
            **kwargs,
        )

    @staticmethod
    def _validate_feature(name: str, value: Tensor, batch_size: int) -> None:
        if tuple(value.shape) != (batch_size, 512):
            raise ValueError(f"{name} must have shape [B,512]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    def _branch_features(
        self,
        visual_scales: Sequence[Tensor],
        sensor_features: Tensor,
        text_prototypes: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        if len(visual_scales) != 3:
            raise ValueError("visual_scales must contain exactly three tensors")
        batch_size = int(visual_scales[0].shape[0])
        for index, value in enumerate(visual_scales):
            self._validate_feature(f"visual_scales[{index}]", value, batch_size)
        self._validate_feature("sensor_features", sensor_features, batch_size)
        if tuple(text_prototypes.shape) != (self.num_classes, 512):
            raise ValueError(
                f"text_prototypes must have shape [{self.num_classes},512]"
            )

        projected = [
            layer(value)
            for layer, value in zip(self.scale_projections, visual_scales)
        ]
        scale_weights = torch.softmax(self.scale_prior_logits, dim=0)
        visual = sum(
            weight * value
            for weight, value in zip(scale_weights, projected)
        )
        query = F.normalize(visual, dim=-1)
        prototype_bank = F.normalize(text_prototypes, dim=-1)
        prototype_scores = query @ prototype_bank.transpose(0, 1)
        prototype_weights = torch.softmax(
            prototype_scores / self.text_temperature,
            dim=-1,
        )
        visual_text = prototype_weights @ text_prototypes
        return {
            "visual": visual,
            "visual_text": visual_text,
            "sensor": sensor_features,
        }, prototype_weights

    def forward_from_features(
        self,
        visual_scales: Sequence[Tensor],
        sensor_features: Tensor,
        text_prototypes: Tensor,
        *,
        sensor_available: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        branch_features, prototype_weights = self._branch_features(
            visual_scales,
            sensor_features,
            text_prototypes,
        )
        adapted = {
            name: self.branch_adapters[name](branch_features[name])
            for name in BRANCH_NAMES
        }
        feature_reference = next(iter(adapted.values()))
        batch_size = feature_reference.shape[0]
        if sensor_available is None:
            availability = feature_reference.new_ones(batch_size, 1)
        else:
            availability = torch.as_tensor(
                sensor_available,
                device=feature_reference.device,
                dtype=feature_reference.dtype,
            )
            if availability.ndim == 1:
                availability = availability.unsqueeze(1)
            if tuple(availability.shape) != (batch_size, 1):
                raise ValueError("sensor_available must have shape [B,1]")
            availability = availability.clamp(0.0, 1.0)

        fused_alpha = feature_reference.new_empty(0)
        if self.method in {"qmf", "tmc"}:
            branch_logits = torch.stack(
                [self.branch_heads[name](adapted[name]) for name in BRANCH_NAMES],
                dim=1,
            )
        else:
            branch_logits = feature_reference.new_zeros(
                batch_size,
                len(BRANCH_NAMES),
                self.num_classes,
            )

        if self.method == "qmf":
            branch_confidence = torch.logsumexp(
                branch_logits, dim=-1
            ) / self.qmf_energy_temperature
            logits = (
                branch_logits * branch_confidence.detach().unsqueeze(-1)
            ).sum(dim=1)
            modality_weights = torch.softmax(branch_confidence, dim=1)
        elif self.method == "tmc":
            branch_alpha = F.softplus(branch_logits) + 1.0
            fused_alpha = branch_alpha[:, 0]
            for index in range(1, len(BRANCH_NAMES)):
                fused_alpha = dempster_shafer_combine(
                    fused_alpha,
                    branch_alpha[:, index],
                )
            probabilities = fused_alpha / fused_alpha.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            logits = probabilities.clamp_min(1e-12).log()
            branch_confidence = 1.0 - (
                self.num_classes
                / branch_alpha.sum(dim=-1).clamp_min(1e-8)
            )
            modality_weights = branch_confidence / branch_confidence.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        elif self.method == "concat_mlp":
            if self.concat_head is None:
                raise RuntimeError("concat_mlp head is missing")
            logits = self.concat_head(
                torch.cat([adapted[name] for name in BRANCH_NAMES], dim=-1)
            )
            modality_weights = logits.new_full(
                (batch_size, len(BRANCH_NAMES)),
                1.0 / len(BRANCH_NAMES),
            )
            branch_confidence = modality_weights
        elif self.method == "gmu":
            if (
                self.gmu_candidates is None
                or self.gmu_gate is None
                or self.gmu_head is None
            ):
                raise RuntimeError("GMU layers are missing")
            candidates = torch.stack(
                [
                    torch.tanh(self.gmu_candidates[name](adapted[name]))
                    for name in BRANCH_NAMES
                ],
                dim=1,
            )
            gate_logits = self.gmu_gate(
                torch.cat([adapted[name] for name in BRANCH_NAMES], dim=-1)
            ).reshape(batch_size, len(BRANCH_NAMES), self.fusion_hidden_dim)
            gates = torch.softmax(gate_logits, dim=1)
            logits = self.gmu_head((gates * candidates).sum(dim=1))
            modality_weights = gates.mean(dim=-1)
            branch_confidence = modality_weights
        else:
            if self.lmf_factors is None or self.lmf_bias is None:
                raise RuntimeError("LMF factors are missing")
            projections = []
            for name, factor in zip(BRANCH_NAMES, self.lmf_factors):
                ones = adapted[name].new_ones(batch_size, 1)
                augmented = torch.cat((ones, adapted[name]), dim=1)
                projections.append(
                    torch.einsum("bd,rdc->brc", augmented, factor)
                )
            interaction = projections[0]
            for projection in projections[1:]:
                interaction = interaction * projection
            logits = interaction.sum(dim=1) + self.lmf_bias
            modality_weights = logits.new_full(
                (batch_size, len(BRANCH_NAMES)),
                1.0 / len(BRANCH_NAMES),
            )
            branch_confidence = modality_weights

        sensor_score = modality_weights[:, 2:3]
        aux = {
            "branch_logits": branch_logits,
            "branch_confidence": branch_confidence,
            "modality_weights": modality_weights,
            "prototype_weights": prototype_weights,
            "sensor_available": availability,
            "sensor_reliability": sensor_score,
            "sensor_consistency": sensor_score,
            "sensor_discrepancy": 1.0 - sensor_score,
            "sensor_uncertainty": 1.0 - sensor_score,
            "fused_alpha": fused_alpha,
        }
        if self.method == "tmc":
            aux["branch_alpha"] = F.softplus(branch_logits) + 1.0
        return (logits, aux) if return_aux else logits

    def forward(
        self,
        images: Tensor,
        sensor_grid: Tensor,
        *,
        class_descriptions: Sequence[str] | None = None,
        sensor_available: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if (
            self.visual_encoder is None
            or self.text_encoder is None
            or self.sensor_encoder is None
        ):
            raise RuntimeError(
                "production encoders are missing; use forward_from_features"
            )
        descriptions = (
            tuple(class_descriptions)
            if class_descriptions is not None
            else self.class_descriptions
        )
        if descriptions is None or len(descriptions) != self.num_classes:
            raise ValueError("all candidate class descriptions are required")
        visual_scales = self.visual_encoder(images)
        sensor_features = self.sensor_encoder(sensor_grid)
        text_prototypes = self.text_encoder(descriptions, images.device)
        return self.forward_from_features(
            visual_scales,
            sensor_features,
            text_prototypes,
            sensor_available=sensor_available,
            return_aux=return_aux,
        )

    def active_parameters(self):
        return (parameter for parameter in self.parameters())


def build_recent_baseline_model(
    config: Mapping[str, Any],
) -> RecentFusionBaseline:
    classes = config.get("classes", [])
    if not classes:
        raise ValueError("config classes must not be empty")
    class_names = [
        value["name"] if isinstance(value, Mapping) else str(value)
        for value in classes
    ]
    descriptions = [
        value.get("description", value["name"])
        if isinstance(value, Mapping)
        else str(value)
        for value in classes
    ]
    baseline = config.get("recent_baseline", {})
    if not isinstance(baseline, Mapping):
        raise ValueError("recent_baseline must be an object")
    method = str(baseline.get("method", "")).strip().lower()
    if method not in BASELINE_METHODS:
        raise ValueError(
            f"recent_baseline.method must be one of {list(BASELINE_METHODS)}"
        )
    model_config = config.get("model", {})
    weights = Path(model_config.get("yolo_weights", "../yolo11s-seg.pt"))
    if not weights.is_absolute():
        weights = (PROJECT_ROOT / weights).resolve()
    model = RecentFusionBaseline.create_production(
        len(class_names),
        descriptions,
        method=method,
        yolo_weights=weights,
        clip_model=str(
            model_config.get(
                "clip_model",
                "openai/clip-vit-base-patch16",
            )
        ),
        clip_local_files_only=bool(
            model_config.get("clip_local_files_only", False)
        ),
        freeze_yolo_layers=int(model_config.get("freeze_yolo_layers", 5)),
        freeze_clip_text_layers=int(
            model_config.get("freeze_clip_text_layers", 3)
        ),
        sensor_standardize=bool(
            model_config.get("sensor_standardize", True)
        ),
        feature_dim=int(model_config.get("feature_dim", 512)),
        dropout=float(model_config.get("dropout", 0.3)),
        scale_prior=model_config.get("scale_prior", [0.3, 0.5, 0.2]),
        text_temperature=float(model_config.get("text_temperature", 0.07)),
        qmf_energy_temperature=float(
            baseline.get("energy_temperature", 10.0)
        ),
        fusion_hidden_dim=int(baseline.get("fusion_hidden_dim", 512)),
        lmf_rank=int(baseline.get("rank", 4)),
    )
    model.class_names = tuple(class_names)
    return model
