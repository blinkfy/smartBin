"""多分类联合损失。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MulticlassFocalLoss(nn.Module):
    """标准多分类 focal loss，支持标量或逐类别 alpha。"""

    def __init__(
        self,
        alpha: float | Sequence[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma 必须非负")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction 必须为 none、mean 或 sum")
        alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
        if alpha_tensor.ndim > 1 or torch.any(alpha_tensor < 0):
            raise ValueError("alpha 必须是非负标量或一维类别权重")
        self.register_buffer("alpha", alpha_tensor)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
            raise ValueError("logits/targets 形状应分别为 [B,C] 和 [B]")
        log_probabilities = F.log_softmax(logits, dim=-1)
        log_pt = log_probabilities.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp()
        if self.alpha.ndim == 0:
            alpha_t = self.alpha.expand_as(pt)
        else:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError("逐类别 alpha 的长度必须等于类别数")
            alpha_t = self.alpha[targets]
        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class JointClassificationLoss(nn.Module):
    """Cross Entropy 与多分类 Focal Loss 的加权和。"""

    def __init__(
        self,
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        alpha: float | Sequence[float] = 0.25,
        gamma: float = 2.0,
        ce_class_weights: Sequence[float] | None = None,
        reconstruction_weight: float = 0.0,
        calibration_weight: float = 0.0,
        variance_prior_weight: float = 0.0,
        variance_prior_target: float = -2.0,
        calibration_clean_weight: float = 1.0,
        sensor_grid_reconstruction_weight: float = 0.0,
        calibration_ranking_weight: float = 0.0,
        calibration_ranking_margin: float = 0.05,
        visual_anchor_safeguard_weight: float = 0.0,
        semantic_alignment_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if ce_weight < 0 or focal_weight < 0 or ce_weight + focal_weight == 0:
            raise ValueError("损失权重必须非负，且至少一个大于 0")
        if (
            reconstruction_weight < 0
            or calibration_weight < 0
            or variance_prior_weight < 0
            or sensor_grid_reconstruction_weight < 0
            or calibration_ranking_weight < 0
            or visual_anchor_safeguard_weight < 0
            or semantic_alignment_weight < 0
        ):
            raise ValueError("auxiliary loss weights must be non-negative")
        if calibration_clean_weight < 1.0:
            raise ValueError("calibration_clean_weight must be at least 1")
        if not torch.isfinite(torch.tensor(float(variance_prior_target))):
            raise ValueError("variance_prior_target must be finite")
        if calibration_ranking_margin < 0.0:
            raise ValueError("calibration_ranking_margin must be non-negative")
        self.ce_weight = float(ce_weight)
        self.focal_weight = float(focal_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.calibration_weight = float(calibration_weight)
        self.variance_prior_weight = float(variance_prior_weight)
        self.variance_prior_target = float(variance_prior_target)
        self.calibration_clean_weight = float(calibration_clean_weight)
        self.sensor_grid_reconstruction_weight = float(
            sensor_grid_reconstruction_weight
        )
        self.calibration_ranking_weight = float(calibration_ranking_weight)
        self.calibration_ranking_margin = float(calibration_ranking_margin)
        self.visual_anchor_safeguard_weight = float(
            visual_anchor_safeguard_weight
        )
        self.semantic_alignment_weight = float(semantic_alignment_weight)
        class_weights = (
            None
            if ce_class_weights is None
            else torch.as_tensor(ce_class_weights, dtype=torch.float32)
        )
        if class_weights is not None and (
            class_weights.ndim != 1
            or class_weights.numel() == 0
            or not bool(torch.isfinite(class_weights).all())
            or bool((class_weights <= 0).any())
        ):
            raise ValueError(
                "ce_class_weights must be a finite positive one-dimensional sequence"
            )
        self.register_buffer("ce_class_weights", class_weights)
        self.focal = MulticlassFocalLoss(alpha=alpha, gamma=gamma)

    @staticmethod
    def _mask_mean(values: Tensor, mask: Tensor) -> Tensor:
        mask_bool = mask.to(device=values.device, dtype=torch.bool)
        if not bool(mask_bool.any()):
            return values.new_tensor(0.0)
        return values[mask_bool].mean()

    def _reconstruction_loss(
        self,
        aux: dict[str, Tensor] | None,
        sensor_target: Tensor | None,
        reconstruction_mask: Tensor | None,
        logits: Tensor,
    ) -> Tensor:
        if (
            self.reconstruction_weight == 0.0
            or aux is None
            or sensor_target is None
            or "predicted_sensor" not in aux
            or "sensor_log_variance" not in aux
        ):
            return logits.new_tensor(0.0)
        predicted = aux["predicted_sensor"]
        log_variance = aux["sensor_log_variance"].clamp(-12.0, 12.0)
        target = sensor_target.detach().to(device=predicted.device, dtype=predicted.dtype)
        if predicted.shape != target.shape or predicted.shape != log_variance.shape:
            raise ValueError(
                "predicted_sensor, sensor_log_variance and target must share shape"
            )
        finite_mask = (
            torch.isfinite(predicted).all(dim=-1)
            & torch.isfinite(log_variance).all(dim=-1)
            & torch.isfinite(target).all(dim=-1)
        )
        if reconstruction_mask is not None:
            mask = reconstruction_mask.to(device=predicted.device)
            if mask.ndim == 2 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            if mask.shape != finite_mask.shape:
                raise ValueError("reconstruction_mask must have shape [B] or [B,1]")
            finite_mask = finite_mask & (mask > 0)
        safe_mask = finite_mask.unsqueeze(1)
        safe_predicted = torch.where(safe_mask, predicted, torch.zeros_like(predicted))
        safe_target = torch.where(safe_mask, target, torch.zeros_like(target))
        safe_log_variance = torch.where(
            safe_mask, log_variance, torch.zeros_like(log_variance)
        )
        per_latent = 0.5 * (
            (safe_predicted - safe_target).pow(2) * torch.exp(-safe_log_variance)
            + safe_log_variance
        )
        per_sample = per_latent.mean(dim=-1)
        return self._mask_mean(per_sample, finite_mask)

    def _calibration_loss(
        self,
        aux: dict[str, Tensor] | None,
        quality_target: Tensor | None,
        logits: Tensor,
    ) -> Tensor:
        if (
            self.calibration_weight == 0.0
            or aux is None
            or quality_target is None
            or "sensor_reliability" not in aux
        ):
            return logits.new_tensor(0.0)
        reliability = aux["sensor_reliability"].clamp(1e-6, 1.0 - 1e-6)
        target = quality_target.to(device=reliability.device, dtype=reliability.dtype)
        if target.ndim == 1:
            target = target.unsqueeze(1)
        if target.shape != reliability.shape:
            raise ValueError("quality_target must have shape [B,1] or [B]")
        target = target.clamp(0.0, 1.0)
        mask = torch.isfinite(reliability) & torch.isfinite(target)
        if not bool(mask.any()):
            return logits.new_tensor(0.0)
        per_value = F.binary_cross_entropy(
            reliability[mask],
            target[mask],
            reduction="none",
        )
        if self.calibration_clean_weight > 1.0:
            clean_weight = torch.where(
                target[mask] >= 1.0 - 1e-6,
                torch.full_like(
                    per_value, self.calibration_clean_weight
                ),
                torch.ones_like(per_value),
            )
            return (per_value * clean_weight).sum() / clean_weight.sum()
        return per_value.mean()

    def _variance_prior_loss(
        self,
        aux: dict[str, Tensor] | None,
        logits: Tensor,
    ) -> Tensor:
        if (
            self.variance_prior_weight == 0.0
            or aux is None
            or "sensor_log_variance" not in aux
        ):
            return logits.new_tensor(0.0)
        log_variance = aux["sensor_log_variance"]
        finite = torch.isfinite(log_variance)
        if not bool(finite.any()):
            return logits.new_tensor(0.0)
        return (
            log_variance[finite] - self.variance_prior_target
        ).pow(2).mean()

    def _sensor_grid_reconstruction_loss(
        self,
        aux: dict[str, Tensor] | None,
        sensor_grid_target: Tensor | None,
        logits: Tensor,
    ) -> Tensor:
        if (
            self.sensor_grid_reconstruction_weight == 0.0
            or aux is None
            or sensor_grid_target is None
            or "predicted_sensor_grid" not in aux
        ):
            return logits.new_tensor(0.0)
        prediction = aux["predicted_sensor_grid"]
        target = sensor_grid_target.detach().to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        if prediction.shape != target.shape or tuple(prediction.shape[1:]) != (
            1,
            4,
            4,
        ):
            raise ValueError(
                "predicted_sensor_grid and sensor_grid_target must share "
                "shape [B,1,4,4]"
            )
        finite = torch.isfinite(prediction) & torch.isfinite(target)
        if not bool(finite.all()):
            raise ValueError("sensor grid reconstruction contains non-finite values")
        return F.smooth_l1_loss(prediction, target)

    def _calibration_ranking_loss(
        self,
        aux: dict[str, Tensor] | None,
        quality_target: Tensor | None,
        logits: Tensor,
    ) -> Tensor:
        if (
            self.calibration_ranking_weight == 0.0
            or aux is None
            or quality_target is None
            or "sensor_reliability" not in aux
        ):
            return logits.new_tensor(0.0)
        reliability = aux["sensor_reliability"].reshape(-1).clamp(
            1e-6,
            1.0 - 1e-6,
        )
        target = quality_target.to(
            device=reliability.device,
            dtype=reliability.dtype,
        ).reshape(-1)
        if reliability.shape != target.shape:
            raise ValueError("quality_target must match sensor reliability")
        valid = torch.isfinite(reliability) & torch.isfinite(target)
        reliability = reliability[valid]
        target = target[valid]
        if reliability.numel() < 2:
            return logits.new_tensor(0.0)
        target_difference = target[:, None] - target[None, :]
        reliability_logits = torch.logit(reliability)
        logit_difference = (
            reliability_logits[:, None] - reliability_logits[None, :]
        )
        pair_mask = torch.triu(
            torch.ones_like(target_difference, dtype=torch.bool),
            diagonal=1,
        ) & (
            target_difference.abs() >= self.calibration_ranking_margin
        )
        if not bool(pair_mask.any()):
            return logits.new_tensor(0.0)
        rank_target = (target_difference[pair_mask] > 0.0).to(
            dtype=reliability.dtype
        )
        return F.binary_cross_entropy_with_logits(
            logit_difference[pair_mask],
            rank_target,
        )

    def _visual_anchor_safeguard_loss(
        self,
        aux: dict[str, Tensor] | None,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        if self.visual_anchor_safeguard_weight <= 0.0 or aux is None:
            return logits.new_tensor(0.0)
        anchor_logits = aux.get("visual_anchor_logits")
        if anchor_logits is None:
            return logits.new_tensor(0.0)
        if anchor_logits.shape != logits.shape:
            raise ValueError("visual_anchor_logits must match logits shape")
        fused_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.ce_class_weights,
            reduction="none",
        )
        anchor_loss = F.cross_entropy(
            anchor_logits,
            targets,
            weight=self.ce_class_weights,
            reduction="none",
        ).detach()
        return F.relu(fused_loss - anchor_loss).mean()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        aux: dict[str, Tensor] | None = None,
        sensor_target: Tensor | None = None,
        sensor_grid_target: Tensor | None = None,
        quality_target: Tensor | None = None,
        reconstruction_mask: Tensor | None = None,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if (
            self.ce_class_weights is not None
            and self.ce_class_weights.numel() != logits.shape[1]
        ):
            raise ValueError(
                "ce_class_weights length must equal the number of classes"
            )
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.ce_class_weights,
        )
        focal = self.focal(logits, targets)
        reconstruction = self._reconstruction_loss(
            aux, sensor_target, reconstruction_mask, logits
        )
        calibration = self._calibration_loss(aux, quality_target, logits)
        variance_prior = self._variance_prior_loss(aux, logits)
        grid_reconstruction = self._sensor_grid_reconstruction_loss(
            aux,
            sensor_grid_target,
            logits,
        )
        calibration_ranking = self._calibration_ranking_loss(
            aux,
            quality_target,
            logits,
        )
        visual_anchor_safeguard = self._visual_anchor_safeguard_loss(
            aux,
            logits,
            targets,
        )
        semantic_alignment = logits.sum() * 0.0
        if self.semantic_alignment_weight > 0.0:
            if aux is None or "fallback_semantic_logits" not in aux:
                raise ValueError(
                    "semantic alignment requires fallback_semantic_logits"
                )
            semantic_logits = aux["fallback_semantic_logits"]
            if semantic_logits.shape != logits.shape:
                raise ValueError(
                    "fallback_semantic_logits must match classification logits"
                )
            semantic_alignment = F.cross_entropy(
                semantic_logits,
                targets,
                weight=self.ce_class_weights,
            )
        total = (
            self.ce_weight * cross_entropy
            + self.focal_weight * focal
            + self.reconstruction_weight * reconstruction
            + self.calibration_weight * calibration
            + self.variance_prior_weight * variance_prior
            + self.sensor_grid_reconstruction_weight * grid_reconstruction
            + self.calibration_ranking_weight * calibration_ranking
            + self.visual_anchor_safeguard_weight * visual_anchor_safeguard
            + self.semantic_alignment_weight * semantic_alignment
        )
        if return_components:
            return total, {
                "cross_entropy": cross_entropy,
                "focal": focal,
                "sensor_reconstruction": reconstruction,
                "reliability_calibration": calibration,
                "sensor_variance_prior": variance_prior,
                "sensor_grid_reconstruction": grid_reconstruction,
                "reliability_ranking": calibration_ranking,
                "visual_anchor_safeguard": visual_anchor_safeguard,
                "semantic_alignment": semantic_alignment,
            }
        return total
