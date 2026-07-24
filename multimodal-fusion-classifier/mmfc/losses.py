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
    ) -> None:
        super().__init__()
        if ce_weight < 0 or focal_weight < 0 or ce_weight + focal_weight == 0:
            raise ValueError("损失权重必须非负，且至少一个大于 0")
        self.ce_weight = float(ce_weight)
        self.focal_weight = float(focal_weight)
        self.focal = MulticlassFocalLoss(alpha=alpha, gamma=gamma)

    def forward(
        self, logits: Tensor, targets: Tensor, *, return_components: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        cross_entropy = F.cross_entropy(logits, targets)
        focal = self.focal(logits, targets)
        total = self.ce_weight * cross_entropy + self.focal_weight * focal
        if return_components:
            return total, {"cross_entropy": cross_entropy, "focal": focal}
        return total
