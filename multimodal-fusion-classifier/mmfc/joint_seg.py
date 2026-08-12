"""Joint text and soft-sensor conditioning for an Ultralytics Segment head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import SegmentationModel


def load_text_prototypes(path: str | Path, *, num_classes: int) -> Tensor:
    values = np.load(Path(path), allow_pickle=False)
    expected = (num_classes, 512)
    if values.shape != expected:
        raise ValueError(
            f"text prototypes must have shape {expected}, got {values.shape}"
        )
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("text prototypes must be finite float32 values")
    tensor = torch.from_numpy(values.copy())
    return F.normalize(tensor, dim=-1)


def load_sensor_target_index(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = str(record.get("source_image_id", ""))
            if not source_id or source_id in records:
                raise ValueError(
                    f"invalid or duplicate source_image_id at line {line_number}"
                )
            for key in ("normalized_grid", "coverage_grid"):
                values = np.asarray(record.get(key), dtype=np.float32)
                if (
                    values.shape != (4, 4)
                    or not np.isfinite(values).all()
                    or float(values.min()) < 0.0
                    or float(values.max()) > 1.0
                ):
                    raise ValueError(f"invalid {key} for source {source_id}")
            records[source_id] = record
    if not records:
        raise ValueError("soft-sensor target manifest is empty")
    return records


class SoftSensorTargetDataset(YOLODataset):
    """YOLO segmentation dataset with training-only soft-sensor targets."""

    def __init__(
        self,
        *args: Any,
        sensor_targets: Mapping[str, Mapping[str, Any]],
        **kwargs: Any,
    ) -> None:
        self.sensor_targets = dict(sensor_targets)
        super().__init__(*args, **kwargs)
        image_source_ids = {Path(path).stem for path in self.im_files}
        missing = sorted(image_source_ids - set(self.sensor_targets))
        if missing:
            raise ValueError(
                f"soft-sensor targets missing for {len(missing)} images: {missing[:5]}"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        source_id = Path(str(sample["im_file"])).stem
        target = self.sensor_targets[source_id]
        sample["soft_sensor_target"] = torch.tensor(
            target["normalized_grid"], dtype=torch.float32
        ).unsqueeze(0)
        sample["soft_sensor_coverage"] = torch.tensor(
            target["coverage_grid"], dtype=torch.float32
        ).unsqueeze(0)
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        targets = [sample.pop("soft_sensor_target") for sample in batch]
        coverage = [sample.pop("soft_sensor_coverage") for sample in batch]
        result = YOLODataset.collate_fn(batch)
        result["soft_sensor_target"] = torch.stack(targets, dim=0)
        result["soft_sensor_coverage"] = torch.stack(coverage, dim=0)
        return result


class SensorTextFeatureFusion(nn.Module):
    """Condition P3/P4/P5 maps before the original Segment head."""

    def __init__(
        self,
        channels: Sequence[int],
        text_prototypes: Tensor,
        *,
        semantic_dim: int = 64,
        sensor_hidden_dim: int = 256,
        temperature: float = 0.10,
        text_enabled: bool = True,
        sensor_enabled: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or any(int(value) <= 0 for value in channels):
            raise ValueError("channels must contain three positive values")
        if text_prototypes.ndim != 2 or text_prototypes.shape[1] != 512:
            raise ValueError("text_prototypes must have shape [classes,512]")
        if semantic_dim <= 0 or sensor_hidden_dim <= 0 or temperature <= 0:
            raise ValueError("fusion dimensions and temperature must be positive")

        self.channels = tuple(int(value) for value in channels)
        self.semantic_dim = int(semantic_dim)
        self.temperature = float(temperature)
        self.text_enabled = bool(text_enabled)
        self.sensor_enabled = bool(sensor_enabled)
        if not (self.text_enabled or self.sensor_enabled):
            raise ValueError("at least one fusion branch must be enabled")
        self.register_buffer(
            "text_prototypes",
            F.normalize(text_prototypes.detach().float(), dim=-1),
        )
        self.text_projection = nn.Linear(512, semantic_dim, bias=False)
        self.visual_semantic = nn.ModuleList(
            nn.Conv2d(channel, semantic_dim, kernel_size=1)
            for channel in self.channels
        )
        self.text_output = nn.ModuleList(
            nn.Conv2d(semantic_dim, channel, kernel_size=1)
            for channel in self.channels
        )

        pooled_dim = sum(self.channels)
        self.sensor_predictor = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, sensor_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        self.sensor_mean = nn.Linear(sensor_hidden_dim, 16)
        self.sensor_uncertainty = nn.Linear(sensor_hidden_dim, 16)
        self.sensor_grid_encoder = nn.Sequential(
            nn.Conv2d(1, semantic_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(semantic_dim, semantic_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.sensor_output = nn.ModuleList(
            nn.Conv2d(semantic_dim, channel, kernel_size=1)
            for channel in self.channels
        )
        for module in (*self.text_output, *self.sensor_output):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.text_scale = nn.Parameter(torch.ones(len(self.channels)))
        self.sensor_scale = nn.Parameter(torch.ones(len(self.channels)))
        self.last_aux: dict[str, Tensor] = {}

    def forward(
        self,
        features: Sequence[Tensor],
        observed_sensor: Tensor | None = None,
    ) -> list[Tensor]:
        if len(features) != len(self.channels):
            raise ValueError("expected exactly three YOLO feature maps")
        text_enabled = bool(getattr(self, "text_enabled", True))
        sensor_enabled = bool(getattr(self, "sensor_enabled", True))
        batch_size = int(features[0].shape[0])
        for feature, channels in zip(features, self.channels):
            if (
                feature.ndim != 4
                or feature.shape[0] != batch_size
                or feature.shape[1] != channels
            ):
                raise ValueError("YOLO feature map shape does not match fusion contract")

        reference = features[0]
        if sensor_enabled:
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
            reliability = torch.exp(
                -uncertainty.mean(dim=(1, 2, 3), keepdim=True)
            )
        else:
            if observed_sensor is not None:
                raise ValueError(
                    "observed_sensor cannot be used when sensor branch is disabled"
                )
            predicted_sensor = reference.new_zeros(batch_size, 1, 4, 4)
            uncertainty = reference.new_zeros(batch_size, 1, 4, 4)
            reliability = reference.new_zeros(batch_size, 1, 1, 1)

        if observed_sensor is None:
            effective_sensor = predicted_sensor
            external_used = reference.new_zeros(batch_size, 1)
        else:
            if observed_sensor.shape != (batch_size, 1, 4, 4):
                raise ValueError("observed_sensor must have shape [B,1,4,4]")
            if not torch.isfinite(observed_sensor).all():
                raise ValueError("observed_sensor contains non-finite values")
            effective_sensor = observed_sensor.to(
                device=predicted_sensor.device,
                dtype=predicted_sensor.dtype,
            ).clamp(0.0, 1.0)
            external_used = predicted_sensor.new_ones(batch_size, 1)

        text_bank = (
            F.normalize(self.text_projection(self.text_prototypes), dim=-1)
            if text_enabled
            else None
        )
        sensor_base = (
            self.sensor_grid_encoder(effective_sensor)
            if sensor_enabled
            else None
        )
        fused: list[Tensor] = []
        entropy_values: list[Tensor] = []
        for index, feature in enumerate(features):
            fused_feature = feature
            if text_enabled:
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
                text_gain = torch.tanh(self.text_scale[index])
                fused_feature = fused_feature + text_gain * text_residual
                entropy_values.append(
                    -(attention.clamp_min(1e-8).log() * attention)
                    .sum(dim=1)
                    .mean(dim=(1, 2))
                )
            else:
                entropy_values.append(reference.new_zeros(batch_size))
            if sensor_enabled:
                sensor_residual = self.sensor_output[index](
                    F.interpolate(
                        sensor_base,
                        size=feature.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )
                sensor_gain = torch.tanh(self.sensor_scale[index])
                fused_feature = (
                    fused_feature
                    + sensor_gain * reliability * sensor_residual
                )
            fused.append(fused_feature)

        self.last_aux = {
            "predicted_sensor_grid": predicted_sensor,
            "effective_sensor_grid": effective_sensor,
            "sensor_uncertainty": uncertainty,
            "sensor_reliability": reliability.flatten(1),
            "external_sensor_used": external_used,
            "text_attention_entropy": torch.stack(entropy_values, dim=1).mean(
                dim=1, keepdim=True
            ),
            "text_enabled": reference.new_full(
                (batch_size, 1), float(text_enabled)
            ),
            "sensor_enabled": reference.new_full(
                (batch_size, 1), float(sensor_enabled)
            ),
        }
        return fused


class JointMultimodalSegmentationModel(SegmentationModel):
    """YOLO segmentation model whose native head consumes fused P3/P4/P5 maps."""

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
        fusion_mode: str = "full",
    ) -> None:
        self._external_sensor_grid: Tensor | None = None
        if fusion_mode not in {"full", "text_only", "sensor_only"}:
            raise ValueError(
                "fusion_mode must be full, text_only, or sensor_only"
            )
        self.fusion_mode = fusion_mode
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        channels = tuple(
            int(branch[0].conv.in_channels) for branch in head.cv2
        )
        self.joint_fusion = SensorTextFeatureFusion(
            channels,
            text_prototypes,
            semantic_dim=semantic_dim,
            sensor_hidden_dim=sensor_hidden_dim,
            temperature=text_temperature,
            text_enabled=fusion_mode != "sensor_only",
            sensor_enabled=fusion_mode != "text_only",
        )
        self.sensor_reconstruction_weight = float(sensor_reconstruction_weight)
        self.sensor_calibration_weight = float(sensor_calibration_weight)
        if (
            self.sensor_reconstruction_weight < 0
            or self.sensor_calibration_weight < 0
        ):
            raise ValueError("sensor loss weights must be non-negative")

    def predict(
        self,
        x: Tensor,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
        embed: list[int] | None = None,
        *,
        sensor_grid: Tensor | None = None,
    ):
        previous = self._external_sensor_grid
        self._external_sensor_grid = sensor_grid
        try:
            return super().predict(
                x,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
            )
        finally:
            self._external_sensor_grid = previous

    def _predict_once(
        self,
        x: Tensor,
        profile: bool = False,
        visualize: bool = False,
        embed: list[int] | None = None,
    ):
        y, dt, embeddings = [], [], []
        embed_set = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed_set)
        for module in self.model:
            if module.f != -1:
                x = (
                    y[module.f]
                    if isinstance(module.f, int)
                    else [x if index == -1 else y[index] for index in module.f]
                )
            if profile:
                self._profile_one_layer(module, x, dt)
            if module is self.model[-1] and hasattr(self, "joint_fusion"):
                x = self.joint_fusion(x, self._external_sensor_grid)
            x = module(x)
            y.append(x if module.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization

                feature_visualization(x, module.type, module.i, save_dir=visualize)
            if module.i in embed_set:
                embeddings.append(
                    F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
                )
                if module.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch: dict[str, Any], preds=None):
        if preds is None:
            preds = self.predict(batch["img"])
        base_total, base_items = super().loss(batch, preds)
        target = batch.get("soft_sensor_target")
        if target is None:
            return base_total, base_items
        batch_size = int(batch["img"].shape[0])
        if not getattr(self.joint_fusion, "sensor_enabled", True):
            sensor_loss = base_total.new_zeros(())
            total = torch.cat(
                [base_total.reshape(-1), sensor_loss.reshape(1)]
            )
            items = torch.cat(
                [base_items.reshape(-1), sensor_loss.detach().reshape(1)]
            )
            return total, items
        target = target.to(
            device=batch["img"].device,
            dtype=batch["img"].dtype,
        )
        coverage = batch.get("soft_sensor_coverage")
        if coverage is None:
            coverage = torch.zeros_like(target)
        else:
            coverage = coverage.to(device=target.device, dtype=target.dtype)
        auxiliary = self.joint_fusion.last_aux
        predicted = auxiliary["predicted_sensor_grid"]
        uncertainty = auxiliary["sensor_uncertainty"]
        if predicted.shape != target.shape:
            raise ValueError("predicted and target sensor grids differ in shape")

        weights = 0.5 + coverage
        reconstruction_elements = F.smooth_l1_loss(
            predicted, target, reduction="none"
        )
        reconstruction = (reconstruction_elements * weights).sum() / weights.sum()
        absolute_error = (predicted.detach() - target).abs()
        calibration = F.smooth_l1_loss(
            uncertainty,
            absolute_error,
            reduction="mean",
        )
        sensor_loss = (
            self.sensor_reconstruction_weight * reconstruction
            + self.sensor_calibration_weight * calibration
        )
        total = torch.cat(
            [base_total.reshape(-1), sensor_loss.reshape(1) * batch_size]
        )
        items = torch.cat(
            [base_items.reshape(-1), sensor_loss.detach().reshape(1)]
        )
        return total, items


def parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    joint = sum(parameter.numel() for parameter in model.joint_fusion.parameters())
    return {
        "total": int(total),
        "trainable": int(trainable),
        "joint_fusion": int(joint),
    }
