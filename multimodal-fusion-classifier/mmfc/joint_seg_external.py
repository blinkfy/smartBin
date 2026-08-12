"""External 4x4 scene-sensor input for the independent joint-head v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .joint_seg import JointMultimodalSegmentationModel


def load_scene_sensor_index(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = record.get("source_image_id")
            grid = np.asarray(
                record.get("normalized_grid"), dtype=np.float32
            )
            source = record.get("source")
            if (
                not isinstance(source_id, str)
                or not source_id
                or source_id in records
                or grid.shape != (4, 4)
                or not np.isfinite(grid).all()
                or float(grid.min()) < 0.0
                or float(grid.max()) > 1.0
                or not isinstance(source, Mapping)
                or source.get("observation_scope") != "full_image"
                or source.get("full_image_only") is not True
                or source.get("annotation_inputs") != []
            ):
                raise ValueError(
                    f"invalid scene sensor record at line {line_number}"
                )
            records[source_id] = record
    if not records:
        raise ValueError("scene sensor observation manifest is empty")
    return records


def pack_scene_sensor_channel(images: Tensor, grids: Tensor) -> Tensor:
    """Append a quantized spatial sensor transport channel to uint8 RGB."""

    if (
        images.ndim != 4
        or images.shape[1] != 3
        or images.dtype != torch.uint8
    ):
        raise ValueError("images must be uint8 [B,3,H,W]")
    if grids.shape != (images.shape[0], 1, 4, 4):
        raise ValueError("grids must have shape [B,1,4,4]")
    if not torch.isfinite(grids).all():
        raise ValueError("grids contain non-finite values")
    raster = F.interpolate(
        grids.float().clamp(0.0, 1.0),
        size=images.shape[-2:],
        mode="nearest",
    )
    raster = (raster * 255.0).round().to(dtype=torch.uint8)
    return torch.cat([images, raster], dim=1)


def split_scene_sensor_channel(
    inputs: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """Recover RGB and the 4x4 observation after trainer normalization."""

    if inputs.ndim != 4 or inputs.shape[1] not in {3, 4}:
        raise ValueError("model input must be [B,3,H,W] or [B,4,H,W]")
    rgb = inputs[:, :3]
    if inputs.shape[1] == 3:
        return rgb, None
    sensor = F.adaptive_avg_pool2d(inputs[:, 3:4], output_size=(4, 4))
    return rgb, sensor.clamp(0.0, 1.0)


class ExternalSceneSensorDataset(YOLODataset):
    """YOLO dataset that transports a label-free 4x4 observation."""

    def __init__(
        self,
        *args: Any,
        sensor_observations: Mapping[str, Mapping[str, Any]],
        **kwargs: Any,
    ) -> None:
        self.sensor_observations = dict(sensor_observations)
        super().__init__(*args, **kwargs)
        source_ids = {Path(path).stem for path in self.im_files}
        missing = sorted(source_ids - set(self.sensor_observations))
        if missing:
            raise ValueError(
                f"scene sensor observations missing for "
                f"{len(missing)} images: {missing[:5]}"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        source_id = Path(str(sample["im_file"])).stem
        record = self.sensor_observations[source_id]
        sample["scene_sensor_observation"] = torch.tensor(
            record["normalized_grid"], dtype=torch.float32
        ).unsqueeze(0)
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        observations = [
            sample.pop("scene_sensor_observation") for sample in batch
        ]
        result = YOLODataset.collate_fn(batch)
        grids = torch.stack(observations, dim=0)
        result["img"] = pack_scene_sensor_channel(result["img"], grids)
        result["scene_sensor_observation"] = grids
        # Preserve the v1 auxiliary predictor as a missing-sensor fallback.
        result["soft_sensor_target"] = grids
        result["soft_sensor_coverage"] = torch.ones_like(grids)
        return result


class ExternalSensorJointSegmentationModel(
    JointMultimodalSegmentationModel
):
    """Joint head with optional packed full-image sensor observations."""

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
        rgb, packed_sensor = split_scene_sensor_channel(x)
        if sensor_grid is not None and packed_sensor is not None:
            raise ValueError(
                "sensor_grid cannot be supplied with a packed fourth channel"
            )
        effective_sensor = (
            sensor_grid if sensor_grid is not None else packed_sensor
        )
        self.last_external_input_used = effective_sensor is not None
        return super().predict(
            rgb,
            profile=profile,
            visualize=visualize,
            augment=augment,
            embed=embed,
            sensor_grid=effective_sensor,
        )


class ExternalSensorSegmentationValidator(SegmentationValidator):
    """Ultralytics validator that requires full-image sensor observations."""

    def __init__(
        self,
        *args: Any,
        sensor_observations: str | Path,
        **kwargs: Any,
    ) -> None:
        self.sensor_observation_index = load_scene_sensor_index(
            sensor_observations
        )
        self.external_sensor_batch_count = 0
        self.external_sensor_sample_count = 0
        super().__init__(*args, **kwargs)

    def build_dataset(
        self,
        img_path: str,
        mode: str = "val",
        batch: int | None = None,
    ):
        return ExternalSceneSensorDataset(
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
        )

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        images = batch.get("img")
        if not isinstance(images, Tensor) or images.shape[1] != 4:
            raise ValueError(
                "external-sensor validation requires a packed fourth channel"
            )
        self.external_sensor_batch_count += 1
        self.external_sensor_sample_count += int(images.shape[0])
        return super().preprocess(batch)


def build_external_sensor_dataset(
    *,
    trainer: Any,
    img_path: str,
    mode: str,
    batch: int | None,
    sensor_observations: Mapping[str, Mapping[str, Any]],
) -> ExternalSceneSensorDataset:
    """Shared dataset builder for the v2 trainer."""

    stride = max(int(unwrap_model(trainer.model).stride.max()), 32)
    return ExternalSceneSensorDataset(
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
        fraction=(
            trainer.args.fraction if mode == "train" else 1.0
        ),
        sensor_observations=sensor_observations,
    )
