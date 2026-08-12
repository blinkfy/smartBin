"""Label-free mismatched-sensor training for the reliable joint Segment head."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.torch_utils import unwrap_model

from .joint_seg_reliable import (
    ReliableExternalSceneSensorDataset,
    pack_reliable_sensor_channels,
)


def build_batch_mismatch_plan(
    source_ids: Sequence[str],
    *,
    probability: float,
    seed: int,
) -> list[int | None]:
    """Choose deterministic non-self donors without reading annotations."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("mismatch probability must be in [0,1]")
    if not source_ids:
        raise ValueError("source_ids must not be empty")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_ids must be unique within a batch")
    if len(source_ids) == 1:
        return [None]

    batch_token = "|".join(sorted(source_ids))
    plan: list[int | None] = []
    for index, source_id in enumerate(source_ids):
        digest = hashlib.sha256(
            f"{seed}:{batch_token}:{source_id}".encode("utf-8")
        ).digest()
        draw = int.from_bytes(digest[:8], "little") / float(2**64)
        if draw >= probability:
            plan.append(None)
            continue
        offset = 1 + int.from_bytes(digest[8:16], "little") % (
            len(source_ids) - 1
        )
        plan.append((index + offset) % len(source_ids))
    return plan


class ContrastiveReliableExternalSceneSensorDataset(
    ReliableExternalSceneSensorDataset
):
    """Add batch-mismatched sensor observations as reliability negatives."""

    def __init__(
        self,
        *args: Any,
        mismatch_probability: float = 0.35,
        **kwargs: Any,
    ) -> None:
        self.mismatch_probability = float(mismatch_probability)
        if not 0.0 <= self.mismatch_probability <= 1.0:
            raise ValueError("mismatch_probability must be in [0,1]")
        super().__init__(*args, **kwargs)

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

        mismatch = torch.zeros(len(batch), 1, dtype=quality.dtype)
        if self.degrade:
            plan = build_batch_mismatch_plan(
                source_ids,
                probability=self.mismatch_probability,
                seed=self.degradation_seed,
            )
            for sample_index, donor_index in enumerate(plan):
                if donor_index is None:
                    continue
                observed[sample_index] = clean_grids[donor_index]
                availability[sample_index] = 1.0
                quality[sample_index] = 0.0
                mismatch[sample_index] = 1.0

        result["img"] = pack_reliable_sensor_channels(
            result["img"], observed, availability
        )
        result["scene_sensor_observation"] = observed
        result["sensor_available"] = availability
        result["sensor_quality_target"] = quality
        result["sensor_mismatch_target"] = mismatch
        result["soft_sensor_target"] = clean_grids
        result["soft_sensor_coverage"] = torch.ones_like(clean_grids)
        return result


def build_contrastive_reliable_sensor_dataset(
    *,
    trainer: Any,
    img_path: str,
    mode: str,
    batch: int | None,
    sensor_observations: Mapping[str, Mapping[str, Any]],
    degradation: Mapping[str, Any],
    degradation_seed: int,
    mismatch_probability: float,
) -> ContrastiveReliableExternalSceneSensorDataset:
    """Build a train-only mismatched-sensor dataset."""

    stride = max(int(unwrap_model(trainer.model).stride.max()), 32)
    return ContrastiveReliableExternalSceneSensorDataset(
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
        prefix=f"{mode}: ",
        task=trainer.args.task,
        classes=trainer.args.classes,
        data=trainer.data,
        fraction=(trainer.args.fraction if mode == "train" else 1.0),
        sensor_observations=sensor_observations,
        degradation=degradation,
        degradation_seed=degradation_seed,
        degrade=mode == "train",
        mismatch_probability=mismatch_probability,
    )
