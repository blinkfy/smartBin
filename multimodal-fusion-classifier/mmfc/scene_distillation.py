"""RGB-only distillation of object-Qwen scene fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from build_scene_sensor_sidecars import discover_images
from .model import PROJECT_ROOT


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return (
        value.resolve()
        if value.is_absolute()
        else (PROJECT_ROOT / value).resolve()
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_project_path(path)
    values: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{resolved}:{line_number}: JSON root must be an object"
                )
            values.append(value)
    return values


def scene_image_inventory(
    images_root: str | Path,
) -> dict[str, dict[str, str]]:
    records = discover_images(resolve_project_path(images_root))
    return {record["source_image_id"]: record for record in records}


def image_to_tensor(image: Image.Image, image_size: int) -> Tensor:
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    resized = oriented.resize(
        (int(image_size), int(image_size)),
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class SceneTeacherDataset(Dataset[dict[str, Any]]):
    """Full RGB images paired with annotation-derived teacher fields."""

    def __init__(
        self,
        teacher_path: str | Path,
        images_root: str | Path,
        *,
        split: str,
        image_size: int = 256,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if image_size < 64:
            raise ValueError("image_size must be at least 64")
        inventory = scene_image_inventory(images_root)
        records = load_jsonl(teacher_path)
        wrong_splits = sorted(
            {
                str(value.get("split"))
                for value in records
                if value.get("split") != split
            }
        )
        if wrong_splits:
            raise ValueError(
                f"teacher file contains non-{split} records: {wrong_splits}"
            )
        if not records:
            raise ValueError(f"teacher split is empty: {split}")
        seen: set[str] = set()
        prepared: list[dict[str, Any]] = []
        for record in sorted(records, key=lambda item: item["source_image_id"]):
            source_id = record.get("source_image_id")
            image_record = inventory.get(str(source_id))
            teacher = np.asarray(record.get("teacher_grid"), dtype=np.float32)
            coverage = np.asarray(
                record.get("coverage_grid"), dtype=np.float32
            )
            if (
                not isinstance(source_id, str)
                or source_id in seen
                or image_record is None
                or image_record["split"] != split
                or teacher.shape != (4, 4)
                or coverage.shape != (4, 4)
                or not np.isfinite(teacher).all()
                or not np.isfinite(coverage).all()
                or float(teacher.min()) < 0.0
                or float(teacher.max()) > 1.0
                or float(coverage.min()) < 0.0
                or float(coverage.max()) > 1.0
            ):
                raise ValueError(f"invalid teacher record: {source_id}")
            seen.add(source_id)
            prepared.append(
                {
                    "source_image_id": source_id,
                    "image_path": resolve_project_path(
                        image_record["image_path"]
                    ),
                    "teacher": teacher,
                    "coverage": coverage,
                }
            )
        self.records = prepared
        self.image_size = int(image_size)
        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            tensor = image_to_tensor(image, self.image_size)
        return {
            "image": tensor,
            "teacher": torch.from_numpy(record["teacher"]).unsqueeze(0),
            "coverage": torch.from_numpy(record["coverage"]).unsqueeze(0),
            "source_image_id": record["source_image_id"],
        }


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SceneQwenDistiller(nn.Module):
    """Compact CNN mapping one full RGB image to a 4x4 Qwen teacher field."""

    def __init__(self, channels: Sequence[int] = (24, 32, 48, 64, 96)) -> None:
        super().__init__()
        if len(channels) != 5 or any(int(value) < 1 for value in channels):
            raise ValueError("channels must contain five positive integers")
        values = [int(value) for value in channels]
        blocks: list[nn.Module] = []
        in_channels = 3
        for index, out_channels in enumerate(values):
            blocks.append(
                ConvNormAct(
                    in_channels,
                    out_channels,
                    kernel_size=5 if index == 0 else 3,
                    stride=2,
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            ConvNormAct(values[-1], 48),
            nn.Conv2d(48, 1, kernel_size=1),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        return torch.sigmoid(self.head(self.pool(self.encoder(images))))


def distillation_loss(
    prediction: Tensor,
    teacher: Tensor,
    coverage: Tensor,
    *,
    foreground_weight: float = 4.0,
    l1_weight: float = 0.1,
    gradient_weight: float = 0.2,
) -> tuple[Tensor, dict[str, Tensor]]:
    if (
        prediction.shape != teacher.shape
        or prediction.shape != coverage.shape
        or prediction.ndim != 4
        or prediction.shape[1:] != (1, 4, 4)
    ):
        raise ValueError("prediction, teacher and coverage must be [B,1,4,4]")
    weights = 1.0 + float(foreground_weight) * torch.sqrt(
        coverage.float().clamp(0.0, 1.0)
    )
    weighted_mse = (weights * (prediction - teacher).square()).mean()
    l1 = (prediction - teacher).abs().mean()
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    true_dx = teacher[:, :, :, 1:] - teacher[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    true_dy = teacher[:, :, 1:, :] - teacher[:, :, :-1, :]
    gradient = 0.5 * (
        (pred_dx - true_dx).abs().mean()
        + (pred_dy - true_dy).abs().mean()
    )
    total = (
        weighted_mse
        + float(l1_weight) * l1
        + float(gradient_weight) * gradient
    )
    return total, {
        "weighted_mse": weighted_mse.detach(),
        "l1": l1.detach(),
        "gradient": gradient.detach(),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while (
            end < values.size
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def correlation(a: np.ndarray, b: np.ndarray, *, rank: bool = False) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("correlation arrays must have equal nontrivial size")
    if rank:
        x = _rankdata(x)
        y = _rankdata(y)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else 0.0


def prediction_metrics(
    prediction: np.ndarray,
    teacher: np.ndarray,
    coverage: np.ndarray,
    *,
    train_mean_grid: np.ndarray,
    threshold: float = 0.05,
) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(teacher, dtype=np.float64)
    cov = np.asarray(coverage, dtype=np.float64)
    mean_grid = np.asarray(train_mean_grid, dtype=np.float64).reshape(1, 1, 4, 4)
    if pred.shape != true.shape or pred.shape != cov.shape:
        raise ValueError("prediction metric arrays must share shape")
    squared = np.square(pred - true)
    foreground = cov > 0.0
    background = ~foreground
    pred_positive = pred > float(threshold)
    true_positive = true > float(threshold)
    intersection = np.logical_and(pred_positive, true_positive).sum()
    union = np.logical_or(pred_positive, true_positive).sum()
    return {
        "sample_count": float(pred.shape[0]),
        "mse": float(squared.mean()),
        "mae": float(np.abs(pred - true).mean()),
        "foreground_mse": float(squared[foreground].mean()),
        "background_mse": float(squared[background].mean()),
        "elementwise_pearson": correlation(pred, true),
        "elementwise_spearman": correlation(pred, true, rank=True),
        "positive_iou_at_0_05": (
            float(intersection / union) if union else 1.0
        ),
        "zero_baseline_mse": float(np.square(true).mean()),
        "train_mean_grid_baseline_mse": float(
            np.square(mean_grid - true).mean()
        ),
    }
