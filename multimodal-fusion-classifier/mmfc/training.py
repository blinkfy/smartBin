"""训练数据读取与训练循环。"""

from __future__ import annotations

import json
import importlib
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from .losses import JointClassificationLoss
from .model import PROJECT_ROOT, build_production_model


def resolve_project_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    """配置路径统一相对项目目录解析，而非相对调用者 cwd。"""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        # CLI 默认配置或用户相对路径均以子项目为锚点。
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
            records.append(value)
    return records


def _extract_grid(value: Any) -> list[list[float]]:
    """兼容清单内嵌、传感器 JSON 和 sidecar JSONL 的常见字段名。"""

    if isinstance(value, dict):
        for key in (
            "normalized_grid",
            "sensor_grid",
            "grid",
            "sensor_array",
            "array",
            "readings",
            "values",
            "data",
        ):
            if key in value:
                return _extract_grid(value[key])
        if "sensor" in value:
            return _extract_grid(value["sensor"])
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (1, 4, 4):
        array = array[0]
    if array.shape != (4, 4):
        raise ValueError(f"传感器阵列必须为 4×4，实际为 {list(array.shape)}")
    if not np.isfinite(array).all():
        raise ValueError("传感器阵列包含 NaN 或无穷值")
    return array.tolist()


def _sample_key(record: dict[str, Any]) -> str:
    """兼容清单早期的 ``id`` 与正式 schema 的 ``sample_id``。"""

    value = record.get("sample_id", record.get("id", ""))
    return str(value) if value is not None else ""


def load_sensor_grid(
    record: dict[str, Any],
    *,
    manifest_dir: Path,
    sidecar: dict[str, Any] | None = None,
) -> Tensor:
    for key in (
        "normalized_grid",
        "sensor_grid",
        "sensor",
        "grid",
        "sensor_array",
    ):
        if key in record:
            grid = _extract_grid(record[key])
            return torch.tensor(grid, dtype=torch.float32).unsqueeze(0)

    sample_id = _sample_key(record)
    if sidecar is not None and sample_id in sidecar:
        grid = _extract_grid(sidecar[sample_id])
        return torch.tensor(grid, dtype=torch.float32).unsqueeze(0)

    sensor_path_value = record.get("sensor_path") or record.get("sensor_json")
    if sensor_path_value:
        sensor_path = Path(sensor_path_value)
        if not sensor_path.is_absolute():
            candidates = (
                manifest_dir / sensor_path,
                PROJECT_ROOT / sensor_path,
                PROJECT_ROOT.parent / sensor_path,
            )
            sensor_path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
        try:
            with sensor_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except OSError as exc:
            raise FileNotFoundError(f"无法读取传感器文件：{sensor_path}") from exc
        grid = _extract_grid(value)
        return torch.tensor(grid, dtype=torch.float32).unsqueeze(0)

    raise ValueError(f"样本 {sample_id or '<unknown>'} 缺少 4×4 传感器阵列")


def _resolve_image_path(value: str | Path, manifest_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (
        manifest_dir / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT.parent / path,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _parse_bbox(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        aliases = (
            ("xmin", "ymin", "xmax", "ymax"),
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
        )
        for keys in aliases:
            if all(key in value for key in keys):
                return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    raise ValueError("bbox 必须为 [x1,y1,x2,y2] 或等价字段对象")


def load_cropped_image(
    image_path: str | Path,
    bbox: Any,
    *,
    manifest_dir: Path = PROJECT_ROOT,
    image_size: int = 640,
    preprocess: dict[str, Any] | None = None,
) -> Tensor:
    """从原图内存裁剪和预处理目标，不创建或改写任何图像文件。"""

    resolved = _resolve_image_path(image_path, manifest_dir)
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到原始图像：{resolved}")
    x1, y1, x2, y2 = _parse_bbox(bbox)
    with Image.open(resolved) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        left = max(0, min(width - 1, math.floor(min(x1, x2))))
        top = max(0, min(height - 1, math.floor(min(y1, y2))))
        right = max(left + 1, min(width, math.ceil(max(x1, x2))))
        bottom = max(top + 1, min(height, math.ceil(max(y1, y2))))
        crop = image.crop((left, top, right, bottom))
        array_uint8 = np.asarray(crop, dtype=np.uint8).copy()
    if preprocess and bool(preprocess.get("enabled", False)):
        array_uint8 = patent_preprocess(array_uint8, preprocess)
    resized = Image.fromarray(array_uint8, mode="RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def patent_preprocess(
    rgb: np.ndarray, options: dict[str, Any] | None = None
) -> np.ndarray:
    """专利 S1 预处理：双边滤波、暗图 CLAHE、Laplacian 和仿射变换。

    输入输出均为 RGB uint8 数组。OpenCV 仅在启用该路径时是必需依赖。
    """

    options = options or {}
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("预处理输入必须是 H×W×3 RGB 图像")
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError(
            "启用专利图像预处理需要 opencv-python；"
            "请安装该依赖或将 preprocess.enabled 设为 false"
        ) from exc

    bilateral = options.get("bilateral", {})
    image = rgb.astype(np.float32) / 255.0
    image = cv2.bilateralFilter(
        image,
        d=int(bilateral.get("diameter", 3)),
        sigmaColor=float(bilateral.get("sigma_color", 0.8)),
        sigmaSpace=float(bilateral.get("sigma_space", 1.0)),
    )
    image = np.clip(image, 0.0, 1.0)

    clahe_options = options.get("clahe", {})
    brightness = float(
        (
            0.299 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.114 * image[..., 2]
        ).mean()
    )
    if bool(clahe_options.get("enabled", True)) and brightness < float(
        clahe_options.get("brightness_threshold", 0.35)
    ):
        image_uint8 = np.rint(image * 255.0).astype(np.uint8)
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        tile_grid = clahe_options.get("tile_grid_size", [8, 8])
        if len(tile_grid) != 2:
            raise ValueError("clahe.tile_grid_size 必须包含两个整数")
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_options.get("clip_limit", 2.0)),
            tileGridSize=(int(tile_grid[0]), int(tile_grid[1])),
        )
        lab[..., 0] = clahe.apply(lab[..., 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    laplacian_options = options.get("laplacian", {})
    if bool(laplacian_options.get("enabled", True)):
        kernel_size = int(laplacian_options.get("kernel_size", 3))
        if kernel_size != 3:
            raise ValueError("专利默认 Laplacian kernel_size 必须为 3")
        laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=kernel_size)
        strength = float(laplacian_options.get("strength", 1.0))
        image = np.clip(image - strength * laplacian, 0.0, 1.0)

    affine = np.asarray(
        options.get("affine_matrix", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        dtype=np.float32,
    )
    if affine.shape != (2, 3):
        raise ValueError("preprocess.affine_matrix 必须为 2×3")
    height, width = image.shape[:2]
    image = cv2.warpAffine(
        image,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


class ManifestDataset(Dataset[dict[str, Any]]):
    """JSONL 分类样本；原图只在 ``__getitem__`` 中读取并内存裁剪。"""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        class_names: Iterable[str],
        image_size: int = 640,
        sensor_manifest: str | Path | None = None,
        preprocess: dict[str, Any] | None = None,
    ) -> None:
        self.manifest_path = resolve_project_path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"找不到数据清单：{self.manifest_path}")
        self.records = read_jsonl(self.manifest_path)
        if not self.records:
            raise ValueError(f"数据清单为空：{self.manifest_path}")
        self.manifest_dir = self.manifest_path.parent
        self.image_size = int(image_size)
        self.preprocess = preprocess
        self.class_to_id = {name: index for index, name in enumerate(class_names)}
        self.sensor_sidecar: dict[str, Any] | None = None
        if sensor_manifest:
            sidecar_path = resolve_project_path(sensor_manifest)
            self.sensor_sidecar = {}
            for line_number, item in enumerate(read_jsonl(sidecar_path), start=1):
                sample_id = _sample_key(item)
                if sample_id:
                    if sample_id in self.sensor_sidecar:
                        raise ValueError(
                            f"{sidecar_path}:{line_number} 包含重复 "
                            f"sample_id/id：{sample_id}"
                        )
                    self.sensor_sidecar[sample_id] = item

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample_id = _sample_key(record)
        required = ("image_path", "bbox")
        missing = [name for name in required if name not in record]
        if not sample_id:
            missing.append("sample_id/id")
        if missing:
            raise ValueError(f"第 {index} 个样本缺少字段：{missing}")
        has_label = "label" in record
        has_label_id = "label_id" in record
        mapped_label_id: int | None = None
        if has_label:
            label = str(record["label"])
            if label not in self.class_to_id:
                raise ValueError(f"样本 {sample_id} 的 label 不在配置 classes 中：{label}")
            mapped_label_id = self.class_to_id[label]
        if has_label_id:
            try:
                label_id = int(record["label_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"样本 {sample_id} 的 label_id 不是整数：{record['label_id']!r}"
                ) from exc
        elif mapped_label_id is not None:
            label_id = mapped_label_id
        else:
            raise ValueError(f"样本 {sample_id} 缺少有效 label/label_id")
        if not 0 <= label_id < len(self.class_to_id):
            raise ValueError(f"样本 {sample_id} 的 label_id 越界：{label_id}")
        if mapped_label_id is not None and label_id != mapped_label_id:
            raise ValueError(
                f"样本 {sample_id} 的 label/label_id 与配置类别顺序不一致："
                f"{record['label']!r} 应为 {mapped_label_id}，实际为 {label_id}"
            )
        return {
            "sample_id": sample_id,
            "image": load_cropped_image(
                record["image_path"],
                record["bbox"],
                manifest_dir=self.manifest_dir,
                image_size=self.image_size,
                preprocess=self.preprocess,
            ),
            "sensor": load_sensor_grid(
                record,
                manifest_dir=self.manifest_dir,
                sidecar=self.sensor_sidecar,
            ),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int
) -> LambdaLR:
    """按 epoch 的线性 warmup + 余弦衰减。"""

    def multiplier(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, multiplier)


def _degradation_options(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    nested = config.get("sensor_degradation")
    return nested if isinstance(nested, dict) else config


def _probability(options: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in options:
            return min(max(float(options[name]), 0.0), 1.0)
    return min(max(default, 0.0), 1.0)


def degrade_sensor_grid(
    sensor_grid: Tensor,
    config: dict[str, Any] | None = None,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Apply software-only sensor degradation for robust training."""

    if sensor_grid.ndim != 4 or tuple(sensor_grid.shape[1:]) != (1, 4, 4):
        raise ValueError("sensor_grid must have shape [B,1,4,4]")
    if not torch.isfinite(sensor_grid).all():
        raise ValueError("sensor_grid contains NaN or infinite values")

    options = _degradation_options(config)
    clean = sensor_grid.float()
    batch_size = clean.shape[0]
    availability = torch.ones(batch_size, 1, device=clean.device, dtype=clean.dtype)
    if not bool(options.get("enabled", False)):
        return {
            "degraded_grid": clean.clone(),
            "availability": availability,
            "quality_target": availability.clone(),
        }

    degraded = clean.clone()
    sample_shape = (batch_size, 1, 1, 1)
    rand_kwargs = {"device": clean.device, "dtype": clean.dtype, "generator": generator}
    active = torch.ones(sample_shape, device=clean.device, dtype=torch.bool)

    missing_mask = (
        torch.rand(sample_shape, **rand_kwargs)
        < _probability(options, "missing_probability", "missing_prob")
    )
    if bool(missing_mask.any()):
        degraded = torch.where(missing_mask, torch.zeros_like(degraded), degraded)
        active = active & ~missing_mask
        availability = (~missing_mask.view(batch_size, 1)).to(dtype=clean.dtype)

    noise_mask = (
        torch.rand(sample_shape, **rand_kwargs)
        < _probability(options, "noise_probability", "noise_prob")
    ) & active
    noise_std = max(float(options.get("noise_std", options.get("noise_strength", 0.0))), 0.0)
    if noise_std > 0.0 and bool(noise_mask.any()):
        noise = torch.randn(clean.shape, **rand_kwargs) * noise_std
        degraded = torch.where(noise_mask, degraded + noise, degraded)

    bias_mask = (
        torch.rand(sample_shape, **rand_kwargs)
        < _probability(options, "bias_probability", "drift_probability", "bias_prob")
    ) & active
    if bool(bias_mask.any()):
        bias_range = options.get("bias_range")
        if isinstance(bias_range, (list, tuple)) and len(bias_range) == 2:
            low, high = float(bias_range[0]), float(bias_range[1])
            amplitude = torch.rand(sample_shape, **rand_kwargs) * (high - low) + low
        else:
            strength = max(
                float(options.get("bias_strength", options.get("drift_strength", 0.0))),
                0.0,
            )
            min_strength = max(
                float(options.get("bias_min_strength", strength * 0.5)),
                0.0,
            )
            min_strength = min(min_strength, strength)
            magnitude = torch.rand(sample_shape, **rand_kwargs) * (
                strength - min_strength
            ) + min_strength
            sign = torch.where(
                torch.rand(sample_shape, **rand_kwargs) < 0.5,
                -torch.ones(sample_shape, device=clean.device, dtype=clean.dtype),
                torch.ones(sample_shape, device=clean.device, dtype=clean.dtype),
            )
            amplitude = sign * magnitude
        coords = torch.linspace(-1.0, 1.0, steps=4, device=clean.device, dtype=clean.dtype)
        row = coords.view(1, 1, 4, 1)
        col = coords.view(1, 1, 1, 4)
        direction = torch.randn((batch_size, 2, 1, 1), **rand_kwargs)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        drift_plane = direction[:, 0:1] * col + direction[:, 1:2] * row
        drift_plane = drift_plane / drift_plane.abs().amax(
            dim=(2, 3), keepdim=True
        ).clamp_min(1e-6)
        drift = amplitude * drift_plane
        degraded = torch.where(bias_mask, degraded + drift, degraded)

    stuck_mask = (
        torch.rand(sample_shape, **rand_kwargs)
        < _probability(options, "stuck_probability", "stuck_prob")
    ) & active
    if bool(stuck_mask.any()):
        if "stuck_value" in options:
            stuck_value = torch.full_like(degraded, float(options["stuck_value"]))
        else:
            stuck_value = clean[:, :, :1, :1].expand_as(degraded)
        degraded = torch.where(stuck_mask, stuck_value, degraded)

    degraded = degraded.clamp(0.0, 1.0)
    quality_scale = max(float(options.get("quality_scale", 0.25)), 1e-6)
    mean_absolute_change = (degraded - clean).abs().mean(dim=(1, 2, 3), keepdim=False)
    quality_target = (1.0 - mean_absolute_change.unsqueeze(1) / quality_scale).clamp(
        0.0, 1.0
    )
    quality_target = quality_target * availability
    return {
        "degraded_grid": degraded,
        "availability": availability,
        "quality_target": quality_target,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: JointClassificationLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    robustness_config: dict[str, Any] | None = None,
    return_metrics: bool = False,
) -> tuple[float, float] | tuple[float, float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    component_totals: dict[str, float] = {}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            sensors = batch["sensor"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            sensor_encoder = getattr(model, "sensor_encoder", None)
            if sensor_encoder is not None:
                with torch.no_grad():
                    clean_sensor_target = sensor_encoder(sensors).detach()
                degradation = (
                    degrade_sensor_grid(sensors, robustness_config)
                    if training
                    else degrade_sensor_grid(sensors, {"enabled": False})
                )
                logits, aux = model(
                    images,
                    degradation["degraded_grid"],
                    sensor_available=degradation["availability"],
                    return_aux=True,
                )
                loss, components = criterion(
                    logits,
                    labels,
                    aux=aux,
                    sensor_target=clean_sensor_target,
                    quality_target=degradation["quality_target"],
                    reconstruction_mask=torch.ones_like(degradation["availability"]),
                    return_components=True,
                )
            else:
                logits = model(images, sensors)
                loss, components = criterion(logits, labels, return_components=True)

            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.numel()
            total_loss += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=-1) == labels).sum())
            count += batch_size
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + (
                    float(value.detach()) * batch_size
                )
            if sensor_encoder is not None:
                for name in ("sensor_reliability", "sensor_consistency"):
                    component_totals[name] = component_totals.get(name, 0.0) + (
                        float(aux[name].detach().mean()) * batch_size
                    )

    average_loss = total_loss / max(count, 1)
    accuracy = correct / max(count, 1)
    if not return_metrics:
        return average_loss, accuracy
    metrics = {name: value / max(count, 1) for name, value in component_totals.items()}
    return average_loss, accuracy, metrics


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    data = config.get("data", {})
    classes = config.get("classes", [])
    class_names = [
        item["name"] if isinstance(item, dict) else str(item) for item in classes
    ]
    if not class_names:
        raise ValueError("配置中的 classes 不能为空")

    seed = int(training.get("seed", 42))
    set_seed(seed)
    device_name = training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    image_size = int(data.get("image_size", 640))
    preprocess = config.get("preprocess", {})
    sensor_manifest = data.get("sensor_manifest")
    train_dataset = ManifestDataset(
        data.get("train_manifest", "data/generated/train.jsonl"),
        class_names=class_names,
        image_size=image_size,
        sensor_manifest=sensor_manifest,
        preprocess=preprocess,
    )
    val_dataset = ManifestDataset(
        data.get("val_manifest", "data/generated/val.jsonl"),
        class_names=class_names,
        image_size=image_size,
        sensor_manifest=sensor_manifest,
        preprocess=preprocess,
    )
    loader_options = {
        "batch_size": int(training.get("batch_size", 16)),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = build_production_model(config).to(device)
    loss_config = config.get("loss", {})
    criterion = JointClassificationLoss(
        ce_weight=float(loss_config.get("ce_weight", 1.0)),
        focal_weight=float(loss_config.get("focal_weight", 1.0)),
        alpha=loss_config.get("alpha", 0.25),
        gamma=float(loss_config.get("gamma", 2.0)),
        reconstruction_weight=float(loss_config.get("reconstruction_weight", 0.0)),
        calibration_weight=float(loss_config.get("calibration_weight", 0.0)),
    ).to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    epochs = int(training.get("epochs", 100))
    scheduler = make_scheduler(
        optimizer, epochs=epochs, warmup_epochs=int(training.get("warmup_epochs", 5))
    )
    checkpoint_dir = resolve_project_path(
        training.get("checkpoint_dir", "checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    history: list[dict[str, float | int]] = []
    robustness_config = config.get("robustness", {})
    for epoch in range(epochs):
        train_loss, train_accuracy, train_components = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            robustness_config=robustness_config,
            return_metrics=True,
        )
        val_loss, val_accuracy, val_components = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            robustness_config=robustness_config,
            return_metrics=True,
        )
        scheduler.step()
        metrics: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for name, value in train_components.items():
            metrics[f"train_{name}"] = value
        for name, value in val_components.items():
            metrics[f"val_{name}"] = value
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "class_names": class_names,
            "config": {
                key: value for key, value in config.items() if not key.startswith("_")
            },
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    return {
        "best_val_accuracy": best_accuracy,
        "epochs": epochs,
        "checkpoint_dir": str(checkpoint_dir),
        "history": history,
    }
