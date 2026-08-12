"""训练数据读取与训练循环。"""

from __future__ import annotations

import hashlib
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
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from .checkpoints import load_checkpoint
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
    resized = Image.fromarray(array_uint8).resize(
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
        load_images: bool = True,
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
        self.load_images = bool(load_images)
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
            "source_image_id": str(
                record.get("source_image_id", sample_id)
            ),
            "image": (
                load_cropped_image(
                    record["image_path"],
                    record["bbox"],
                    manifest_dir=self.manifest_dir,
                    image_size=self.image_size,
                    preprocess=self.preprocess,
                )
                if self.load_images
                else torch.zeros(3, self.image_size, self.image_size)
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


class _SingleWorkerOrderRandomSampler(Sampler[int]):
    """Reproduce DataLoader(shuffle=True, num_workers=0) RNG consumption."""

    def __init__(self, dataset: Dataset, generator: torch.Generator) -> None:
        self.dataset = dataset
        self.generator = generator

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        size = len(self.dataset)
        if size <= 0:
            return
        # The legacy loader consumes one base seed before RandomSampler. Its
        # no-replacement implementation then evaluates a second empty slice.
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        yield from torch.randperm(size, generator=self.generator).tolist()
        torch.randperm(size, generator=self.generator)


def _limited_dataset(
    dataset: Dataset,
    value: Any,
    *,
    option_name: str,
) -> Dataset:
    if value is None:
        return dataset
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"{option_name} must be greater than 0")
    return Subset(dataset, range(min(limit, len(dataset))))


def compute_class_weighting(
    records: Iterable[dict[str, Any]],
    class_names: Iterable[str],
    options: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute reproducible train-split class weights without sampling labels."""

    names = list(class_names)
    if not names:
        raise ValueError("class_names must not be empty")
    class_to_id = {name: index for index, name in enumerate(names)}
    counts = np.zeros(len(names), dtype=np.int64)
    for record in records:
        label = record.get("label")
        if label is not None and str(label) in class_to_id:
            label_id = class_to_id[str(label)]
        else:
            try:
                label_id = int(record["label_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "class weighting records require a valid label or label_id"
                ) from exc
        if not 0 <= label_id < len(names):
            raise ValueError(f"class weighting label_id out of range: {label_id}")
        counts[label_id] += 1
    if bool((counts == 0).any()):
        missing = [names[index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f"class weighting found empty train classes: {missing}")

    if options is None:
        settings: dict[str, Any] = {"method": "none"}
    elif isinstance(options, str):
        settings = {"method": options}
    elif isinstance(options, dict):
        settings = dict(options)
    else:
        raise ValueError("loss.class_weighting must be a string or object")
    method = str(settings.get("method", "none")).strip().lower()
    valid_methods = {"none", "inverse_frequency", "sqrt_inverse", "effective_number"}
    if method not in valid_methods:
        raise ValueError(
            f"class weighting method must be one of {sorted(valid_methods)}"
        )

    weights: np.ndarray | None = None
    if method != "none":
        frequencies = counts.astype(np.float64)
        if method == "inverse_frequency":
            weights = 1.0 / frequencies
        elif method == "sqrt_inverse":
            weights = 1.0 / np.sqrt(frequencies)
        else:
            beta = float(settings.get("beta", 0.999))
            if not 0.0 <= beta < 1.0:
                raise ValueError("effective-number beta must be in [0,1)")
            weights = (1.0 - beta) / (1.0 - np.power(beta, frequencies))
        weights /= weights.mean()
        max_weight = settings.get("max_weight")
        if max_weight is not None:
            max_weight = float(max_weight)
            if not math.isfinite(max_weight) or max_weight <= 0.0:
                raise ValueError("class weighting max_weight must be finite and > 0")
            weights = np.minimum(weights, max_weight)
        if not bool(np.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            raise ValueError("computed class weights must be finite and positive")

    return {
        "method": method,
        "apply_to_ce": bool(settings.get("apply_to_ce", True)),
        "apply_to_focal": bool(settings.get("apply_to_focal", False)),
        "counts": {name: int(count) for name, count in zip(names, counts)},
        "weights": None if weights is None else weights.tolist(),
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _training_config_fingerprint(config: dict[str, Any]) -> str:
    """Hash training-critical configuration while ignoring resume controls."""

    payload = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    training = payload.get("training")
    if isinstance(training, dict):
        training.pop("resume", None)
        training.pop("resume_checkpoint", None)
        # Checkpoint cadence is operational and does not alter optimization.
        training.pop("checkpoint_every_epochs", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resume_path(
    training: dict[str, Any], checkpoint_dir: Path
) -> Path | None:
    explicit = training.get("resume_checkpoint")
    if explicit:
        return resolve_project_path(explicit)
    resume = training.get("resume", False)
    if isinstance(resume, str) and resume.strip():
        return resolve_project_path(resume)
    return checkpoint_dir / "last.pt" if bool(resume) else None


_VISUAL_ANCHOR_STATE_MAPPINGS = (
    ("visual_encoder.", "visual_encoder."),
    ("scale_projections.", "visual_anchor_scale_projections."),
    ("scale_prior_logits", "visual_anchor_scale_prior_logits"),
    ("visual_adapter.", "visual_anchor_adapter."),
    ("dynamic_scale_gate.", "visual_anchor_dynamic_scale_gate."),
    ("output_norm.", "visual_anchor_output_norm."),
    ("classifier.", "visual_anchor_classifier."),
)


def _validate_initialization_checkpoint(
    checkpoint: dict[str, Any],
    *,
    class_names: list[str],
    path: Path,
) -> dict[str, Tensor]:
    if checkpoint.get("class_names") != class_names:
        raise ValueError(
            f"initialization checkpoint class_names do not match config: {path}"
        )
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"initialization checkpoint has no model_state: {path}")
    return state


def _initialize_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    class_names: list[str],
    visual_anchor_only: bool = False,
) -> dict[str, Any]:
    path = resolve_project_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"initialization checkpoint not found: {path}")
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    source_state = _validate_initialization_checkpoint(
        checkpoint,
        class_names=class_names,
        path=path,
    )
    target_state = model.state_dict()
    compatible: dict[str, Tensor] = {}
    shape_mismatches: list[str] = []

    def shape_matches(target: Tensor, source: Tensor) -> bool:
        try:
            return target.shape == source.shape
        except RuntimeError as exc:
            if "uninitialized parameter or buffer" not in str(exc):
                raise
            return True

    if visual_anchor_only:
        if not bool(getattr(model, "visual_anchor_enabled", False)):
            raise ValueError(
                "visual_anchor_checkpoint requires model.visual_anchor_enabled"
            )
        for source_prefix, target_prefix in _VISUAL_ANCHOR_STATE_MAPPINGS:
            for source_key, value in source_state.items():
                if source_key == source_prefix:
                    suffix = ""
                elif source_key.startswith(source_prefix):
                    suffix = source_key[len(source_prefix) :]
                else:
                    continue
                target_key = target_prefix + suffix
                if target_key not in target_state:
                    continue
                if not shape_matches(target_state[target_key], value):
                    shape_mismatches.append(target_key)
                    continue
                compatible[target_key] = value
        required = {
            key
            for key in target_state
            if key.startswith("visual_encoder.")
            or key.startswith("visual_anchor_")
            and not key.startswith("visual_anchor_gate.")
        }
        missing = sorted(required - set(compatible))
        if missing:
            raise RuntimeError(
                "visual anchor initialization is incomplete: "
                f"missing={missing[:8]} total={len(missing)}"
            )
    else:
        for key, value in source_state.items():
            if key not in target_state:
                continue
            if not shape_matches(target_state[key], value):
                shape_mismatches.append(key)
                continue
            compatible[key] = value
        if not compatible:
            raise RuntimeError("initialization checkpoint has no compatible keys")
    if shape_mismatches:
        raise RuntimeError(
            "initialization checkpoint has shape mismatches: "
            f"{shape_mismatches[:8]}"
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "path": str(path),
        "epoch": int(checkpoint.get("epoch", 0)),
        "mode": "visual_anchor" if visual_anchor_only else "compatible",
        "loaded_key_count": len(compatible),
    }


def _initialize_semantic_memory_from_training_sensors(
    model: nn.Module,
    dataset: ManifestDataset,
    *,
    class_names: list[str],
    device: torch.device,
    batch_size: int = 512,
    trainable: bool = False,
) -> dict[str, Any]:
    """Initialize class-bound fallback values from train-split sensors only."""

    sensor_encoder = getattr(model, "sensor_encoder", None)
    initializer = getattr(model, "initialize_fallback_sensor_value_bank", None)
    if sensor_encoder is None or not callable(initializer):
        raise ValueError("model does not support semantic sensor memory")
    if batch_size <= 0:
        raise ValueError("semantic memory initialization batch size must be positive")

    class_to_id = {name: index for index, name in enumerate(class_names)}
    sums = torch.zeros(
        len(class_names),
        int(getattr(model, "feature_dim", 512)),
        device=device,
    )
    counts = torch.zeros(len(class_names), device=device)
    grids: list[Tensor] = []
    labels: list[int] = []

    def flush() -> None:
        if not grids:
            return
        grid_batch = torch.stack(grids).to(device)
        label_batch = torch.tensor(labels, device=device, dtype=torch.long)
        with torch.inference_mode():
            encoded = sensor_encoder(grid_batch)
        sums.index_add_(0, label_batch, encoded)
        counts.index_add_(
            0,
            label_batch,
            torch.ones_like(label_batch, dtype=counts.dtype),
        )
        grids.clear()
        labels.clear()

    sensor_encoder.eval()
    for record in dataset.records:
        label = record.get("label")
        if label is not None and str(label) in class_to_id:
            label_id = class_to_id[str(label)]
        else:
            label_id = int(record["label_id"])
        if not 0 <= label_id < len(class_names):
            raise ValueError(f"semantic memory label_id out of range: {label_id}")
        grids.append(
            load_sensor_grid(
                record,
                manifest_dir=dataset.manifest_dir,
                sidecar=dataset.sensor_sidecar,
            )
        )
        labels.append(label_id)
        if len(grids) >= batch_size:
            flush()
    flush()
    if bool((counts == 0).any()):
        missing = [
            class_names[index]
            for index in torch.nonzero(counts == 0, as_tuple=False)
            .flatten()
            .tolist()
        ]
        raise ValueError(f"semantic memory has empty classes: {missing}")
    centers = sums / counts.unsqueeze(1)
    initializer(centers, trainable=trainable)
    return {
        "mode": "train_sensor_class_centers",
        "sample_count": int(counts.sum().item()),
        "class_counts": {
            name: int(count)
            for name, count in zip(class_names, counts.detach().cpu().tolist())
        },
        "trainable": bool(trainable),
    }


def _restore_training_checkpoint(
    checkpoint_path: Path,
    *,
    config: dict[str, Any],
    class_names: list[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    data_loader_generator: torch.Generator,
    device: torch.device,
) -> tuple[int, float, list[dict[str, float | int]]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"resume checkpoint not found: {checkpoint_path}"
        )
    checkpoint = load_checkpoint(checkpoint_path, device)
    expected_fingerprint = _training_config_fingerprint(config)
    actual_fingerprint = checkpoint.get("config_fingerprint")
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "resume checkpoint configuration fingerprint mismatch: "
            f"expected={expected_fingerprint}, actual={actual_fingerprint}"
        )
    if checkpoint.get("class_names") != class_names:
        raise ValueError("resume checkpoint class_names do not match config")
    for required in (
        "optimizer_state",
        "scheduler_state",
        "history",
        "best_val_accuracy",
        "data_loader_generator_state",
        "torch_rng_state",
    ):
        if required not in checkpoint:
            raise ValueError(
                f"resume checkpoint is missing required field: {required}"
            )

    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = int(checkpoint.get("epoch", 0))
    history = checkpoint["history"]
    if not isinstance(history, list) or len(history) != start_epoch:
        raise ValueError(
            "resume checkpoint history length must equal completed epoch count"
        )
    data_loader_generator.set_state(
        checkpoint["data_loader_generator_state"].cpu()
    )
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_states = checkpoint.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    return start_epoch, float(checkpoint["best_val_accuracy"]), history


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
    noise_std_range = options.get("noise_std_range")
    if (
        isinstance(noise_std_range, (list, tuple))
        and len(noise_std_range) == 2
    ):
        noise_std_low = max(float(noise_std_range[0]), 0.0)
        noise_std_high = max(float(noise_std_range[1]), 0.0)
        if noise_std_low > noise_std_high:
            raise ValueError("noise_std_range must be ordered [low, high]")
        sample_noise_std = (
            torch.rand(sample_shape, **rand_kwargs)
            * (noise_std_high - noise_std_low)
            + noise_std_low
        )
    else:
        noise_std = max(
            float(
                options.get(
                    "noise_std",
                    options.get("noise_strength", 0.0),
                )
            ),
            0.0,
        )
        sample_noise_std = torch.full(
            sample_shape,
            noise_std,
            device=clean.device,
            dtype=clean.dtype,
        )
    if bool((sample_noise_std > 0.0).any()) and bool(noise_mask.any()):
        noise = torch.randn(clean.shape, **rand_kwargs) * sample_noise_std
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
    confusion: np.ndarray | None = None
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
            sensor_path_enabled = bool(
                getattr(
                    model,
                    "sensor_modality_enabled",
                    sensor_encoder is not None,
                )
            )
            if sensor_encoder is not None and sensor_path_enabled:
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
                    sensor_grid_target=sensors,
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
            predictions = logits.argmax(dim=-1)
            total_loss += float(loss.detach()) * batch_size
            correct += int((predictions == labels).sum())
            count += batch_size
            if return_metrics:
                num_classes = int(logits.shape[-1])
                if confusion is None:
                    confusion = np.zeros(
                        (num_classes, num_classes),
                        dtype=np.int64,
                    )
                elif confusion.shape != (num_classes, num_classes):
                    raise RuntimeError("number of classes changed within an epoch")
                encoded = labels * num_classes + predictions
                confusion += (
                    torch.bincount(
                        encoded,
                        minlength=num_classes * num_classes,
                    )
                    .reshape(num_classes, num_classes)
                    .detach()
                    .cpu()
                    .numpy()
                )
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + (
                    float(value.detach()) * batch_size
                )
            if sensor_encoder is not None and sensor_path_enabled:
                for name in ("sensor_reliability", "sensor_consistency"):
                    component_totals[name] = component_totals.get(name, 0.0) + (
                        float(aux[name].detach().mean()) * batch_size
                    )
                if bool(getattr(model, "visual_anchor_enabled", False)):
                    for name in (
                        "visual_anchor_gate",
                        "visual_anchor_confidence",
                        "visual_anchor_disagreement",
                    ):
                        component_totals[name] = component_totals.get(
                            name, 0.0
                        ) + float(aux[name].detach().mean()) * batch_size
                    anchor_accuracy = (
                        aux["visual_anchor_logits"].detach().argmax(dim=-1)
                        == labels
                    ).float().mean()
                    component_totals["visual_anchor_accuracy"] = (
                        component_totals.get("visual_anchor_accuracy", 0.0)
                        + float(anchor_accuracy) * batch_size
                    )

    average_loss = total_loss / max(count, 1)
    accuracy = correct / max(count, 1)
    if not return_metrics:
        return average_loss, accuracy
    metrics = {name: value / max(count, 1) for name, value in component_totals.items()}
    if confusion is None:
        metrics["macro_f1"] = 0.0
    else:
        true_positive = np.diag(confusion).astype(np.float64)
        predicted_total = confusion.sum(axis=0, dtype=np.float64)
        actual_total = confusion.sum(axis=1, dtype=np.float64)
        precision = np.divide(
            true_positive,
            predicted_total,
            out=np.zeros_like(true_positive),
            where=predicted_total > 0,
        )
        recall = np.divide(
            true_positive,
            actual_total,
            out=np.zeros_like(true_positive),
            where=actual_total > 0,
        )
        metrics["macro_f1"] = float(
            np.divide(
                2.0 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) > 0,
            ).mean()
        )
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
    selection_metric = str(
        training.get("selection_metric", "accuracy")
    ).strip().lower()
    supported_selection_metrics = {
        "accuracy",
        "macro_f1",
        "sensor_grid_reconstruction",
        "sensor_reconstruction",
        "semantic_alignment",
    }
    if selection_metric not in supported_selection_metrics:
        raise ValueError(
            "training.selection_metric must be one of "
            f"{sorted(supported_selection_metrics)}"
        )
    default_selection_mode = (
        "max" if selection_metric in {"accuracy", "macro_f1"} else "min"
    )
    selection_mode = str(
        training.get("selection_mode", default_selection_mode)
    ).strip().lower()
    if selection_mode not in {"max", "min"}:
        raise ValueError("training.selection_mode must be max or min")
    set_seed(seed)
    device_name = training.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    image_size = int(data.get("image_size", 640))
    preprocess = config.get("preprocess", {})
    sensor_manifest = data.get("sensor_manifest")
    enabled_modalities = set(
        config.get("model", {}).get(
            "enabled_modalities", ("visual", "text", "sensor")
        )
    )
    load_images = "visual" in enabled_modalities
    full_train_dataset = ManifestDataset(
        data.get("train_manifest", "data/generated/train.jsonl"),
        class_names=class_names,
        image_size=image_size,
        sensor_manifest=sensor_manifest,
        preprocess=preprocess,
        load_images=load_images,
    )
    full_val_dataset = ManifestDataset(
        data.get("val_manifest", "data/generated/val.jsonl"),
        class_names=class_names,
        image_size=image_size,
        sensor_manifest=sensor_manifest,
        preprocess=preprocess,
        load_images=load_images,
    )
    train_dataset = _limited_dataset(
        full_train_dataset,
        training.get("max_train_samples"),
        option_name="training.max_train_samples",
    )
    val_dataset = _limited_dataset(
        full_val_dataset,
        training.get("max_val_samples"),
        option_name="training.max_val_samples",
    )
    num_workers = int(training.get("num_workers", 0))
    loader_options = {
        "batch_size": int(training.get("batch_size", 16)),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(training.get("prefetch_factor", 2)),
            }
        )
    generator = torch.Generator().manual_seed(seed)
    preserve_single_worker_order = bool(
        training.get("preserve_single_worker_order", False)
    )
    if num_workers > 0 and preserve_single_worker_order:
        sampler = _SingleWorkerOrderRandomSampler(train_dataset, generator)
        worker_generator = torch.Generator().manual_seed(seed + 1_000_003)
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            generator=worker_generator,
            **loader_options,
        )
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **loader_options
        )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = build_production_model(config).to(device)
    initialization: list[dict[str, Any]] = []
    initialization_checkpoint = training.get("initialization_checkpoint")
    visual_anchor_checkpoint = training.get("visual_anchor_checkpoint")
    if (
        (initialization_checkpoint or visual_anchor_checkpoint)
        and (training.get("resume") or training.get("resume_checkpoint"))
    ):
        raise ValueError("initialization checkpoints cannot be combined with resume")
    if initialization_checkpoint:
        initialization.append(
            _initialize_model_from_checkpoint(
                model,
                initialization_checkpoint,
                class_names=class_names,
            )
        )
    if visual_anchor_checkpoint:
        initialization.append(
            _initialize_model_from_checkpoint(
                model,
                visual_anchor_checkpoint,
                class_names=class_names,
                visual_anchor_only=True,
            )
        )
    if bool(training.get("freeze_visual_anchor", False)):
        freeze_visual_anchor = getattr(model, "freeze_visual_anchor", None)
        if not callable(freeze_visual_anchor):
            raise ValueError("model does not support a frozen visual anchor")
        freeze_visual_anchor()
    frozen_modules = training.get("freeze_modules", ())
    if frozen_modules:
        if isinstance(frozen_modules, (str, bytes)):
            raise ValueError("training.freeze_modules must be a list of names")
        freeze_modules = getattr(model, "freeze_modules", None)
        if not callable(freeze_modules):
            raise ValueError("model does not support explicit module freezing")
        freeze_modules(tuple(str(name) for name in frozen_modules))
    if bool(training.get("initialize_fallback_sensor_value_bank", False)):
        initialization.append(
            _initialize_semantic_memory_from_training_sensors(
                model,
                full_train_dataset,
                class_names=class_names,
                device=device,
                batch_size=int(
                    training.get(
                        "semantic_memory_initialization_batch_size",
                        512,
                    )
                ),
                trainable=bool(
                    training.get("semantic_memory_trainable", False)
                ),
            )
        )
    loss_config = config.get("loss", {})
    class_weighting = compute_class_weighting(
        full_train_dataset.records,
        class_names,
        loss_config.get("class_weighting"),
    )
    class_weights = class_weighting["weights"]
    focal_alpha = loss_config.get("alpha", 0.25)
    if class_weights is not None and class_weighting["apply_to_focal"]:
        alpha_values = np.asarray(focal_alpha, dtype=np.float64)
        if alpha_values.ndim == 0:
            alpha_values = alpha_values * np.asarray(class_weights)
        elif alpha_values.ndim == 1 and alpha_values.size == len(class_names):
            alpha_values = alpha_values * np.asarray(class_weights)
        else:
            raise ValueError(
                "loss.alpha must be scalar or match classes when focal "
                "class weighting is enabled"
            )
        focal_alpha = alpha_values.tolist()
    criterion = JointClassificationLoss(
        ce_weight=float(loss_config.get("ce_weight", 1.0)),
        focal_weight=float(loss_config.get("focal_weight", 1.0)),
        alpha=focal_alpha,
        gamma=float(loss_config.get("gamma", 2.0)),
        ce_class_weights=(
            class_weights
            if class_weights is not None and class_weighting["apply_to_ce"]
            else None
        ),
        reconstruction_weight=float(loss_config.get("reconstruction_weight", 0.0)),
        calibration_weight=float(loss_config.get("calibration_weight", 0.0)),
        variance_prior_weight=float(
            loss_config.get("variance_prior_weight", 0.0)
        ),
        variance_prior_target=float(
            loss_config.get("variance_prior_target", -2.0)
        ),
        calibration_clean_weight=float(
            loss_config.get("calibration_clean_weight", 1.0)
        ),
        sensor_grid_reconstruction_weight=float(
            loss_config.get("sensor_grid_reconstruction_weight", 0.0)
        ),
        calibration_ranking_weight=float(
            loss_config.get("calibration_ranking_weight", 0.0)
        ),
        calibration_ranking_margin=float(
            loss_config.get("calibration_ranking_margin", 0.05)
        ),
        visual_anchor_safeguard_weight=float(
            loss_config.get("visual_anchor_safeguard_weight", 0.0)
        ),
        semantic_alignment_weight=float(
            loss_config.get("semantic_alignment_weight", 0.0)
        ),
    ).to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    epochs = int(training.get("epochs", 100))
    if epochs <= 0:
        raise ValueError("training.epochs must be greater than 0")
    scheduler = make_scheduler(
        optimizer, epochs=epochs, warmup_epochs=int(training.get("warmup_epochs", 5))
    )
    checkpoint_dir = resolve_project_path(
        training.get("checkpoint_dir", "checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = -1.0
    best_selection_score = (
        float("-inf") if selection_mode == "max" else float("inf")
    )
    best_checkpoint_epoch = 0
    best_checkpoint_val_accuracy = -1.0
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    resumed_from: str | None = None
    robustness_config = config.get("robustness", {})
    save_checkpoints = bool(training.get("save_checkpoints", True))
    checkpoint_every_epochs = int(
        training.get("checkpoint_every_epochs", 1)
    )
    if checkpoint_every_epochs <= 0:
        raise ValueError("training.checkpoint_every_epochs must be positive")
    config_fingerprint = _training_config_fingerprint(config)
    resume_checkpoint = _resume_path(training, checkpoint_dir)
    if resume_checkpoint is not None:
        start_epoch, best_accuracy, history = _restore_training_checkpoint(
            resume_checkpoint,
            config=config,
            class_names=class_names,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader_generator=generator,
            device=device,
        )
        resumed_from = str(resume_checkpoint.resolve())
        selection_history_key = f"val_{selection_metric}"
        selection_history = [
            float(row[selection_history_key])
            for row in history
            if selection_history_key in row
        ]
        if history and not selection_history:
            raise ValueError(
                "resume history does not contain configured selection metric: "
                f"{selection_history_key}"
            )
        best_selection_score = (
            (max(selection_history) if selection_mode == "max" else min(selection_history))
            if selection_history
            else (
                float("-inf")
                if selection_mode == "max"
                else float("inf")
            )
        )
        if selection_history:
            eligible_history = (
                row
                for row in history
                if selection_history_key in row
            )
            best_history_row = (
                max
                if selection_mode == "max"
                else min
            )(
                eligible_history,
                key=lambda row: float(row[selection_history_key]),
            )
            best_checkpoint_epoch = int(best_history_row["epoch"])
            best_checkpoint_val_accuracy = float(
                best_history_row["val_accuracy"]
            )
        if start_epoch > epochs:
            raise ValueError(
                "resume checkpoint epoch exceeds configured training.epochs"
            )
        print(
            json.dumps(
                {
                    "event": "resumed",
                    "checkpoint": resumed_from,
                    "completed_epochs": start_epoch,
                    "target_epochs": epochs,
                    "best_val_accuracy": best_accuracy,
                },
                ensure_ascii=False,
            )
        )

    for epoch in range(start_epoch, epochs):
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
        selection_score = (
            val_accuracy
            if selection_metric == "accuracy"
            else float(val_components[selection_metric])
        )
        is_best = (
            selection_score > best_selection_score
            if selection_mode == "max"
            else selection_score < best_selection_score
        )
        best_accuracy = max(best_accuracy, val_accuracy)
        if is_best:
            best_selection_score = selection_score
            best_checkpoint_epoch = epoch + 1
            best_checkpoint_val_accuracy = val_accuracy
        history_payload = {
            "experiment": config.get("experiment"),
            "config_fingerprint": config_fingerprint,
            "seed": seed,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "full_train_samples": len(full_train_dataset),
            "full_validation_samples": len(full_val_dataset),
            "checkpoints_saved": save_checkpoints,
            "class_weighting": class_weighting,
            "initialization": initialization,
            "resumed_from": resumed_from,
            "best_val_accuracy": best_accuracy,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "best_selection_score": best_selection_score,
            "best_checkpoint_epoch": best_checkpoint_epoch,
            "best_checkpoint_val_accuracy": best_checkpoint_val_accuracy,
            "history": history,
        }
        _atomic_write_json(checkpoint_dir / "history.json", history_payload)
        if save_checkpoints:
            checkpoint_metadata = {
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy,
                "best_val_accuracy": best_accuracy,
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_score": selection_score,
                "best_selection_score": best_selection_score,
                "class_names": class_names,
                "config_fingerprint": config_fingerprint,
                "class_weighting": class_weighting,
                "initialization": initialization,
                "history": history,
                "config": {
                    key: value
                    for key, value in config.items()
                    if not key.startswith("_")
                },
            }
            should_save_resume = (
                (epoch + 1) % checkpoint_every_epochs == 0
                or epoch + 1 == epochs
            )
            if should_save_resume:
                last_checkpoint = {
                    **checkpoint_metadata,
                    "checkpoint_role": "last_resume",
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "data_loader_generator_state": generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else []
                    ),
                }
                _atomic_torch_save(
                    checkpoint_dir / "last.pt", last_checkpoint
                )
            if is_best:
                best_checkpoint = {
                    **checkpoint_metadata,
                    "checkpoint_role": "best_evaluation",
                    "model_state": model.state_dict(),
                }
                _atomic_torch_save(
                    checkpoint_dir / "best.pt", best_checkpoint
                )

    result = {
        "best_val_accuracy": best_accuracy,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "best_selection_score": best_selection_score,
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "best_checkpoint_val_accuracy": best_checkpoint_val_accuracy,
        "epochs": epochs,
        "start_epoch": start_epoch,
        "completed_epochs": len(history),
        "resumed_from": resumed_from,
        "config_fingerprint": config_fingerprint,
        "checkpoint_dir": str(checkpoint_dir),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "full_train_samples": len(full_train_dataset),
        "full_validation_samples": len(full_val_dataset),
        "checkpoints_saved": save_checkpoints,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "class_weighting": class_weighting,
        "initialization": initialization,
        "history": history,
    }
    _atomic_write_json(checkpoint_dir / "training_summary.json", result)
    return result
