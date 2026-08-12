"""专利启发的多模态分类模型。

生产编码器是可选依赖；融合核心只依赖 PyTorch，并可通过
``forward_from_features`` 在不加载任何外部权重的情况下测试。
"""

from __future__ import annotations

import copy
import importlib
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMARTBIN_ROOT = PROJECT_ROOT.parent
DEFAULT_YOLO_WEIGHTS = SMARTBIN_ROOT / "yolo11s-seg.pt"
LOCAL_ULTRALYTICS_ROOT = (SMARTBIN_ROOT / "ultralytics").resolve()
MIN_ULTRALYTICS_VERSION = (8, 3)
MAX_ULTRALYTICS_VERSION = (9, 0)
VALID_MODALITIES = ("visual", "text", "sensor")
VALID_FUSION_MODES = ("attention", "mean")
VALID_FALLBACK_SEMANTIC_MODES = (
    "legacy_descriptions",
    "vision_only",
    "descriptions",
    "class_names",
    "shuffled_descriptions",
    "class_id_keys",
)
SENSOR_RELIABILITY_EVIDENCE = (
    "discrepancy",
    "consistency",
    "uncertainty",
    "availability",
)
SENSOR_GRID_QUALITY_DIM = 7
VISUAL_ANCHOR_EVIDENCE_DIM = 9


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _ultralytics_origin(module: Any) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        spec = getattr(module, "__spec__", None)
        value = getattr(spec, "origin", None)
    if not value or value in {"built-in", "frozen"}:
        raise RuntimeError("无法确认 ultralytics 的实际安装路径")
    return Path(value).resolve()


def _validate_external_ultralytics(module: Any) -> Any:
    """拒绝 smartBin 内旧源码，并验证受支持的外部包版本。"""

    origin = _ultralytics_origin(module)
    if _path_is_within(origin, LOCAL_ULTRALYTICS_ROOT):
        raise RuntimeError(
            "检测到已加载 smartBin/ultralytics 本地旧源码："
            f"{origin}；请启动未预加载该模块的隔离环境"
        )

    version = str(getattr(module, "__version__", ""))
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        raise RuntimeError(f"无法识别 ultralytics 版本：{version or '<missing>'}")
    version_key = (int(match.group(1)), int(match.group(2)))
    if not MIN_ULTRALYTICS_VERSION <= version_key < MAX_ULTRALYTICS_VERSION:
        raise RuntimeError(
            "ultralytics 版本不兼容："
            f"{version}（要求 >=8.3,<9；已测试 8.4.70）"
        )
    return module


def _reject_loaded_local_ultralytics_modules() -> None:
    """检测整个已加载包树，且不删除或替换其中任何模块。"""

    for name, module in tuple(sys.modules.items()):
        if name != "ultralytics" and not name.startswith("ultralytics."):
            continue
        value = getattr(module, "__file__", None)
        if not value:
            continue
        try:
            origin = Path(value).resolve()
        except (OSError, RuntimeError):
            continue
        if _path_is_within(origin, LOCAL_ULTRALYTICS_ROOT):
            raise RuntimeError(
                "检测到已加载 smartBin/ultralytics 本地旧源码模块："
                f"{name}（{origin}）；请启动未预加载该模块的隔离环境"
            )


def _import_external_ultralytics() -> Any:
    """从环境包导入 Ultralytics，同时隔离 cwd 与 smartBin 旧源码。"""

    _reject_loaded_local_ultralytics_modules()
    loaded = sys.modules.get("ultralytics")
    if loaded is not None:
        # 已加载模块绝不从 sys.modules 移除或替换；本地旧源码会明确失败。
        return _validate_external_ultralytics(loaded)

    blocked_roots = {SMARTBIN_ROOT.resolve(), Path.cwd().resolve()}
    original_sys_path = list(sys.path)
    filtered_sys_path: list[str] = []
    for entry in original_sys_path:
        try:
            resolved = Path(entry or ".").resolve()
        except (OSError, RuntimeError):
            filtered_sys_path.append(entry)
            continue
        if resolved not in blocked_roots:
            filtered_sys_path.append(entry)

    try:
        sys.path[:] = filtered_sys_path
        try:
            module = importlib.import_module("ultralytics")
        except Exception as exc:
            raise RuntimeError(
                "无法从隔离环境导入外部 ultralytics；"
                "请安装兼容版本 >=8.3,<9（已测试 8.4.70）"
            ) from exc
    finally:
        sys.path[:] = original_sys_path
    return _validate_external_ultralytics(module)


def _first_feature_tensor(value: Any) -> Tensor:
    """从 Ultralytics 层的输出中取得第一个四维特征张量。"""

    if isinstance(value, Tensor) and value.ndim == 4:
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_feature_tensor(item)
            except (TypeError, ValueError):
                continue
    raise TypeError("YOLO 特征层没有产生形如 [B,C,H,W] 的张量")


class YOLOMultiScaleEncoder(nn.Module):
    """加载本地 YOLO11s-seg，并池化分割头所用的三个主干尺度。"""

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_YOLO_WEIGHTS,
        output_dim: int = 512,
        freeze_layers: int = 5,
    ) -> None:
        super().__init__()
        weights_path = Path(weights_path).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"找不到本地 YOLO 权重：{weights_path}")

        ultralytics = _import_external_ultralytics()
        try:
            YOLO = getattr(ultralytics, "YOLO")
        except AttributeError as exc:
            raise RuntimeError(
                "外部 ultralytics 未公开 YOLO 接口；请检查安装是否完整"
            ) from exc

        try:
            detector = YOLO(str(weights_path))
        except Exception as exc:  # Ultralytics 会包装多种底层加载异常
            version = getattr(ultralytics, "__version__", "unknown")
            raise RuntimeError(
                f"ultralytics {version} 加载本地 YOLO 权重失败：{weights_path}"
            ) from exc

        self.backbone = detector.model
        layers = getattr(self.backbone, "model", None)
        if layers is None or len(layers) < 6:
            version = getattr(ultralytics, "__version__", "unknown")
            raise RuntimeError(
                f"无法识别 ultralytics {version} 的 YOLO 模型层结构"
            )

        for layer in list(layers)[:freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

        source_indices = getattr(layers[-1], "f", None)
        if not isinstance(source_indices, (list, tuple)):
            raise RuntimeError("YOLO 分割头未公开三尺度来源索引（head.f）")
        source_indices = [int(index) for index in source_indices if isinstance(index, int)]
        if len(source_indices) < 3:
            raise RuntimeError(f"YOLO 分割头尺度不足 3 个：{source_indices}")
        self.source_indices = tuple(source_indices[:3])
        self._captured: dict[int, Tensor] = {}
        self._hook_handles = []
        for index in self.source_indices:
            resolved = index if index >= 0 else len(layers) + index
            if not 0 <= resolved < len(layers):
                raise RuntimeError(f"YOLO 特征层索引越界：{index}")
            handle = layers[resolved].register_forward_hook(self._capture(index))
            self._hook_handles.append(handle)

        # YOLO 不同规模的通道数不同，LazyLinear 在首次本地前向时确定输入维度。
        self.projections = nn.ModuleList(
            [nn.LazyLinear(output_dim) for _ in self.source_indices]
        )

    def _capture(self, index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self._captured[index] = _first_feature_tensor(output)

        return hook

    def forward(self, images: Tensor) -> list[Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("视觉输入必须为 [B,3,H,W]")
        self._captured.clear()
        _ = self.backbone(images)
        missing = [index for index in self.source_indices if index not in self._captured]
        if missing:
            raise RuntimeError(f"未捕获到 YOLO 三尺度特征，缺失层：{missing}")
        features: list[Tensor] = []
        for index, projection in zip(self.source_indices, self.projections):
            feature_map = self._captured[index]
            pooled = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
            features.append(projection(pooled))
        return features


class CLIPPrototypeEncoder(nn.Module):
    """CLIP ViT-B/16 文本塔；一次编码全部候选类别描述。"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze_layers: int = 3,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            transformers = importlib.import_module("transformers")
            encoder_class = getattr(transformers, "CLIPTextModelWithProjection")
            tokenizer_class = getattr(transformers, "AutoTokenizer")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "CLIP 文本编码器需要 transformers；请安装 transformers 后重试"
            ) from exc
        try:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            self.text_model = encoder_class.from_pretrained(
                model_name, local_files_only=local_files_only
            )
        except Exception as exc:
            locality = "本地缓存" if local_files_only else "模型仓库或本地缓存"
            raise RuntimeError(
                f"无法从{locality}加载 CLIP ViT-B/16 文本编码器：{model_name}"
            ) from exc

        layers = self.text_model.text_model.encoder.layers
        for layer in list(layers)[:freeze_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        self._token_cache: dict[
            tuple[str, ...], dict[str, Tensor]
        ] = {}
        self._embedding_cache: dict[tuple[str, ...], Tensor] = {}

    def forward(self, descriptions: Sequence[str], device: torch.device) -> Tensor:
        if not descriptions:
            raise ValueError("候选类别描述不能为空")
        cache_key = tuple(str(description) for description in descriptions)
        frozen_eval = (
            not self.training
            and not any(parameter.requires_grad for parameter in self.parameters())
        )
        cached_embedding = self._embedding_cache.get(cache_key)
        if frozen_eval and cached_embedding is not None:
            return cached_embedding.to(device, non_blocking=True)
        cached = self._token_cache.get(cache_key)
        if cached is None:
            tokenized = self.tokenizer(
                list(cache_key),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            cached = {
                name: value.detach().cpu()
                for name, value in tokenized.items()
            }
            self._token_cache[cache_key] = cached
        encoded = {
            name: value.to(device, non_blocking=True)
            for name, value in cached.items()
        }
        output = self.text_model(**encoded)
        prototypes = output.text_embeds
        if prototypes.shape[-1] != 512:
            raise RuntimeError(f"CLIP 文本特征应为 512 维，实际为 {prototypes.shape[-1]}")
        prototypes = F.normalize(prototypes, dim=-1)
        if frozen_eval:
            self._embedding_cache[cache_key] = prototypes.detach().cpu()
        return prototypes


class SensorEncoder(nn.Module):
    """4×4 传感器阵列编码器：归一化、上采样、1×1 卷积和池化。"""

    def __init__(self, output_dim: int = 512, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.projection = nn.Conv2d(1, output_dim, kernel_size=1)
        nn.init.kaiming_normal_(
            self.projection.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.projection.bias)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, sensor_grid: Tensor) -> Tensor:
        if sensor_grid.ndim != 4 or tuple(sensor_grid.shape[1:]) != (1, 4, 4):
            raise ValueError("传感器输入必须为 [B,1,4,4]")
        if not torch.isfinite(sensor_grid).all():
            raise ValueError("传感器输入包含 NaN 或无穷值")
        values = sensor_grid.float()
        if self.normalize:
            mean = values.mean(dim=(2, 3), keepdim=True)
            std = values.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
            values = (values - mean) / std
        values = F.interpolate(
            values, size=(16, 16), mode="bilinear", align_corners=False
        )
        values = self.activation(self.projection(values))
        return F.adaptive_avg_pool2d(values, output_size=1).flatten(1)


def sensor_grid_quality_features(sensor_grid: Tensor) -> Tensor:
    """Extract differentiable raw-grid statistics for reliability estimation."""

    if sensor_grid.ndim != 4 or tuple(sensor_grid.shape[1:]) != (1, 4, 4):
        raise ValueError("sensor_grid must have shape [B,1,4,4]")
    if not torch.isfinite(sensor_grid).all():
        raise ValueError("sensor_grid contains NaN or infinite values")
    values = sensor_grid.float()
    flattened = values.flatten(start_dim=1)
    mean = flattened.mean(dim=1)
    std = flattened.std(dim=1, unbiased=False)
    dynamic_range = flattened.amax(dim=1) - flattened.amin(dim=1)
    horizontal_gradient = (
        values[:, :, :, 1:] - values[:, :, :, :-1]
    ).abs().mean(dim=(1, 2, 3))
    vertical_gradient = (
        values[:, :, 1:, :] - values[:, :, :-1, :]
    ).abs().mean(dim=(1, 2, 3))
    saturation = (
        (values <= 0.02) | (values >= 0.98)
    ).float().mean(dim=(1, 2, 3))
    flatness = torch.exp(-10.0 * std)
    return torch.stack(
        (
            mean,
            std,
            dynamic_range,
            horizontal_gradient,
            vertical_gradient,
            saturation,
            flatness,
        ),
        dim=1,
    )


SENSOR_CROSS_MODAL_QUALITY_DIM = 22


def cross_modal_sensor_grid_quality_features(
    predicted_grid: Tensor,
    observed_grid: Tensor,
    availability: Tensor,
) -> Tensor:
    """Combine raw-grid statistics with visual prediction discrepancies."""

    expected = (1, 4, 4)
    if (
        predicted_grid.ndim != 4
        or observed_grid.ndim != 4
        or tuple(predicted_grid.shape[1:]) != expected
        or tuple(observed_grid.shape[1:]) != expected
        or predicted_grid.shape[0] != observed_grid.shape[0]
    ):
        raise ValueError("predicted_grid and observed_grid must have shape [B,1,4,4]")
    availability = availability.to(
        device=observed_grid.device,
        dtype=observed_grid.dtype,
    )
    if availability.ndim == 1:
        availability = availability.unsqueeze(1)
    if tuple(availability.shape) != (observed_grid.shape[0], 1):
        raise ValueError("availability must have shape [B,1] or [B]")

    predicted = predicted_grid.to(
        device=observed_grid.device,
        dtype=observed_grid.dtype,
    )
    difference = observed_grid - predicted
    absolute = difference.abs()
    mean_absolute = absolute.mean(dim=(1, 2, 3)).unsqueeze(1)
    root_mean_square = difference.square().mean(dim=(1, 2, 3)).sqrt().unsqueeze(1)
    maximum_absolute = absolute.amax(dim=(1, 2, 3)).unsqueeze(1)
    absolute_mean_bias = difference.mean(dim=(1, 2, 3)).abs().unsqueeze(1)
    cosine = F.cosine_similarity(
        observed_grid.flatten(1),
        predicted.flatten(1),
        dim=1,
        eps=1e-6,
    ).unsqueeze(1)
    horizontal_difference = (
        (
            (observed_grid[..., 1:] - observed_grid[..., :-1])
            - (predicted[..., 1:] - predicted[..., :-1])
        )
        .abs()
        .mean(dim=(1, 2, 3))
        .unsqueeze(1)
    )
    vertical_difference = (
        (
            (observed_grid[..., 1:, :] - observed_grid[..., :-1, :])
            - (predicted[..., 1:, :] - predicted[..., :-1, :])
        )
        .abs()
        .mean(dim=(1, 2, 3))
        .unsqueeze(1)
    )
    features = torch.cat(
        (
            sensor_grid_quality_features(observed_grid),
            sensor_grid_quality_features(predicted),
            mean_absolute,
            root_mean_square,
            maximum_absolute,
            absolute_mean_bias,
            cosine,
            horizontal_difference,
            vertical_difference,
            availability,
        ),
        dim=1,
    )
    if features.shape[1] != SENSOR_CROSS_MODAL_QUALITY_DIM:
        raise RuntimeError("cross-modal sensor quality feature dimension mismatch")
    return features


class ModalityAdapter(nn.Sequential):
    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
        )
        nn.init.xavier_normal_(self[0].weight)
        nn.init.zeros_(self[0].bias)


class MultimodalFusionClassifier(nn.Module):
    """无目标标签输入的三模态融合分类器。"""

    def __init__(
        self,
        num_classes: int,
        class_descriptions: Sequence[str] | None = None,
        *,
        visual_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        fallback_text_encoder: nn.Module | None = None,
        sensor_encoder: nn.Module | None = None,
        feature_dim: int = 512,
        attention_dim: int = 128,
        attention_heads: int = 8,
        dropout: float = 0.3,
        film_gamma_init: float = 1.0,
        film_beta_init: float = 0.1,
        scale_prior: Sequence[float] = (0.3, 0.5, 0.2),
        text_temperature: float = 0.07,
        sensor_log_variance_min: float = -6.0,
        sensor_log_variance_max: float = 2.0,
        sensor_reliability_hidden_dim: int = 32,
        sensor_reliability_init: float = 0.85,
        enabled_modalities: Sequence[str] = VALID_MODALITIES,
        fusion_mode: str = "attention",
        film_enabled: bool = True,
        sensor_reliability_enabled: bool = True,
        constant_sensor_reliability_gate: bool = False,
        sensor_grid_quality_enabled: bool = False,
        sensor_grid_quality_hidden_dim: int = 16,
        sensor_cross_modal_quality_enabled: bool = False,
        sensor_grid_decoder_hidden_dim: int = 64,
        sensor_cross_modal_quality_hidden_dim: int = 32,
        detach_predicted_grid_for_reliability: bool = True,
        visual_anchor_enabled: bool = False,
        visual_anchor_gate_hidden_dim: int = 32,
        visual_anchor_gate_init: float = 0.1,
        visual_anchor_gate_max: float = 1.0,
        fallback_semantic_mode: str = "legacy_descriptions",
        fallback_class_names: Sequence[str] | None = None,
        fallback_semantic_descriptions: Sequence[str] | None = None,
        fallback_semantic_shuffle_seed: int = 2026,
        fallback_semantic_memory_enabled: bool = False,
        fallback_semantic_direct_latent: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes 必须至少为 2")
        if feature_dim != 512:
            raise ValueError("专利配置要求三模态特征维度为 512")
        if attention_dim % attention_heads:
            raise ValueError("attention_dim 必须能被 attention_heads 整除")
        if len(scale_prior) != 3 or any(value <= 0 for value in scale_prior):
            raise ValueError("scale_prior 必须包含三个正数")

        if sensor_log_variance_min >= sensor_log_variance_max:
            raise ValueError("sensor_log_variance_min must be < sensor_log_variance_max")
        if not 0.0 < sensor_reliability_init < 1.0:
            raise ValueError("sensor_reliability_init must be in (0,1)")
        if not math.isfinite(film_gamma_init) or film_gamma_init <= 0.0:
            raise ValueError("film_gamma_init must be finite and > 0")
        if not math.isfinite(film_beta_init):
            raise ValueError("film_beta_init must be finite")
        modalities = tuple(
            dict.fromkeys(str(name).strip().lower() for name in enabled_modalities)
        )
        unknown_modalities = sorted(set(modalities) - set(VALID_MODALITIES))
        if unknown_modalities:
            raise ValueError(f"unknown enabled modalities: {unknown_modalities}")
        if not modalities:
            raise ValueError("at least one modality must be enabled")
        if not {"visual", "sensor"}.intersection(modalities):
            raise ValueError(
                "at least one sample-observation modality (visual or sensor) "
                "must be enabled"
            )
        fusion_mode = str(fusion_mode).strip().lower()
        if fusion_mode not in VALID_FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {list(VALID_FUSION_MODES)}"
            )
        if sensor_reliability_enabled and not {
            "visual",
            "text",
            "sensor",
        }.issubset(
            modalities
        ):
            raise ValueError(
                "sensor_reliability_enabled requires visual, text, and sensor "
                "modalities"
            )
        if sensor_grid_quality_enabled and not sensor_reliability_enabled:
            raise ValueError(
                "sensor_grid_quality_enabled requires sensor reliability"
            )
        if sensor_cross_modal_quality_enabled and not sensor_reliability_enabled:
            raise ValueError(
                "sensor_cross_modal_quality_enabled requires sensor reliability"
            )
        if visual_anchor_enabled and "visual" not in modalities:
            raise ValueError("visual_anchor_enabled requires the visual modality")
        if not 0.0 < visual_anchor_gate_init < visual_anchor_gate_max:
            raise ValueError(
                "visual_anchor_gate_init must be in (0, visual_anchor_gate_max)"
            )
        if not 0.0 < visual_anchor_gate_max <= 1.0:
            raise ValueError("visual_anchor_gate_max must be in (0,1]")
        fallback_semantic_mode = str(fallback_semantic_mode).strip().lower()
        if fallback_semantic_mode not in VALID_FALLBACK_SEMANTIC_MODES:
            raise ValueError(
                "fallback_semantic_mode must be one of "
                f"{list(VALID_FALLBACK_SEMANTIC_MODES)}"
            )

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.enabled_modalities = modalities
        self.fusion_mode = fusion_mode
        self.film_enabled = bool(film_enabled)
        self.sensor_reliability_enabled = bool(sensor_reliability_enabled)
        self.constant_sensor_reliability_gate = bool(
            constant_sensor_reliability_gate
        )
        self.sensor_grid_quality_enabled = bool(sensor_grid_quality_enabled)
        self.sensor_cross_modal_quality_enabled = bool(
            sensor_cross_modal_quality_enabled
        )
        self.detach_predicted_grid_for_reliability = bool(
            detach_predicted_grid_for_reliability
        )
        self.visual_anchor_enabled = bool(visual_anchor_enabled)
        self.visual_anchor_gate_max = float(visual_anchor_gate_max)
        self.visual_anchor_frozen = False
        self._frozen_module_names: set[str] = set()
        self.fallback_semantic_mode = fallback_semantic_mode
        self.fallback_semantic_memory_enabled = bool(
            fallback_semantic_memory_enabled
        )
        self.fallback_semantic_direct_latent = bool(
            fallback_semantic_direct_latent
        )
        if (
            self.fallback_semantic_direct_latent
            and not self.fallback_semantic_memory_enabled
        ):
            raise ValueError(
                "fallback_semantic_direct_latent requires semantic memory"
            )
        self.fallback_semantic_shuffle_seed = int(
            fallback_semantic_shuffle_seed
        )
        modality_mask = torch.tensor(
            [name in modalities for name in VALID_MODALITIES],
            dtype=torch.float32,
        )
        self.register_buffer("modality_mask", modality_mask, persistent=False)
        self.register_buffer(
            "sensor_reliability_evidence_mask",
            torch.ones(len(SENSOR_RELIABILITY_EVIDENCE), dtype=torch.float32),
            persistent=False,
        )
        self.class_descriptions = (
            tuple(str(item) for item in class_descriptions)
            if class_descriptions is not None
            else None
        )
        if self.class_descriptions is not None and len(self.class_descriptions) != num_classes:
            raise ValueError("类别描述数量必须与 num_classes 相同")
        self.fallback_class_names = (
            tuple(str(item) for item in fallback_class_names)
            if fallback_class_names is not None
            else tuple(f"class {index}" for index in range(num_classes))
        )
        self.fallback_semantic_descriptions = (
            tuple(str(item) for item in fallback_semantic_descriptions)
            if fallback_semantic_descriptions is not None
            else (
                self.class_descriptions
                if self.class_descriptions is not None
                else self.fallback_class_names
            )
        )
        if len(self.fallback_class_names) != num_classes:
            raise ValueError("fallback_class_names must match num_classes")
        if feature_dim < num_classes:
            raise ValueError("feature_dim must accommodate orthogonal class-ID keys")
        self.register_buffer(
            "fallback_class_id_keys",
            torch.eye(num_classes, feature_dim, dtype=torch.float32),
            persistent=False,
        )
        if (
            self.fallback_semantic_descriptions is None
            or len(self.fallback_semantic_descriptions) != num_classes
        ):
            raise ValueError(
                "fallback_semantic_descriptions must match num_classes"
            )
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.fallback_text_encoder = fallback_text_encoder
        self.sensor_encoder = sensor_encoder
        self.text_temperature = float(text_temperature)

        self.scale_projections = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(3)]
        )
        for projection in self.scale_projections:
            nn.init.eye_(projection.weight)
            nn.init.zeros_(projection.bias)

        self.visual_adapter = ModalityAdapter(feature_dim, dropout)
        self.text_adapter = ModalityAdapter(feature_dim, dropout)
        self.sensor_adapter = ModalityAdapter(feature_dim, dropout)

        self.sensor_log_variance_min = float(sensor_log_variance_min)
        self.sensor_log_variance_max = float(sensor_log_variance_max)
        reliability_hidden_dim = max(2, int(sensor_reliability_hidden_dim))
        sensor_prediction_dim = feature_dim * 2
        self.sensor_mean_predictor = nn.Sequential(
            nn.Linear(sensor_prediction_dim, feature_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        self.sensor_logvar_predictor = nn.Sequential(
            nn.Linear(sensor_prediction_dim, feature_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        if self.fallback_semantic_memory_enabled:
            self.fallback_query_projection = nn.Linear(
                feature_dim,
                feature_dim,
                bias=False,
            )
            nn.init.eye_(self.fallback_query_projection.weight)
            self.fallback_sensor_value_bank = nn.Parameter(
                torch.empty(num_classes, feature_dim)
            )
            nn.init.normal_(
                self.fallback_sensor_value_bank,
                mean=0.0,
                std=0.02,
            )
            self.fallback_sensor_center_bank = nn.Parameter(
                torch.zeros(num_classes, feature_dim),
                requires_grad=False,
            )
        else:
            self.fallback_query_projection = None
            self.register_parameter("fallback_sensor_value_bank", None)
            self.register_parameter("fallback_sensor_center_bank", None)
        self.sensor_reliability_predictor = nn.Sequential(
            nn.Linear(4, reliability_hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(reliability_hidden_dim, 1),
        )
        quality_hidden_dim = max(2, int(sensor_grid_quality_hidden_dim))
        self.sensor_grid_quality_predictor = (
            nn.Sequential(
                nn.Linear(SENSOR_GRID_QUALITY_DIM, quality_hidden_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(quality_hidden_dim, 1),
            )
            if self.sensor_grid_quality_enabled
            else None
        )
        grid_decoder_hidden_dim = max(4, int(sensor_grid_decoder_hidden_dim))
        self.sensor_grid_decoder = (
            nn.Sequential(
                nn.Linear(feature_dim * 3, grid_decoder_hidden_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(grid_decoder_hidden_dim, 16),
            )
            if self.sensor_cross_modal_quality_enabled
            else None
        )
        cross_modal_hidden_dim = max(
            4,
            int(sensor_cross_modal_quality_hidden_dim),
        )
        self.sensor_cross_modal_quality_predictor = (
            nn.Sequential(
                nn.Linear(
                    SENSOR_CROSS_MODAL_QUALITY_DIM,
                    cross_modal_hidden_dim,
                ),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(cross_modal_hidden_dim, 1),
            )
            if self.sensor_cross_modal_quality_enabled
            else None
        )
        for module in (
            self.sensor_mean_predictor,
            self.sensor_logvar_predictor,
            self.sensor_reliability_predictor,
            self.sensor_grid_quality_predictor,
            self.sensor_grid_decoder,
            self.sensor_cross_modal_quality_predictor,
        ):
            if module is None:
                continue
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.sensor_logvar_predictor[-1].weight)
        nn.init.constant_(self.sensor_logvar_predictor[-1].bias, -2.0)
        nn.init.zeros_(self.sensor_reliability_predictor[-1].weight)
        reliability_logit = math.log(
            sensor_reliability_init / (1.0 - sensor_reliability_init)
        )
        nn.init.constant_(self.sensor_reliability_predictor[-1].bias, reliability_logit)
        if self.sensor_grid_quality_predictor is not None:
            nn.init.zeros_(self.sensor_grid_quality_predictor[-1].weight)
            nn.init.zeros_(self.sensor_grid_quality_predictor[-1].bias)
        if self.sensor_cross_modal_quality_predictor is not None:
            nn.init.zeros_(self.sensor_cross_modal_quality_predictor[-1].weight)
            nn.init.zeros_(self.sensor_cross_modal_quality_predictor[-1].bias)

        # 三种模态各自一组 FiLM 参数；初始化严格对应 γ=1、β=0.1。
        self.film_gamma = nn.Parameter(
            torch.full((3, feature_dim), float(film_gamma_init))
        )
        self.film_beta = nn.Parameter(
            torch.full((3, feature_dim), float(film_beta_init))
        )

        self.down_projection = nn.Linear(feature_dim, attention_dim)
        nn.init.xavier_normal_(self.down_projection.weight)
        nn.init.zeros_(self.down_projection.bias)
        self.modality_attention = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.up_projection = nn.Linear(attention_dim, feature_dim)
        nn.init.xavier_normal_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

        prior = torch.as_tensor(scale_prior, dtype=torch.float32)
        prior = prior / prior.sum()
        self.scale_prior_logits = nn.Parameter(prior.log())
        self.dynamic_scale_gate = nn.Linear(feature_dim, 3)
        nn.init.zeros_(self.dynamic_scale_gate.weight)
        nn.init.zeros_(self.dynamic_scale_gate.bias)

        self.output_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        if self.visual_anchor_enabled:
            self.visual_anchor_scale_projections = copy.deepcopy(
                self.scale_projections
            )
            self.visual_anchor_scale_prior_logits = nn.Parameter(
                self.scale_prior_logits.detach().clone()
            )
            self.visual_anchor_adapter = copy.deepcopy(self.visual_adapter)
            self.visual_anchor_dynamic_scale_gate = copy.deepcopy(
                self.dynamic_scale_gate
            )
            self.visual_anchor_output_norm = copy.deepcopy(self.output_norm)
            self.visual_anchor_classifier = copy.deepcopy(self.classifier)
            gate_hidden_dim = max(2, int(visual_anchor_gate_hidden_dim))
            self.visual_anchor_gate = nn.Sequential(
                nn.Linear(VISUAL_ANCHOR_EVIDENCE_DIM, gate_hidden_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, 1),
            )
            for layer in self.visual_anchor_gate.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.visual_anchor_gate[-1].weight)
            normalized_init = visual_anchor_gate_init / visual_anchor_gate_max
            gate_logit = math.log(normalized_init / (1.0 - normalized_init))
            nn.init.constant_(self.visual_anchor_gate[-1].bias, gate_logit)
        else:
            self.visual_anchor_scale_projections = None
            self.register_parameter("visual_anchor_scale_prior_logits", None)
            self.visual_anchor_adapter = None
            self.visual_anchor_dynamic_scale_gate = None
            self.visual_anchor_output_norm = None
            self.visual_anchor_classifier = None
            self.visual_anchor_gate = None
        self.legacy_sensor_passthrough = False
        self.last_aux: dict[str, Tensor] = {}
        self._freeze_inactive_parameters()

    @classmethod
    def create_production(
        cls,
        num_classes: int,
        class_descriptions: Sequence[str],
        *,
        yolo_weights: str | Path = DEFAULT_YOLO_WEIGHTS,
        clip_model: str = "openai/clip-vit-base-patch16",
        clip_local_files_only: bool = False,
        freeze_yolo_layers: int = 5,
        freeze_clip_text_layers: int = 3,
        fallback_text_encoder_independent: bool = False,
        sensor_standardize: bool = True,
        **kwargs: Any,
    ) -> "MultimodalFusionClassifier":
        """构建使用本地 YOLO 与 CLIP ViT-B/16 的生产模型。"""

        modalities = tuple(
            str(name).strip().lower()
            for name in kwargs.get("enabled_modalities", VALID_MODALITIES)
        )
        visual = (
            YOLOMultiScaleEncoder(
                yolo_weights, freeze_layers=freeze_yolo_layers
            )
            if "visual" in modalities
            else None
        )
        text = (
            CLIPPrototypeEncoder(
                clip_model,
                freeze_layers=freeze_clip_text_layers,
                local_files_only=clip_local_files_only,
            )
            if "text" in modalities
            else None
        )
        fallback_text = (
            CLIPPrototypeEncoder(
                clip_model,
                freeze_layers=freeze_clip_text_layers,
                local_files_only=clip_local_files_only,
            )
            if "text" in modalities and fallback_text_encoder_independent
            else None
        )
        if fallback_text is not None:
            for parameter in fallback_text.parameters():
                parameter.requires_grad = False
        sensor = (
            SensorEncoder(output_dim=512, normalize=sensor_standardize)
            if "sensor" in modalities
            else None
        )
        return cls(
            num_classes,
            class_descriptions,
            visual_encoder=visual,
            text_encoder=text,
            fallback_text_encoder=fallback_text,
            sensor_encoder=sensor,
            **kwargs,
        )

    @staticmethod
    def _validate_features(name: str, features: Tensor, batch_size: int | None = None) -> None:
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError(f"{name} 必须为 [B,512]")
        if batch_size is not None and features.shape[0] != batch_size:
            raise ValueError(f"{name} 的 batch 大小不一致")

    def set_legacy_sensor_passthrough(self, enabled: bool = True) -> None:
        self.legacy_sensor_passthrough = bool(enabled)

    def set_disabled_sensor_reliability_evidence(
        self,
        names: Iterable[str] = (),
    ) -> None:
        """Mask selected reliability-head inputs for inference interventions."""

        disabled = {str(name).strip().lower() for name in names}
        unknown = sorted(disabled - set(SENSOR_RELIABILITY_EVIDENCE))
        if unknown:
            raise ValueError(f"unknown sensor reliability evidence: {unknown}")
        mask = torch.tensor(
            [name not in disabled for name in SENSOR_RELIABILITY_EVIDENCE],
            device=self.sensor_reliability_evidence_mask.device,
            dtype=self.sensor_reliability_evidence_mask.dtype,
        )
        self.sensor_reliability_evidence_mask.copy_(mask)

    @property
    def visual_modality_enabled(self) -> bool:
        return "visual" in self.enabled_modalities

    @property
    def sensor_modality_enabled(self) -> bool:
        return "sensor" in self.enabled_modalities

    @property
    def text_modality_enabled(self) -> bool:
        return "text" in self.enabled_modalities

    def set_fallback_semantic_mode(self, mode: str) -> None:
        """Switch fallback conditioning without changing classifier text input."""

        normalized = str(mode).strip().lower()
        if normalized not in VALID_FALLBACK_SEMANTIC_MODES:
            raise ValueError(
                "fallback semantic mode must be one of "
                f"{list(VALID_FALLBACK_SEMANTIC_MODES)}"
            )
        self.fallback_semantic_mode = normalized

    def initialize_fallback_sensor_value_bank(
        self,
        values: Tensor,
        *,
        trainable: bool = False,
    ) -> None:
        """Load train-split archived sensor-state centers into semantic memory."""

        if self.fallback_sensor_center_bank is None:
            raise RuntimeError("fallback semantic memory is not enabled")
        expected = (self.num_classes, self.feature_dim)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"fallback sensor values must have shape {expected}"
            )
        if not torch.isfinite(values).all():
            raise ValueError("fallback sensor values contain non-finite values")
        with torch.no_grad():
            self.fallback_sensor_center_bank.copy_(
                values.to(
                    device=self.fallback_sensor_center_bank.device,
                    dtype=self.fallback_sensor_center_bank.dtype,
                )
            )
        self.fallback_sensor_center_bank.requires_grad = bool(trainable)

    def fallback_semantic_prompts(
        self,
        mode: str | None = None,
    ) -> tuple[str, ...] | None:
        """Return class-bound prompts used only by the fallback predictor."""

        selected = (
            self.fallback_semantic_mode
            if mode is None
            else str(mode).strip().lower()
        )
        if selected not in VALID_FALLBACK_SEMANTIC_MODES:
            raise ValueError(
                "fallback semantic mode must be one of "
                f"{list(VALID_FALLBACK_SEMANTIC_MODES)}"
            )
        if selected in {"legacy_descriptions", "vision_only", "class_id_keys"}:
            return None
        if selected == "class_names":
            return tuple(
                f"a photo of {name}" for name in self.fallback_class_names
            )
        prompts = tuple(
            f"a photo of {name}. {description}"
            for name, description in zip(
                self.fallback_class_names,
                self.fallback_semantic_descriptions,
            )
        )
        if selected == "shuffled_descriptions":
            offset = abs(self.fallback_semantic_shuffle_seed) % (
                self.num_classes - 1
            ) + 1
            return prompts[offset:] + prompts[:offset]
        return prompts

    def _freeze_inactive_parameters(self) -> None:
        def freeze(module: nn.Module | None) -> None:
            if module is None:
                return
            for parameter in module.parameters():
                parameter.requires_grad = False

        if not self.visual_modality_enabled:
            freeze(self.visual_encoder)
            freeze(self.scale_projections)
            freeze(self.visual_adapter)
            freeze(self.dynamic_scale_gate)
            self.scale_prior_logits.requires_grad = False
        if not self.text_modality_enabled:
            freeze(self.text_encoder)
            freeze(self.fallback_text_encoder)
            freeze(self.text_adapter)
        if not self.sensor_modality_enabled:
            freeze(self.sensor_encoder)
            freeze(self.sensor_adapter)
        if not self.sensor_reliability_enabled:
            freeze(self.sensor_mean_predictor)
            freeze(self.sensor_logvar_predictor)
            freeze(self.sensor_reliability_predictor)
        if not self.fallback_semantic_memory_enabled:
            freeze(self.fallback_query_projection)
        if not self.sensor_grid_quality_enabled:
            freeze(self.sensor_grid_quality_predictor)
        if not self.sensor_cross_modal_quality_enabled:
            freeze(self.sensor_grid_decoder)
            freeze(self.sensor_cross_modal_quality_predictor)
        if self.fusion_mode != "attention":
            freeze(self.down_projection)
            freeze(self.modality_attention)
            freeze(self.up_projection)
        if not self.film_enabled:
            self.film_gamma.requires_grad = False
            self.film_beta.requires_grad = False

    def freeze_visual_anchor(self) -> None:
        """Freeze the pretrained visual expert while fusion learns corrections."""

        if not self.visual_anchor_enabled:
            raise RuntimeError("visual anchor is not enabled")

        def freeze(module: nn.Module | None) -> None:
            if module is None:
                return
            for parameter in module.parameters():
                parameter.requires_grad = False

        freeze(self.visual_encoder)
        freeze(self.visual_anchor_scale_projections)
        freeze(self.visual_anchor_adapter)
        freeze(self.visual_anchor_dynamic_scale_gate)
        freeze(self.visual_anchor_output_norm)
        freeze(self.visual_anchor_classifier)
        if self.visual_anchor_scale_prior_logits is not None:
            self.visual_anchor_scale_prior_logits.requires_grad = False
        self.visual_anchor_frozen = True
        self._set_frozen_visual_anchor_eval()

    def _set_frozen_visual_anchor_eval(self) -> None:
        if not self.visual_anchor_frozen:
            return
        for module in (
            self.visual_encoder,
            self.visual_anchor_scale_projections,
            self.visual_anchor_adapter,
            self.visual_anchor_dynamic_scale_gate,
            self.visual_anchor_output_norm,
            self.visual_anchor_classifier,
        ):
            if module is not None:
                module.eval()

    def freeze_modules(self, names: Sequence[str]) -> None:
        """Freeze named top-level modules for controlled ablations."""

        for raw_name in names:
            name = str(raw_name).strip()
            module = getattr(self, name, None)
            if not name or not isinstance(module, nn.Module):
                raise ValueError(f"unknown or non-module model component: {name!r}")
            for parameter in module.parameters():
                parameter.requires_grad = False
            module.eval()
            self._frozen_module_names.add(name)

    def _set_explicitly_frozen_modules_eval(self) -> None:
        for name in self._frozen_module_names:
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.eval()

    def train(self, mode: bool = True) -> "MultimodalFusionClassifier":
        super().train(mode)
        self._set_frozen_visual_anchor_eval()
        self._set_explicitly_frozen_modules_eval()
        return self

    def active_parameters(self) -> Iterable[nn.Parameter]:
        """Yield parameters that participate in this ablation's forward path."""

        modules: list[nn.Module | None] = [self.output_norm, self.classifier]
        if self.visual_modality_enabled:
            modules.extend(
                (
                    self.visual_encoder,
                    self.scale_projections,
                    self.visual_adapter,
                    self.dynamic_scale_gate,
                )
            )
        if self.text_modality_enabled:
            modules.extend(
                (
                    self.text_encoder,
                    self.fallback_text_encoder,
                    self.text_adapter,
                )
            )
        if self.sensor_modality_enabled:
            modules.extend((self.sensor_encoder, self.sensor_adapter))
        if self.sensor_reliability_enabled:
            modules.extend(
                (
                    self.sensor_mean_predictor,
                    self.sensor_logvar_predictor,
                    self.sensor_reliability_predictor,
                )
            )
        if self.fallback_semantic_memory_enabled:
            modules.append(self.fallback_query_projection)
        if self.sensor_grid_quality_enabled:
            modules.append(self.sensor_grid_quality_predictor)
        if self.sensor_cross_modal_quality_enabled:
            modules.extend(
                (
                    self.sensor_grid_decoder,
                    self.sensor_cross_modal_quality_predictor,
                )
            )
        if self.fusion_mode == "attention":
            modules.extend(
                (
                    self.down_projection,
                    self.modality_attention,
                    self.up_projection,
                )
            )
        if self.visual_anchor_enabled:
            modules.extend(
                (
                    self.visual_anchor_scale_projections,
                    self.visual_anchor_adapter,
                    self.visual_anchor_dynamic_scale_gate,
                    self.visual_anchor_output_norm,
                    self.visual_anchor_classifier,
                    self.visual_anchor_gate,
                )
            )

        seen: set[int] = set()
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                identifier = id(parameter)
                if identifier not in seen:
                    seen.add(identifier)
                    yield parameter
        direct_parameters = []
        if self.visual_modality_enabled:
            direct_parameters.append(self.scale_prior_logits)
        if self.film_enabled:
            direct_parameters.extend((self.film_gamma, self.film_beta))
        if (
            self.fallback_semantic_memory_enabled
            and self.fallback_sensor_value_bank is not None
        ):
            direct_parameters.append(self.fallback_sensor_value_bank)
        if (
            self.fallback_semantic_memory_enabled
            and self.fallback_sensor_center_bank is not None
        ):
            direct_parameters.append(self.fallback_sensor_center_bank)
        if (
            self.visual_anchor_enabled
            and self.visual_anchor_scale_prior_logits is not None
        ):
            direct_parameters.append(self.visual_anchor_scale_prior_logits)
        for parameter in direct_parameters:
            identifier = id(parameter)
            if identifier not in seen:
                seen.add(identifier)
                yield parameter

    def _prototype_context(
        self, visual: Tensor, sensor: Tensor, prototypes: Tensor
    ) -> tuple[Tensor, Tensor]:
        if prototypes.ndim != 2 or tuple(prototypes.shape) != (self.num_classes, 512):
            raise ValueError(
                f"text_prototypes 必须为 [{self.num_classes},512]，"
                f"实际为 {list(prototypes.shape)}"
            )
        # 查询仅来自观测模态，不接受真实标签；所有候选描述共同参与 softmax。
        query = F.normalize(visual + sensor, dim=-1)
        bank = F.normalize(prototypes, dim=-1)
        scores = query @ bank.transpose(0, 1)
        weights = torch.softmax(scores / self.text_temperature, dim=-1)
        return weights @ prototypes, weights

    def _visual_prototype_context(
        self, visual: Tensor, prototypes: Tensor
    ) -> tuple[Tensor, Tensor]:
        if prototypes.ndim != 2 or tuple(prototypes.shape) != (self.num_classes, 512):
            raise ValueError(
                f"text_prototypes must have shape [{self.num_classes},512], "
                f"got {list(prototypes.shape)}"
            )
        query = F.normalize(visual, dim=-1)
        bank = F.normalize(prototypes, dim=-1)
        scores = query @ bank.transpose(0, 1)
        weights = torch.softmax(scores / self.text_temperature, dim=-1)
        return weights @ prototypes, weights

    @staticmethod
    def _prepare_sensor_availability(
        sensor_available: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if sensor_available is None:
            return torch.ones(batch_size, 1, device=device, dtype=dtype)
        availability = torch.as_tensor(sensor_available, device=device, dtype=dtype)
        if availability.ndim == 1:
            availability = availability.unsqueeze(1)
        if tuple(availability.shape) != (batch_size, 1):
            raise ValueError("sensor_available must have shape [B,1] or [B]")
        if not torch.isfinite(availability).all():
            raise ValueError("sensor_available contains NaN or infinite values")
        return availability.clamp(0.0, 1.0)

    def _predict_sensor_latent(
        self, visual: Tensor, visual_text_context: Tensor
    ) -> tuple[Tensor, Tensor]:
        context = torch.cat((visual, visual_text_context), dim=-1)
        predicted_sensor = self.sensor_mean_predictor(context)
        raw_log_variance = self.sensor_logvar_predictor(context)
        sensor_log_variance = self.sensor_log_variance_min + (
            self.sensor_log_variance_max - self.sensor_log_variance_min
        ) * torch.sigmoid(raw_log_variance)
        return predicted_sensor, sensor_log_variance

    def _estimate_sensor_reliability(
        self,
        actual_sensor: Tensor,
        predicted_sensor: Tensor,
        sensor_log_variance: Tensor,
        sensor_available: Tensor,
        sensor_quality_features: Tensor | None,
        sensor_cross_modal_quality_features: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        precision = torch.exp(-sensor_log_variance)
        discrepancy = ((actual_sensor - predicted_sensor).pow(2) * precision).mean(
            dim=-1, keepdim=True
        )
        discrepancy = discrepancy.clamp(min=0.0, max=1e6)
        uncertainty = torch.exp(sensor_log_variance).mean(dim=-1, keepdim=True)
        uncertainty = uncertainty.clamp(min=0.0, max=1e6)
        cosine = F.cosine_similarity(
            actual_sensor, predicted_sensor, dim=-1, eps=1e-6
        ).unsqueeze(1)
        consistency = ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)
        reliability_features = torch.cat(
            (
                torch.log1p(discrepancy),
                consistency,
                torch.log1p(uncertainty),
                sensor_available,
            ),
            dim=-1,
        )
        reliability_features = reliability_features * (
            self.sensor_reliability_evidence_mask.to(
                device=reliability_features.device,
                dtype=reliability_features.dtype,
            ).unsqueeze(0)
        )
        if self.constant_sensor_reliability_gate:
            # Reuse the dynamic head's learned output bias so the matched
            # control has exactly the same parameter inventory and initialization.
            reliability_logits = self.sensor_reliability_predictor[-1].bias.view(
                1, 1
            ).expand(actual_sensor.shape[0], 1)
        else:
            reliability_logits = self.sensor_reliability_predictor(
                reliability_features
            )
        if self.sensor_grid_quality_enabled and not self.constant_sensor_reliability_gate:
            if sensor_quality_features is None or tuple(
                sensor_quality_features.shape
            ) != (actual_sensor.shape[0], SENSOR_GRID_QUALITY_DIM):
                raise ValueError(
                    "sensor_quality_features must have shape "
                    f"[B,{SENSOR_GRID_QUALITY_DIM}]"
                )
            reliability_logits = reliability_logits + (
                self.sensor_grid_quality_predictor(sensor_quality_features)
            )
        if (
            self.sensor_cross_modal_quality_enabled
            and not self.constant_sensor_reliability_gate
        ):
            if sensor_cross_modal_quality_features is None or tuple(
                sensor_cross_modal_quality_features.shape
            ) != (actual_sensor.shape[0], SENSOR_CROSS_MODAL_QUALITY_DIM):
                raise ValueError(
                    "sensor_cross_modal_quality_features must have shape "
                    f"[B,{SENSOR_CROSS_MODAL_QUALITY_DIM}]"
                )
            reliability_logits = reliability_logits + (
                self.sensor_cross_modal_quality_predictor(
                    sensor_cross_modal_quality_features
                )
            )
        raw_reliability = torch.sigmoid(reliability_logits)
        sensor_reliability = raw_reliability * sensor_available
        return (
            sensor_reliability,
            consistency,
            discrepancy,
            uncertainty,
            reliability_features,
        )

    @staticmethod
    def _pairwise_modality_weights(tokens: Tensor) -> Tensor:
        normalized = F.normalize(tokens, dim=-1)
        pairwise = normalized @ normalized.transpose(1, 2)
        # 排除恒为 1 的自相似，每个模态取与另外两模态的平均余弦相似度。
        scores = (pairwise.sum(dim=-1) - pairwise.diagonal(dim1=1, dim2=2)) / 2.0
        return torch.softmax(scores, dim=-1)

    def forward_from_features(
        self,
        visual_scales: Sequence[Tensor],
        sensor_features: Tensor,
        text_prototypes: Tensor,
        *,
        fallback_text_prototypes: Tensor | None = None,
        sensor_available: Tensor | None = None,
        sensor_quality_features: Tensor | None = None,
        sensor_grid_observation: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """从预计算特征执行融合；此接口不需要 Ultralytics/Transformers。"""

        if self.visual_modality_enabled:
            if len(visual_scales) != 3:
                raise ValueError("visual_scales 必须恰好包含三个尺度")
            batch_size = visual_scales[0].shape[0]
            for index, features in enumerate(visual_scales):
                self._validate_features(
                    f"visual_scales[{index}]", features, batch_size
                )
            projected_scales = [
                projection(features)
                for projection, features in zip(
                    self.scale_projections, visual_scales
                )
            ]
            initial_scale_weights = torch.softmax(
                self.scale_prior_logits, dim=0
            )
            visual_base = sum(
                weight * features
                for weight, features in zip(
                    initial_scale_weights, projected_scales
                )
            )
        else:
            self._validate_features("sensor_features", sensor_features)
            batch_size = sensor_features.shape[0]
            visual_base = sensor_features.new_zeros(batch_size, self.feature_dim)
            projected_scales = [torch.zeros_like(visual_base) for _ in range(3)]
        if self.sensor_modality_enabled:
            self._validate_features("sensor_features", sensor_features, batch_size)
        if self.text_modality_enabled:
            (
                visual_text_context,
                visual_prototype_weights,
            ) = self._visual_prototype_context(visual_base, text_prototypes)
        else:
            visual_text_context = torch.zeros_like(visual_base)
            visual_prototype_weights = visual_base.new_zeros(
                batch_size, self.num_classes
            )
        if self.fallback_semantic_mode == "vision_only":
            fallback_semantic_context = torch.zeros_like(visual_base)
            fallback_prototype_weights = visual_base.new_zeros(
                batch_size, self.num_classes
            )
            fallback_semantic_logits = visual_base.new_zeros(
                batch_size, self.num_classes
            )
            fallback_sensor_center_context = torch.zeros_like(visual_base)
        elif self.fallback_semantic_mode == "legacy_descriptions":
            fallback_semantic_context = visual_text_context
            fallback_prototype_weights = visual_prototype_weights
            fallback_semantic_logits = visual_base.new_zeros(
                batch_size, self.num_classes
            )
            fallback_sensor_center_context = torch.zeros_like(visual_base)
        else:
            class_keys = (
                self.fallback_class_id_keys.to(
                    device=visual_base.device,
                    dtype=visual_base.dtype,
                )
                if self.fallback_semantic_mode == "class_id_keys"
                else fallback_text_prototypes
            )
            if class_keys is None:
                raise ValueError(
                    "class keys are required for fallback mode "
                    f"{self.fallback_semantic_mode!r}"
                )
            if self.fallback_semantic_memory_enabled:
                if (
                    self.fallback_query_projection is None
                    or self.fallback_sensor_value_bank is None
                ):
                    raise RuntimeError("fallback semantic memory is incomplete")
                query = F.normalize(
                    self.fallback_query_projection(visual_base),
                    dim=-1,
                )
                keys = F.normalize(class_keys, dim=-1)
                fallback_semantic_logits = (
                    query @ keys.transpose(0, 1)
                ) / self.text_temperature
                fallback_prototype_weights = torch.softmax(
                    fallback_semantic_logits,
                    dim=-1,
                )
                fallback_semantic_context = (
                    fallback_prototype_weights
                    @ self.fallback_sensor_value_bank
                )
                fallback_sensor_center_context = (
                    fallback_prototype_weights
                    @ self.fallback_sensor_center_bank
                    if self.fallback_sensor_center_bank is not None
                    else torch.zeros_like(visual_base)
                )
            else:
                (
                    fallback_semantic_context,
                    fallback_prototype_weights,
                ) = self._visual_prototype_context(
                    visual_base,
                    class_keys,
                )
                fallback_semantic_logits = visual_base.new_zeros(
                    batch_size, self.num_classes
                )
                fallback_sensor_center_context = torch.zeros_like(visual_base)
        sensor_available_tensor = self._prepare_sensor_availability(
            sensor_available,
            batch_size=batch_size,
            device=visual_base.device,
            dtype=visual_base.dtype,
        )
        if not self.sensor_modality_enabled:
            sensor_available_tensor = torch.zeros_like(sensor_available_tensor)
            actual_sensor = torch.zeros_like(visual_base)
        else:
            actual_sensor = sensor_features
        if sensor_quality_features is None:
            grid_quality_features = visual_base.new_zeros(
                batch_size, SENSOR_GRID_QUALITY_DIM
            )
        else:
            grid_quality_features = sensor_quality_features.to(
                device=visual_base.device,
                dtype=visual_base.dtype,
            )
            if tuple(grid_quality_features.shape) != (
                batch_size,
                SENSOR_GRID_QUALITY_DIM,
            ):
                raise ValueError(
                    "sensor_quality_features must have shape "
                    f"[B,{SENSOR_GRID_QUALITY_DIM}]"
                )
        observed_sensor_grid = None
        if sensor_grid_observation is not None:
            observed_sensor_grid = sensor_grid_observation.to(
                device=visual_base.device,
                dtype=visual_base.dtype,
            )
            if tuple(observed_sensor_grid.shape) != (batch_size, 1, 4, 4):
                raise ValueError(
                    "sensor_grid_observation must have shape [B,1,4,4]"
                )
        if (
            self.sensor_cross_modal_quality_enabled
            and observed_sensor_grid is None
        ):
            raise ValueError(
                "sensor_grid_observation is required when cross-modal "
                "sensor quality is enabled"
            )
        predicted_sensor_grid = visual_base.new_zeros(
            batch_size,
            1,
            4,
            4,
        )
        cross_modal_quality_features = None
        if self.legacy_sensor_passthrough and self.sensor_reliability_enabled:
            if bool((sensor_available_tensor < 1.0).any()):
                raise ValueError(
                    "legacy checkpoint requires sensor_available=1; "
                    "missing sensor inference requires a checkpoint retrained "
                    "with reliability heads"
                )
            predicted_sensor = actual_sensor
            sensor_log_variance = torch.zeros_like(actual_sensor)
            sensor_reliability = torch.ones_like(sensor_available_tensor)
            sensor_consistency = torch.ones_like(sensor_available_tensor)
            sensor_discrepancy = torch.zeros_like(sensor_available_tensor)
            sensor_uncertainty = torch.zeros_like(sensor_available_tensor)
            sensor_reliability_features = visual_base.new_zeros(
                batch_size,
                len(SENSOR_RELIABILITY_EVIDENCE),
            )
            effective_sensor = actual_sensor
        elif self.sensor_reliability_enabled:
            predicted_sensor, sensor_log_variance = self._predict_sensor_latent(
                visual_base, fallback_semantic_context
            )
            grid_prediction_latent = predicted_sensor
            if (
                self.fallback_semantic_direct_latent
                and self.fallback_semantic_memory_enabled
                and self.fallback_semantic_mode
                not in {"vision_only", "legacy_descriptions"}
            ):
                predicted_sensor = fallback_sensor_center_context
            if self.sensor_cross_modal_quality_enabled:
                decoder_input = torch.cat(
                    (
                        visual_base,
                        fallback_semantic_context,
                        grid_prediction_latent,
                    ),
                    dim=-1,
                )
                predicted_sensor_grid = torch.sigmoid(
                    self.sensor_grid_decoder(decoder_input)
                ).reshape(batch_size, 1, 4, 4)
                reliability_grid = (
                    predicted_sensor_grid.detach()
                    if self.detach_predicted_grid_for_reliability
                    else predicted_sensor_grid
                )
                cross_modal_quality_features = (
                    cross_modal_sensor_grid_quality_features(
                        reliability_grid,
                        observed_sensor_grid,
                        sensor_available_tensor,
                    )
                )
            (
                sensor_reliability,
                sensor_consistency,
                sensor_discrepancy,
                sensor_uncertainty,
                sensor_reliability_features,
            ) = self._estimate_sensor_reliability(
                actual_sensor,
                predicted_sensor,
                sensor_log_variance,
                sensor_available_tensor,
                grid_quality_features,
                cross_modal_quality_features,
            )
            effective_sensor = predicted_sensor + sensor_reliability * (
                actual_sensor - predicted_sensor
            )
        else:
            predicted_sensor = torch.zeros_like(visual_base)
            sensor_log_variance = torch.zeros_like(visual_base)
            sensor_reliability = sensor_available_tensor
            sensor_consistency = sensor_available_tensor
            sensor_discrepancy = torch.zeros_like(sensor_available_tensor)
            sensor_uncertainty = torch.zeros_like(sensor_available_tensor)
            sensor_reliability_features = visual_base.new_zeros(
                batch_size,
                len(SENSOR_RELIABILITY_EVIDENCE),
            )
            effective_sensor = actual_sensor * sensor_available_tensor
        if self.text_modality_enabled:
            text_context, prototype_weights = self._prototype_context(
                visual_base, effective_sensor, text_prototypes
            )
        else:
            text_context = torch.zeros_like(visual_base)
            prototype_weights = visual_base.new_zeros(
                batch_size, self.num_classes
            )

        tokens = torch.stack(
            (
                self.visual_adapter(visual_base),
                self.text_adapter(text_context),
                self.sensor_adapter(effective_sensor),
            ),
            dim=1,
        )
        if self.film_enabled:
            tokens = (
                tokens * self.film_gamma.unsqueeze(0)
                + self.film_beta.unsqueeze(0)
            )
        modality_mask = self.modality_mask.to(
            device=tokens.device, dtype=tokens.dtype
        )
        tokens = tokens * modality_mask.view(1, 3, 1)
        if self.fusion_mode == "attention":
            modality_weights = self._pairwise_modality_weights(tokens)
            if not bool(torch.all(self.modality_mask > 0)):
                modality_weights = modality_weights * modality_mask.unsqueeze(0)
                modality_weights = modality_weights / modality_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                key_padding_mask = (~self.modality_mask.bool()).unsqueeze(0).expand(
                    batch_size, -1
                )
            else:
                key_padding_mask = None
            attention_input = self.down_projection(
                tokens * modality_weights.unsqueeze(-1)
            )
            attention_output, attention_weights = self.modality_attention(
                attention_input,
                attention_input,
                attention_input,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            refined_tokens = tokens + self.up_projection(attention_output)
        else:
            modality_weights = modality_mask.unsqueeze(0).expand(batch_size, -1)
            modality_weights = modality_weights / modality_weights.sum(
                dim=-1, keepdim=True
            )
            attention_weights = tokens.new_zeros(
                batch_size, self.modality_attention.num_heads, 3, 3
            )
            refined_tokens = tokens
        fused = (refined_tokens * modality_weights.unsqueeze(-1)).sum(dim=1)

        scale_logits = self.scale_prior_logits.unsqueeze(0) + self.dynamic_scale_gate(fused)
        scale_weights = torch.softmax(scale_logits, dim=-1)
        scale_stack = torch.stack(projected_scales, dim=1)
        refined_visual = (scale_stack * scale_weights.unsqueeze(-1)).sum(dim=1)
        multimodal_expert_logits = self.classifier(
            self.output_norm(fused + refined_visual)
        )
        if self.visual_anchor_enabled:
            if any(
                module is None
                for module in (
                    self.visual_anchor_scale_projections,
                    self.visual_anchor_adapter,
                    self.visual_anchor_dynamic_scale_gate,
                    self.visual_anchor_output_norm,
                    self.visual_anchor_classifier,
                    self.visual_anchor_gate,
                )
            ) or self.visual_anchor_scale_prior_logits is None:
                raise RuntimeError("visual anchor modules are incomplete")
            anchor_projected_scales = [
                projection(features)
                for projection, features in zip(
                    self.visual_anchor_scale_projections,
                    visual_scales,
                )
            ]
            anchor_prior = torch.softmax(
                self.visual_anchor_scale_prior_logits,
                dim=0,
            )
            anchor_visual_base = sum(
                weight * features
                for weight, features in zip(
                    anchor_prior,
                    anchor_projected_scales,
                )
            )
            anchor_token = self.visual_anchor_adapter(anchor_visual_base)
            anchor_scale_logits = (
                self.visual_anchor_scale_prior_logits.unsqueeze(0)
                + self.visual_anchor_dynamic_scale_gate(anchor_token)
            )
            anchor_scale_weights = torch.softmax(anchor_scale_logits, dim=-1)
            anchor_scale_stack = torch.stack(anchor_projected_scales, dim=1)
            anchor_refined_visual = (
                anchor_scale_stack * anchor_scale_weights.unsqueeze(-1)
            ).sum(dim=1)
            anchor_feature = anchor_token + anchor_refined_visual
            visual_anchor_logits = self.visual_anchor_classifier(
                self.visual_anchor_output_norm(anchor_feature)
            )

            anchor_probabilities = torch.softmax(
                visual_anchor_logits.detach(), dim=-1
            )
            expert_probabilities = torch.softmax(
                multimodal_expert_logits.detach(), dim=-1
            )
            anchor_confidence = anchor_probabilities.amax(dim=-1, keepdim=True)
            anchor_entropy = -(
                anchor_probabilities
                * anchor_probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1, keepdim=True) / math.log(self.num_classes)
            expert_disagreement = 0.5 * (
                anchor_probabilities - expert_probabilities
            ).abs().sum(dim=-1, keepdim=True)
            gate_evidence = torch.cat(
                (
                    anchor_confidence,
                    anchor_entropy,
                    sensor_reliability,
                    sensor_consistency,
                    sensor_uncertainty,
                    sensor_available_tensor,
                    modality_weights[:, 1:2],
                    modality_weights[:, 2:3],
                    expert_disagreement,
                ),
                dim=-1,
            )
            visual_anchor_gate = self.visual_anchor_gate_max * torch.sigmoid(
                self.visual_anchor_gate(gate_evidence)
            )
            logits = visual_anchor_logits + visual_anchor_gate * (
                multimodal_expert_logits - visual_anchor_logits
            )
        else:
            visual_anchor_logits = multimodal_expert_logits
            visual_anchor_gate = multimodal_expert_logits.new_ones(
                batch_size, 1
            )
            anchor_confidence = torch.softmax(
                multimodal_expert_logits.detach(), dim=-1
            ).amax(dim=-1, keepdim=True)
            expert_disagreement = multimodal_expert_logits.new_zeros(
                batch_size, 1
            )
            anchor_scale_weights = scale_weights
            logits = multimodal_expert_logits

        aux = {
            "prototype_weights": prototype_weights,
            "visual_prototype_weights": visual_prototype_weights,
            "modality_weights": modality_weights,
            "scale_weights": scale_weights,
            "visual_anchor_scale_weights": anchor_scale_weights,
            "attention_weights": attention_weights,
            "sensor_reliability": sensor_reliability,
            "predicted_sensor": predicted_sensor,
            "sensor_log_variance": sensor_log_variance,
            "sensor_available": sensor_available_tensor,
            "sensor_consistency": sensor_consistency,
            "sensor_discrepancy": sensor_discrepancy,
            "sensor_uncertainty": sensor_uncertainty,
            "sensor_reliability_features": sensor_reliability_features,
            "effective_sensor": effective_sensor,
            "modality_mask": modality_mask.unsqueeze(0).expand(batch_size, -1),
            "sensor_reliability_evidence_mask": (
                self.sensor_reliability_evidence_mask.to(
                    device=visual_base.device,
                    dtype=visual_base.dtype,
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            ),
            "sensor_grid_quality_features": grid_quality_features,
            "predicted_sensor_grid": predicted_sensor_grid,
            "sensor_cross_modal_quality_features": (
                cross_modal_quality_features
                if cross_modal_quality_features is not None
                else visual_base.new_zeros(
                    batch_size,
                    SENSOR_CROSS_MODAL_QUALITY_DIM,
                )
            ),
            "visual_base": visual_base,
            "visual_text_context": visual_text_context,
            "fallback_semantic_context": fallback_semantic_context,
            "fallback_prototype_weights": fallback_prototype_weights,
            "fallback_semantic_logits": fallback_semantic_logits,
            "fallback_sensor_center_context": fallback_sensor_center_context,
            "visual_anchor_logits": visual_anchor_logits,
            "multimodal_expert_logits": multimodal_expert_logits,
            "visual_anchor_gate": visual_anchor_gate,
            "visual_anchor_confidence": anchor_confidence,
            "visual_anchor_disagreement": expert_disagreement,
        }
        self.last_aux = {name: value.detach() for name, value in aux.items()}
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
        """生产前向。签名刻意不包含 label/target 参数，避免文本标签泄漏。"""

        descriptions = (
            tuple(class_descriptions)
            if class_descriptions is not None
            else self.class_descriptions
        )
        if self.text_modality_enabled and (
            descriptions is None or len(descriptions) != self.num_classes
        ):
            raise ValueError("必须提供全部候选类别描述，且数量与类别数一致")
        if self.visual_modality_enabled:
            if self.visual_encoder is None:
                raise RuntimeError("visual modality is enabled without an encoder")
            visual_scales = self.visual_encoder(images)
        else:
            visual_scales = ()
        if self.sensor_modality_enabled:
            if self.sensor_encoder is None:
                raise RuntimeError("sensor modality is enabled without an encoder")
            sensor_features = self.sensor_encoder(sensor_grid)
            grid_quality_features = (
                sensor_grid_quality_features(sensor_grid)
                if self.sensor_grid_quality_enabled
                else None
            )
        else:
            sensor_features = images.new_zeros(images.shape[0], self.feature_dim)
            grid_quality_features = None
        if self.text_modality_enabled:
            if self.text_encoder is None:
                raise RuntimeError("text modality is enabled without an encoder")
            text_prototypes = self.text_encoder(descriptions, images.device)
            fallback_prompts = self.fallback_semantic_prompts()
            semantic_encoder = self.fallback_text_encoder or self.text_encoder
            fallback_text_prototypes = (
                semantic_encoder(fallback_prompts, images.device)
                if fallback_prompts is not None
                else None
            )
        else:
            text_prototypes = images.new_zeros(self.num_classes, self.feature_dim)
            fallback_text_prototypes = None
        return self.forward_from_features(
            visual_scales,
            sensor_features,
            text_prototypes,
            fallback_text_prototypes=fallback_text_prototypes,
            sensor_available=sensor_available,
            sensor_quality_features=grid_quality_features,
            sensor_grid_observation=sensor_grid,
            return_aux=return_aux,
        )


def build_production_model(config: dict[str, Any]) -> MultimodalFusionClassifier:
    """由 JSON 配置构建生产模型。"""

    classes = config.get("classes", [])
    if not classes:
        raise ValueError("配置中的 classes 不能为空")
    names = [item["name"] if isinstance(item, dict) else str(item) for item in classes]
    descriptions = [
        item.get("description", item["name"]) if isinstance(item, dict) else str(item)
        for item in classes
    ]
    fallback_class_names = [
        item.get("semantic_name", item["name"])
        if isinstance(item, dict)
        else str(item)
        for item in classes
    ]
    fallback_semantic_descriptions = [
        item.get("semantic_description", item.get("description", item["name"]))
        if isinstance(item, dict)
        else str(item)
        for item in classes
    ]
    model_config = config.get("model", {})
    weights = model_config.get("yolo_weights", "../yolo11s-seg.pt")
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = (PROJECT_ROOT / weights_path).resolve()
    model = MultimodalFusionClassifier.create_production(
        len(names),
        descriptions,
        yolo_weights=weights_path,
        clip_model=model_config.get(
            "clip_model", "openai/clip-vit-base-patch16"
        ),
        clip_local_files_only=bool(model_config.get("clip_local_files_only", False)),
        freeze_yolo_layers=int(model_config.get("freeze_yolo_layers", 5)),
        freeze_clip_text_layers=int(
            model_config.get("freeze_clip_text_layers", 3)
        ),
        fallback_text_encoder_independent=bool(
            model_config.get("fallback_text_encoder_independent", False)
        ),
        sensor_standardize=bool(model_config.get("sensor_standardize", True)),
        feature_dim=int(model_config.get("feature_dim", 512)),
        attention_dim=int(model_config.get("attention_dim", 128)),
        attention_heads=int(model_config.get("attention_heads", 8)),
        dropout=float(model_config.get("dropout", 0.3)),
        film_gamma_init=float(model_config.get("film_gamma_init", 1.0)),
        film_beta_init=float(model_config.get("film_beta_init", 0.1)),
        scale_prior=model_config.get("scale_prior", [0.3, 0.5, 0.2]),
        text_temperature=float(model_config.get("text_temperature", 0.07)),
        sensor_log_variance_min=float(
            model_config.get("sensor_log_variance_min", -6.0)
        ),
        sensor_log_variance_max=float(
            model_config.get("sensor_log_variance_max", 2.0)
        ),
        sensor_reliability_hidden_dim=int(
            model_config.get("sensor_reliability_hidden_dim", 32)
        ),
        sensor_reliability_init=float(
            model_config.get("sensor_reliability_init", 0.85)
        ),
        enabled_modalities=model_config.get(
            "enabled_modalities", list(VALID_MODALITIES)
        ),
        fusion_mode=str(model_config.get("fusion_mode", "attention")),
        film_enabled=bool(model_config.get("film_enabled", True)),
        sensor_reliability_enabled=bool(
            model_config.get("sensor_reliability_enabled", True)
        ),
        constant_sensor_reliability_gate=bool(
            model_config.get("constant_sensor_reliability_gate", False)
        ),
        sensor_grid_quality_enabled=bool(
            model_config.get("sensor_grid_quality_enabled", False)
        ),
        sensor_grid_quality_hidden_dim=int(
            model_config.get("sensor_grid_quality_hidden_dim", 16)
        ),
        sensor_cross_modal_quality_enabled=bool(
            model_config.get("sensor_cross_modal_quality_enabled", False)
        ),
        sensor_grid_decoder_hidden_dim=int(
            model_config.get("sensor_grid_decoder_hidden_dim", 64)
        ),
        sensor_cross_modal_quality_hidden_dim=int(
            model_config.get("sensor_cross_modal_quality_hidden_dim", 32)
        ),
        detach_predicted_grid_for_reliability=bool(
            model_config.get("detach_predicted_grid_for_reliability", True)
        ),
        visual_anchor_enabled=bool(
            model_config.get("visual_anchor_enabled", False)
        ),
        visual_anchor_gate_hidden_dim=int(
            model_config.get("visual_anchor_gate_hidden_dim", 32)
        ),
        visual_anchor_gate_init=float(
            model_config.get("visual_anchor_gate_init", 0.1)
        ),
        visual_anchor_gate_max=float(
            model_config.get("visual_anchor_gate_max", 1.0)
        ),
        fallback_semantic_mode=str(
            model_config.get(
                "fallback_semantic_mode",
                "legacy_descriptions",
            )
        ),
        fallback_class_names=fallback_class_names,
        fallback_semantic_descriptions=fallback_semantic_descriptions,
        fallback_semantic_shuffle_seed=int(
            model_config.get("fallback_semantic_shuffle_seed", 2026)
        ),
        fallback_semantic_memory_enabled=bool(
            model_config.get("fallback_semantic_memory_enabled", False)
        ),
        fallback_semantic_direct_latent=bool(
            model_config.get("fallback_semantic_direct_latent", False)
        ),
    )
    model.class_names = tuple(names)
    return model
