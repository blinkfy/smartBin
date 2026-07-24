"""专利启发的多模态分类模型。

生产编码器是可选依赖；融合核心只依赖 PyTorch，并可通过
``forward_from_features`` 在不加载任何外部权重的情况下测试。
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMARTBIN_ROOT = PROJECT_ROOT.parent
DEFAULT_YOLO_WEIGHTS = SMARTBIN_ROOT / "yolo11s-seg.pt"
LOCAL_ULTRALYTICS_ROOT = (SMARTBIN_ROOT / "ultralytics").resolve()
MIN_ULTRALYTICS_VERSION = (8, 3)
MAX_ULTRALYTICS_VERSION = (9, 0)


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

    def forward(self, descriptions: Sequence[str], device: torch.device) -> Tensor:
        if not descriptions:
            raise ValueError("候选类别描述不能为空")
        encoded = self.tokenizer(
            list(descriptions),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        output = self.text_model(**encoded)
        prototypes = output.text_embeds
        if prototypes.shape[-1] != 512:
            raise RuntimeError(f"CLIP 文本特征应为 512 维，实际为 {prototypes.shape[-1]}")
        return F.normalize(prototypes, dim=-1)


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
        sensor_encoder: nn.Module | None = None,
        feature_dim: int = 512,
        attention_dim: int = 128,
        attention_heads: int = 8,
        dropout: float = 0.3,
        scale_prior: Sequence[float] = (0.3, 0.5, 0.2),
        text_temperature: float = 0.07,
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

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_descriptions = (
            tuple(str(item) for item in class_descriptions)
            if class_descriptions is not None
            else None
        )
        if self.class_descriptions is not None and len(self.class_descriptions) != num_classes:
            raise ValueError("类别描述数量必须与 num_classes 相同")
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
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

        # 三种模态各自一组 FiLM 参数；初始化严格对应 γ=1、β=0.1。
        self.film_gamma = nn.Parameter(torch.ones(3, feature_dim))
        self.film_beta = nn.Parameter(torch.full((3, feature_dim), 0.1))

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
        self.last_aux: dict[str, Tensor] = {}

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
        **kwargs: Any,
    ) -> "MultimodalFusionClassifier":
        """构建使用本地 YOLO 与 CLIP ViT-B/16 的生产模型。"""

        visual = YOLOMultiScaleEncoder(
            yolo_weights, freeze_layers=freeze_yolo_layers
        )
        text = CLIPPrototypeEncoder(
            clip_model,
            freeze_layers=freeze_clip_text_layers,
            local_files_only=clip_local_files_only,
        )
        sensor = SensorEncoder(output_dim=512, normalize=True)
        return cls(
            num_classes,
            class_descriptions,
            visual_encoder=visual,
            text_encoder=text,
            sensor_encoder=sensor,
            **kwargs,
        )

    @staticmethod
    def _validate_features(name: str, features: Tensor, batch_size: int | None = None) -> None:
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError(f"{name} 必须为 [B,512]")
        if batch_size is not None and features.shape[0] != batch_size:
            raise ValueError(f"{name} 的 batch 大小不一致")

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
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """从预计算特征执行融合；此接口不需要 Ultralytics/Transformers。"""

        if len(visual_scales) != 3:
            raise ValueError("visual_scales 必须恰好包含三个尺度")
        batch_size = visual_scales[0].shape[0]
        for index, features in enumerate(visual_scales):
            self._validate_features(f"visual_scales[{index}]", features, batch_size)
        self._validate_features("sensor_features", sensor_features, batch_size)

        projected_scales = [
            projection(features)
            for projection, features in zip(self.scale_projections, visual_scales)
        ]
        initial_scale_weights = torch.softmax(self.scale_prior_logits, dim=0)
        visual_base = sum(
            weight * features
            for weight, features in zip(initial_scale_weights, projected_scales)
        )
        text_context, prototype_weights = self._prototype_context(
            visual_base, sensor_features, text_prototypes
        )

        tokens = torch.stack(
            (
                self.visual_adapter(visual_base),
                self.text_adapter(text_context),
                self.sensor_adapter(sensor_features),
            ),
            dim=1,
        )
        tokens = tokens * self.film_gamma.unsqueeze(0) + self.film_beta.unsqueeze(0)
        modality_weights = self._pairwise_modality_weights(tokens)

        attention_input = self.down_projection(
            tokens * modality_weights.unsqueeze(-1)
        )
        attention_output, attention_weights = self.modality_attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=True,
            average_attn_weights=False,
        )
        refined_tokens = tokens + self.up_projection(attention_output)
        fused = (refined_tokens * modality_weights.unsqueeze(-1)).sum(dim=1)

        scale_logits = self.scale_prior_logits.unsqueeze(0) + self.dynamic_scale_gate(fused)
        scale_weights = torch.softmax(scale_logits, dim=-1)
        scale_stack = torch.stack(projected_scales, dim=1)
        refined_visual = (scale_stack * scale_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(self.output_norm(fused + refined_visual))

        aux = {
            "prototype_weights": prototype_weights,
            "modality_weights": modality_weights,
            "scale_weights": scale_weights,
            "attention_weights": attention_weights,
        }
        self.last_aux = {name: value.detach() for name, value in aux.items()}
        return (logits, aux) if return_aux else logits

    def forward(
        self,
        images: Tensor,
        sensor_grid: Tensor,
        *,
        class_descriptions: Sequence[str] | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """生产前向。签名刻意不包含 label/target 参数，避免文本标签泄漏。"""

        if self.visual_encoder is None or self.text_encoder is None or self.sensor_encoder is None:
            raise RuntimeError(
                "未配置生产编码器；测试请调用 forward_from_features，"
                "生产请使用 create_production"
            )
        descriptions = (
            tuple(class_descriptions)
            if class_descriptions is not None
            else self.class_descriptions
        )
        if descriptions is None or len(descriptions) != self.num_classes:
            raise ValueError("必须提供全部候选类别描述，且数量与类别数一致")
        visual_scales = self.visual_encoder(images)
        sensor_features = self.sensor_encoder(sensor_grid)
        text_prototypes = self.text_encoder(descriptions, images.device)
        return self.forward_from_features(
            visual_scales,
            sensor_features,
            text_prototypes,
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
        feature_dim=int(model_config.get("feature_dim", 512)),
        attention_dim=int(model_config.get("attention_dim", 128)),
        attention_heads=int(model_config.get("attention_heads", 8)),
        dropout=float(model_config.get("dropout", 0.3)),
        scale_prior=model_config.get("scale_prior", [0.3, 0.5, 0.2]),
        text_temperature=float(model_config.get("text_temperature", 0.07)),
    )
    model.class_names = tuple(names)
    return model
