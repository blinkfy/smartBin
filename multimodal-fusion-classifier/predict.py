"""单目标多模态预测 CLI；不接受真实类别或目标文本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmfc.model import build_production_model
from mmfc.training import (
    PROJECT_ROOT,
    _extract_grid,
    load_config,
    load_cropped_image,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测一个目标框内的智能垃圾类别")
    parser.add_argument("--image", type=Path, required=True, help="原始图像路径")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="原图坐标目标框",
    )
    sensor_group = parser.add_mutually_exclusive_group(required=True)
    sensor_group.add_argument(
        "--sensor-json", type=Path, help="包含 4×4 grid 的 UTF-8 JSON 文件"
    )
    sensor_group.add_argument(
        "--sensor-grid",
        type=str,
        help='4×4 JSON 数组，例如 "[[0,...],...]"；也接受长度 16 的数组',
    )
    sensor_group.add_argument(
        "--sensor-missing",
        action="store_true",
        help="Use a zero sensor grid with availability=0.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.json",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _load_cli_grid(
    args: argparse.Namespace, *, return_availability: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if getattr(args, "sensor_missing", False):
        grid_tensor = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        availability = torch.zeros(1, 1, dtype=torch.float32)
        return (grid_tensor, availability) if return_availability else grid_tensor

    if args.sensor_json is not None:
        with args.sensor_json.expanduser().open("r", encoding="utf-8") as handle:
            value: Any = json.load(handle)
    else:
        try:
            value = json.loads(args.sensor_grid)
        except json.JSONDecodeError as exc:
            raise ValueError("--sensor-grid 必须是合法 JSON") from exc
    array = np.asarray(value if not isinstance(value, dict) else _extract_grid(value))
    if array.size == 16 and array.shape != (4, 4):
        array = array.reshape(4, 4)
        value = array.tolist()
    grid = _extract_grid(value)
    grid_array = np.asarray(grid, dtype=np.float32)
    if np.any((grid_array < 0.0) | (grid_array > 1.0)):
        raise ValueError("预测传感器阵列值必须在 [0,1] 范围内")
    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    availability = torch.ones(1, 1, dtype=torch.float32)
    return (grid_tensor, availability) if return_availability else grid_tensor


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "当前 PyTorch 不支持安全的 weights_only=True 检查点加载；"
            "请升级到支持该参数的 PyTorch 版本"
        ) from exc
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"检查点格式无效：{path}")
    return checkpoint


def _validated_checkpoint_class_names(
    checkpoint: dict[str, Any], config: dict[str, Any]
) -> tuple[str, ...]:
    """确保输出索引沿用训练检查点且与当前配置顺序完全一致。"""

    config_names = tuple(
        item["name"] if isinstance(item, dict) else str(item)
        for item in config.get("classes", [])
    )
    raw_checkpoint_names = checkpoint.get("class_names")
    if not isinstance(raw_checkpoint_names, (list, tuple)) or not all(
        isinstance(name, str) and name for name in raw_checkpoint_names
    ):
        raise ValueError("检查点缺少有效 class_names，无法安全解释分类输出")
    checkpoint_names = tuple(raw_checkpoint_names)
    if checkpoint_names != config_names:
        raise ValueError(
            "检查点 class_names 与当前配置 classes 的顺序不一致；"
            f"检查点={list(checkpoint_names)!r}，配置={list(config_names)!r}"
        )
    return checkpoint_names


_COMPATIBLE_MISSING_PREFIXES = (
    "sensor_mean_predictor.",
    "sensor_logvar_predictor.",
    "sensor_reliability_predictor.",
)


def _new_sensor_head_keys(model: torch.nn.Module) -> set[str]:
    return {
        key
        for key in model.state_dict()
        if key.startswith(_COMPATIBLE_MISSING_PREFIXES)
    }


def _load_model_state_compatible(model: torch.nn.Module, state: dict[str, Any]) -> None:
    new_head_keys = _new_sensor_head_keys(model)
    state_keys = set(state)
    missing_new_heads = sorted(new_head_keys - state_keys)
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    disallowed_missing = [
        key
        for key in result.missing_keys
        if key not in new_head_keys
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "checkpoint state is incompatible: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    if missing_new_heads:
        if len(missing_new_heads) != len(new_head_keys):
            raise RuntimeError(
                "checkpoint state is incompatible: partial sensor reliability "
                f"head missing={missing_new_heads}"
            )
        setter = getattr(model, "set_legacy_sensor_passthrough", None)
        if callable(setter):
            setter(True)
        else:
            setattr(model, "legacy_sensor_passthrough", True)
    else:
        setter = getattr(model, "set_legacy_sensor_passthrough", None)
        if callable(setter):
            setter(False)
        else:
            setattr(model, "legacy_sensor_passthrough", False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(
        args.device
        or config.get("training", {}).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    checkpoint_path = resolve_project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到模型检查点：{checkpoint_path}")

    checkpoint = _torch_load(checkpoint_path, device)
    class_names = _validated_checkpoint_class_names(checkpoint, config)

    # 模型内部固定编码配置中的全部候选类别描述，不接收目标类别文本。
    model = build_production_model(config).to(device)
    _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()

    image_size = int(config.get("data", {}).get("image_size", 640))
    image = load_cropped_image(
        args.image.resolve(),
        args.bbox,
        manifest_dir=PROJECT_ROOT,
        image_size=image_size,
        preprocess=config.get("preprocess", {}),
    ).unsqueeze(0).to(device)
    sensor, sensor_availability = _load_cli_grid(args, return_availability=True)
    sensor = sensor.to(device)
    sensor_availability = sensor_availability.to(device)
    with torch.no_grad():
        logits, aux = model(
            image,
            sensor,
            sensor_available=sensor_availability,
            return_aux=True,
        )
        probabilities = torch.softmax(logits, dim=-1)[0]
    predicted_id = int(probabilities.argmax())
    result = {
        "label_id": predicted_id,
        "label": class_names[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": {
            name: float(probability)
            for name, probability in zip(class_names, probabilities.cpu())
        },
        "modality_weights": aux["modality_weights"][0].cpu().tolist(),
        "scale_weights": aux["scale_weights"][0].cpu().tolist(),
        "sensor_reliability": float(aux["sensor_reliability"][0, 0].cpu()),
        "sensor_available": float(aux["sensor_available"][0, 0].cpu()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
