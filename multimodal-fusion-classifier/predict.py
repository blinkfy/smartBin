"""单目标多模态预测 CLI；不接受真实类别或目标文本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmfc.checkpoints import (
    load_checkpoint as _torch_load,
    load_model_state_compatible as _load_model_state_compatible,
    validated_checkpoint_class_names as _validated_checkpoint_class_names,
)
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
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "full_cross_modal_quality"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "ablations"
            / "full_cross_modal_quality.json"
        ),
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
