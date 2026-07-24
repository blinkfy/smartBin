"""Deterministic rule sensors and optional Qwen3-VL image-only enhancement."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageOps


PROPERTY_RANGES: dict[str, tuple[float, float]] = {
    "weight_g": (0.1, 5000.0),
    "temperature_c": (-20.0, 80.0),
    "moisture_pct": (0.0, 100.0),
    "odor_index": (0.0, 1.0),
    "conductivity_us_cm": (0.0, 5000.0),
}
LEAKAGE_KEYS = frozenset(
    {
        "label",
        "labels",
        "class",
        "classes",
        "target",
        "targets",
        "target_label",
        "class_name",
        "category",
        "category_id",
        "label_id",
        "class_id",
        "target_id",
        "label_index",
        "class_index",
        "target_index",
        "category_index",
    }
)
QWEN_PROMPT = """仅根据这张目标裁剪图估计可由物理传感器观测的属性。不要识别、猜测或输出物品类别，也不要根据类别先验推断。请只返回 JSON 对象，格式为：
{"properties":{"weight_g":数字,"temperature_c":数字,"moisture_pct":数字,"odor_index":数字,"conductivity_us_cm":数字},"normalized_grid":[[4列数字，共4行]]}
normalized_grid 每项范围 0 到 1；其余字段使用常见物理单位。"""


def contains_leakage_key(value: object) -> bool:
    """Recursively detect target/classification keys in a JSON-like value."""

    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in LEAKAGE_KEYS or contains_leakage_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_leakage_key(item) for item in value)
    return False


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _seed_value(sample_id: str, seed: int) -> tuple[int, str]:
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big"), digest.hex()[:16]


def crop_object(image: Image.Image, bbox: list[float] | tuple[float, ...]) -> Image.Image:
    """Crop a bbox in memory after applying EXIF orientation."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    x1 = max(0, min(width - 1, int(math.floor(float(bbox[0])))))
    y1 = max(0, min(height - 1, int(math.floor(float(bbox[1])))))
    x2 = max(x1 + 1, min(width, int(math.ceil(float(bbox[2])))))
    y2 = max(y1 + 1, min(height, int(math.ceil(float(bbox[3])))))
    return image.crop((x1, y1, x2, y2))


def _crop_statistics(crop: Image.Image) -> tuple[dict[str, Any], np.ndarray]:
    array = np.asarray(crop, dtype=np.float32) / 255.0
    rgb_mean = array.reshape(-1, 3).mean(axis=0)
    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        np.maximum(maximum, 1e-6),
        out=np.zeros_like(maximum),
        where=maximum > 1e-6,
    )
    gray = (
        array[:, :, 0] * 0.299 + array[:, :, 1] * 0.587 + array[:, :, 2] * 0.114
    )
    vertical = float(np.abs(np.diff(gray, axis=0)).mean()) if gray.shape[0] > 1 else 0.0
    horizontal = (
        float(np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    )
    resized = np.asarray(
        crop.convert("L").resize((4, 4), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    stats = {
        "brightness": round(float(gray.mean()), 6),
        "saturation": round(float(saturation.mean()), 6),
        "texture": round((vertical + horizontal) / 2.0, 6),
        "rgb_mean": [round(float(value), 6) for value in rgb_mean],
        "crop_width": crop.width,
        "crop_height": crop.height,
    }
    return stats, resized


def simulate_rule_sensor(
    image: Image.Image,
    bbox: list[float] | tuple[float, ...],
    sample_id: str,
    *,
    seed: int = 20260724,
    source_image_id: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic sensor record without accepting a target label."""
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    crop = crop_object(oriented, bbox)
    stats, coarse_luma = _crop_statistics(crop)
    numeric_seed, seed_token = _seed_value(sample_id, seed)
    rng = np.random.default_rng(numeric_seed)
    bbox_area_ratio = (crop.width * crop.height) / max(1.0, oriented.width * oriented.height)
    aspect_ratio = crop.width / max(1.0, crop.height)
    brightness = float(stats["brightness"])
    saturation = float(stats["saturation"])
    texture = float(stats["texture"])

    pressure = (
        0.48 * (1.0 - coarse_luma)
        + 0.32 * min(1.0, math.sqrt(bbox_area_ratio) * 2.0)
        + 0.20 * texture
        + rng.normal(0.0, 0.018, size=(4, 4))
    )
    pressure = np.clip(pressure, 0.0, 1.0)
    span = float(np.ptp(pressure))
    normalized = (
        (pressure - float(pressure.min())) / span
        if span > 1e-9
        else np.full((4, 4), 0.5, dtype=np.float64)
    )

    properties = {
        "weight_g": _clamp(
            12.0 + 1450.0 * bbox_area_ratio + 35.0 * texture + rng.normal(0.0, 2.0),
            *PROPERTY_RANGES["weight_g"],
        ),
        "temperature_c": _clamp(
            20.0 + 7.0 * brightness + rng.normal(0.0, 0.35),
            *PROPERTY_RANGES["temperature_c"],
        ),
        "moisture_pct": _clamp(
            8.0 + 58.0 * saturation + 18.0 * (1.0 - brightness) + rng.normal(0.0, 1.0),
            *PROPERTY_RANGES["moisture_pct"],
        ),
        "odor_index": _clamp(
            0.04 + 0.42 * saturation + 0.32 * texture + rng.normal(0.0, 0.012),
            *PROPERTY_RANGES["odor_index"],
        ),
        "conductivity_us_cm": _clamp(
            30.0 + 21.0 * (8.0 + 58.0 * saturation) + rng.normal(0.0, 18.0),
            *PROPERTY_RANGES["conductivity_us_cm"],
        ),
    }
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "grid_shape": [4, 4],
        "grid": [[round(float(value), 6) for value in row] for row in pressure],
        "normalized_grid": [
            [round(float(value), 6) for value in row] for row in normalized
        ],
        "properties": {key: round(float(value), 4) for key, value in properties.items()},
        "source": {
            "generator": "rule-v1",
            "seed": seed_token,
            "source_image_id": source_image_id,
            "inputs": ["crop_statistics", "bbox_geometry", "sample_seed"],
            "bbox_area_ratio": round(float(bbox_area_ratio), 8),
            "bbox_aspect_ratio": round(float(aspect_ratio), 6),
            "crop_statistics": stats,
        },
    }


def simulate_rule_sensor_from_path(
    image_path: Path | str,
    bbox: list[float] | tuple[float, ...],
    sample_id: str,
    *,
    seed: int = 20260724,
    source_image_id: str | None = None,
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        return simulate_rule_sensor(
            image,
            bbox,
            sample_id,
            seed=seed,
            source_image_id=source_image_id,
        )


def _image_data_url(crop: Image.Image) -> str:
    buffer = io.BytesIO()
    crop.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _default_transport(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_json_content(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("response choice has no message")
    content = message.get("content")
    if isinstance(content, list):
        text_parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        content = "".join(text_parts)
    if isinstance(content, str):
        content = content.strip()
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```")
            content = content.removesuffix("```").strip()
        value = json.loads(content)
    else:
        value = content
    if not isinstance(value, Mapping):
        raise ValueError("response content is not a JSON object")
    return value


def _validated_enhancement(
    value: Mapping[str, Any], rule_record: Mapping[str, Any]
) -> tuple[dict[str, float], list[list[float]]]:
    properties = value.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("properties must be an object")
    clean_properties: dict[str, float] = {}
    for name, bounds in PROPERTY_RANGES.items():
        raw = properties.get(name)
        if isinstance(raw, bool):
            raise ValueError(f"{name} must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        clean_properties[name] = round(_clamp(number, *bounds), 4)

    raw_grid = value.get("normalized_grid")
    if (
        not isinstance(raw_grid, list)
        or len(raw_grid) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in raw_grid)
    ):
        raise ValueError("normalized_grid must be 4x4")
    clean_grid: list[list[float]] = []
    for row in raw_grid:
        clean_row: list[float] = []
        for raw in row:
            if isinstance(raw, bool):
                raise ValueError("normalized_grid values must be numeric")
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("normalized_grid values must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError("normalized_grid values must be finite")
            clean_row.append(round(_clamp(number, 0.0, 1.0), 6))
        clean_grid.append(clean_row)
    return clean_properties, clean_grid


class QwenSensorEnhancer:
    """OpenAI-compatible Qwen3-VL enhancer with safe rule fallback."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-vl-flash",
        retries: int = 2,
        timeout: float = 45.0,
        transport: Callable[
            [str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]
        ] = _default_transport,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = max(0, retries)
        self.timeout = timeout
        self.transport = transport

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def enhance(
        self,
        image: Image.Image,
        bbox: list[float] | tuple[float, ...],
        rule_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Enhance one sample; never call the transport when no key is present."""
        fallback = deepcopy(dict(rule_record))
        if not self.api_key:
            fallback["source"] = {
                **dict(fallback["source"]),
                "generator": "rule-v1",
                "enhancer": "qwen3-vl-flash",
                "enhancement_status": "fallback_no_api_key",
            }
            return fallback

        crop = crop_object(image, bbox)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QWEN_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(crop)},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }
        errors: list[str] = []
        for attempt in range(self.retries + 1):
            try:
                response = self.transport(
                    self.endpoint, headers, payload, self.timeout
                )
                value = _extract_json_content(response)
                properties, grid = _validated_enhancement(value, rule_record)
                result = deepcopy(dict(rule_record))
                result["properties"] = properties
                result["normalized_grid"] = grid
                result["source"] = {
                    **dict(result["source"]),
                    "generator": self.model,
                    "model": self.model,
                    "enhancement_status": "success",
                    "attempts": attempt + 1,
                }
                return result
            except Exception as exc:  # network/provider/schema errors all fall back
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < self.retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
        fallback["source"] = {
            **dict(fallback["source"]),
            "generator": "rule-v1",
            "enhancer": self.model,
            "enhancement_status": "fallback_error",
            "attempts": self.retries + 1,
            "error": errors[-1][:500] if errors else "unknown error",
        }
        return fallback


def qwen_api_key(env_name: str = "DASHSCOPE_API_KEY") -> str | None:
    value = os.environ.get(env_name)
    return value.strip() if value and value.strip() else None
