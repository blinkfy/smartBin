"""Annotation-free full-image soft-sensor observations."""

from __future__ import annotations

from copy import deepcopy
import json
import time
from typing import Any, Callable, Mapping

from PIL import Image, ImageOps

from .sensors import (
    PROPERTY_RANGES,
    _default_transport,
    _extract_json_content,
    _image_data_url,
    _validated_enhancement,
    simulate_rule_sensor,
)


QWEN_SCENE_PROMPT = """仅根据这张完整场景图估计垃圾投放区域可能产生的物理传感器观测。不要识别、猜测或输出物品类别，不要输出目标框、位置标签或分割信息。请只返回 JSON 对象，格式为：
{"properties":{"weight_g":数字,"temperature_c":数字,"moisture_pct":数字,"odor_index":数字,"conductivity_us_cm":数字},"normalized_grid":[[4列数字，共4行]]}
normalized_grid 表示按整张图从左到右、从上到下排列的 4×4 空间传感器响应，每项范围 0 到 1；其余字段使用常见物理单位。"""


def simulate_scene_rule_sensor(
    image: Image.Image,
    source_image_id: str,
    *,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Generate a deterministic full-image fallback without annotations."""

    oriented = ImageOps.exif_transpose(image).convert("RGB")
    base = simulate_rule_sensor(
        oriented,
        [0.0, 0.0, float(oriented.width), float(oriented.height)],
        f"scene:{source_image_id}",
        seed=seed,
        source_image_id=source_image_id,
    )
    return {
        "schema_version": 1,
        "source_image_id": source_image_id,
        "grid_shape": [4, 4],
        "grid": deepcopy(base["grid"]),
        "normalized_grid": deepcopy(base["normalized_grid"]),
        "properties": deepcopy(base["properties"]),
        "source": {
            "generator": "scene-rule-v1",
            "seed": base["source"]["seed"],
            "observation_scope": "full_image",
            "full_image_only": True,
            "annotation_inputs": [],
            "inputs": ["full_image_statistics", "source_seed"],
            "image_statistics": deepcopy(
                base["source"]["crop_statistics"]
            ),
        },
    }


class QwenSceneSensorEnhancer:
    """Qwen full-image enhancer with a complete scene-rule fallback."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3.5-flash",
        retries: int = 2,
        timeout: float = 45.0,
        transport: Callable[
            [str, Mapping[str, str], Mapping[str, Any], float],
            Mapping[str, Any],
        ] = _default_transport,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = max(0, int(retries))
        self.timeout = float(timeout)
        self.transport = transport

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def enhance(
        self,
        image: Image.Image,
        rule_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        fallback = deepcopy(dict(rule_record))
        if not self.api_key:
            fallback["source"] = {
                **dict(fallback["source"]),
                "enhancer": self.model,
                "enhancement_status": "fallback_no_api_key",
            }
            return fallback

        oriented = ImageOps.exif_transpose(image).convert("RGB")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QWEN_SCENE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(oriented)
                            },
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
                properties, grid = _validated_enhancement(
                    value, rule_record
                )
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
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < self.retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
        fallback["source"] = {
            **dict(fallback["source"]),
            "enhancer": self.model,
            "enhancement_status": "fallback_error",
            "attempts": self.retries + 1,
            "error": errors[-1][:500] if errors else "unknown error",
        }
        return fallback


def scene_record_usable(
    record: Mapping[str, Any],
    *,
    mode: str,
    qwen_model: str,
) -> bool:
    """Check whether a saved full-image record satisfies a resume mode."""

    source_id = record.get("source_image_id")
    if (
        record.get("schema_version") != 1
        or not isinstance(source_id, str)
        or not source_id
        or record.get("grid_shape") != [4, 4]
    ):
        return False
    source = record.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("observation_scope") != "full_image"
        or source.get("full_image_only") is not True
        or source.get("annotation_inputs") != []
    ):
        return False
    for key in ("grid", "normalized_grid"):
        value = record.get(key)
        if (
            not isinstance(value, list)
            or len(value) != 4
            or any(
                not isinstance(row, list) or len(row) != 4
                for row in value
            )
        ):
            return False
        try:
            flattened = [float(item) for row in value for item in row]
        except (TypeError, ValueError):
            return False
        if not all(0.0 <= item <= 1.0 for item in flattened):
            return False
    properties = record.get("properties")
    if not isinstance(properties, Mapping):
        return False
    for name, (lower, upper) in PROPERTY_RANGES.items():
        try:
            value = float(properties[name])
        except (KeyError, TypeError, ValueError):
            return False
        if not lower <= value <= upper:
            return False
    if mode == "rule":
        return source.get("generator") == "scene-rule-v1"
    if mode == "qwen":
        return (
            source.get("generator") == qwen_model
            and source.get("enhancement_status") == "success"
        )
    return False
