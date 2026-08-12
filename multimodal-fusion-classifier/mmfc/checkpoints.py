"""Safe checkpoint loading shared by prediction and evaluation CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load project checkpoints without allowing arbitrary pickle execution."""

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


def validated_checkpoint_class_names(
    checkpoint: dict[str, Any], config: dict[str, Any]
) -> tuple[str, ...]:
    """Require checkpoint and configuration class orders to match exactly."""

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


def _inactive_encoder_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    enabled = set(getattr(model, "enabled_modalities", ()))
    return tuple(
        f"{name}_encoder."
        for name in ("visual", "text", "sensor")
        if name not in enabled
    )


def load_model_state_compatible(
    model: torch.nn.Module, state: dict[str, Any]
) -> None:
    """Load current or fully legacy states while rejecting partial head damage."""

    new_head_keys = _new_sensor_head_keys(model)
    inactive_prefixes = _inactive_encoder_prefixes(model)
    compatible_state = {
        key: value
        for key, value in state.items()
        if not key.startswith(inactive_prefixes)
    }
    state_keys = set(compatible_state)
    missing_new_heads = sorted(new_head_keys - state_keys)
    result = model.load_state_dict(compatible_state, strict=False)
    unexpected = list(result.unexpected_keys)
    disallowed_missing = [
        key for key in result.missing_keys if key not in new_head_keys
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
