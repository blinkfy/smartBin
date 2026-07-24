"""医疗废品多模态融合分类器。

顶层包刻意不导入 PyTorch。这样 ``mmfc.data``、``mmfc.simulator`` 等数据
工具在没有可用 PyTorch 运行库的环境中仍可独立使用；模型符号仅在实际访问
时通过 PEP 562 惰性加载。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "CLIPPrototypeEncoder": ".model",
    "JointClassificationLoss": ".losses",
    "MulticlassFocalLoss": ".losses",
    "MultimodalFusionClassifier": ".model",
    "SensorEncoder": ".model",
    "YOLOMultiScaleEncoder": ".model",
    "build_production_model": ".model",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
