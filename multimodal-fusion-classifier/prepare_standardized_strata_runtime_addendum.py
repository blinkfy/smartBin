"""Bind the batch-8 recovery after the frozen batch-32 mask OOM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "standardized-instance-strata"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
DEFAULT_FAILED_RGB = REPORT_ROOT / "evaluation" / "rgb_yolo.json"
DEFAULT_OUTPUT = REPORT_ROOT / "RUNTIME_RECOVERY_ADDENDUM.json"
EVALUATOR = PROJECT_ROOT / "evaluate_standardized_instance_strata.py"
ORIGINAL_EVALUATOR_SHA256 = (
    "66c15aa422f0c2338fbd2206a6ddb9471283199f1c3673b18b8f6b012a4c8190"
)
ORIGINAL_PROTOCOL_SHA256 = (
    "eab7d649ce0a0b4fddb4e60c58ad9f7060b288f0cee1f842ee5b571826a1fba8"
)
FAILED_RGB_SHA256 = (
    "5b1eff4cae2f9c80bdd3b9534f12f5712972382f559e5910e4c33f6f675d4050"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the standardized-strata OOM recovery addendum"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--failed-rgb", type=Path, default=DEFAULT_FAILED_RGB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build_addendum(
    protocol_path: Path = DEFAULT_PROTOCOL,
    failed_rgb_path: Path = DEFAULT_FAILED_RGB,
) -> dict[str, Any]:
    protocol_path = _resolve(protocol_path)
    failed_rgb_path = _resolve(failed_rgb_path)
    for path in (protocol_path, failed_rgb_path, EVALUATOR):
        if not path.is_file():
            raise FileNotFoundError(f"recovery input is missing: {path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    rgb = json.loads(failed_rgb_path.read_text(encoding="utf-8-sig"))
    original_source = protocol.get("source_bindings", {}).get("evaluator", {})
    if (
        _sha256(protocol_path) != ORIGINAL_PROTOCOL_SHA256
        or original_source.get("sha256") != ORIGINAL_EVALUATOR_SHA256
        or _sha256(failed_rgb_path) != FAILED_RGB_SHA256
        or rgb.get("valid") is not True
        or rgb.get("model", {}).get("model") != "rgb_yolo"
        or rgb.get("model", {}).get("source_image_count") != 570
        or rgb.get("model", {}).get("target_count") != 825
        or protocol.get("inference", {}).get("batch_size") != 32
    ):
        raise ValueError("frozen OOM evidence is inconsistent")
    revised_hash = _sha256(EVALUATOR)
    if revised_hash == ORIGINAL_EVALUATOR_SHA256:
        raise ValueError("evaluator recovery implementation is not revised")
    return {
        "schema_version": 1,
        "status": "frozen_after_batch32_oom_before_recovery_rerun",
        "valid": True,
        "formal_paper_metric": False,
        "failure": {
            "type": "torch.OutOfMemoryError",
            "stage": "v5_rgb_fallback_prediction",
            "completed_models_before_failure": ["rgb_yolo"],
            "failed_batch_size": 32,
            "retina_masks": True,
            "gpu": "NVIDIA GeForce RTX 4090",
        },
        "authorized_runtime_change": {
            "batch_size": {"from": 32, "to": 8},
            "output": {
                "from": (
                    "reports/qwen-material/standardized-instance-strata/"
                    "evaluation"
                ),
                "to": (
                    "reports/qwen-material/standardized-instance-strata/"
                    "evaluation-recovery-batch8"
                ),
            },
            "scientific_settings_unchanged": [
                "three checkpoints",
                "570 test images and 825 targets",
                "image size 640",
                "confidence 0.25",
                "NMS IoU 0.5",
                "FP32 inference",
                "retina masks",
                "IoU thresholds 0.50:0.95",
                "target strata and bootstrap",
            ],
        },
        "batch_invariance_check": {
            "required": True,
            "reference_model": "rgb_yolo",
            "reference_artifact": _binding(failed_rgb_path),
            "required_comparison": (
                "all non-runtime rgb_yolo rows must exactly match between "
                "batch32 and batch8"
            ),
        },
        "source_revision": {
            "original_evaluator_sha256": ORIGINAL_EVALUATOR_SHA256,
            "revised_evaluator_sha256": revised_hash,
        },
        "bindings": {
            "original_protocol": _binding(protocol_path),
            "completed_batch32_rgb_result": _binding(failed_rgb_path),
            "revised_evaluator": _binding(EVALUATOR),
            "addendum_builder": _binding(Path(__file__)),
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }


def write_addendum(
    output: Path = DEFAULT_OUTPUT,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    failed_rgb_path: Path = DEFAULT_FAILED_RGB,
) -> dict[str, Any]:
    output = _resolve(output)
    addendum = build_addendum(protocol_path, failed_rgb_path)
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8-sig"))
        left = dict(existing)
        right = dict(addendum)
        left.pop("frozen_at", None)
        right.pop("frozen_at", None)
        if left != right:
            raise FileExistsError(
                "refusing to replace a drifted runtime addendum"
            )
        return existing
    _atomic_text(
        output,
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 标准化实例分层运行时恢复附录",
        "",
        "- 原冻结 batch：32。",
        "- 失败：RGB模型完成后，v5 retina-mask 推理发生CUDA OOM。",
        "- 恢复 batch：8；新输出目录独立保存。",
        "- 模型、测试集、阈值、FP32、掩膜、分层与bootstrap均不变。",
        "- 必须验证RGB模型batch32与batch8的全部非运行时指标完全一致。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return addendum


def main() -> None:
    args = parse_args()
    result = write_addendum(
        args.output,
        protocol_path=args.protocol,
        failed_rgb_path=args.failed_rgb,
    )
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": result["valid"],
                "status": result["status"],
                "batch_size": result["authorized_runtime_change"][
                    "batch_size"
                ],
                "revised_evaluator_sha256": result["source_revision"][
                    "revised_evaluator_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
