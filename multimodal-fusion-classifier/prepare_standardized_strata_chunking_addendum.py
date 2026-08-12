"""Bind chunked prediction after the long-list v5 batch-8 OOM."""

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
PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
PREVIOUS_ADDENDUM = REPORT_ROOT / "RUNTIME_RECOVERY_ADDENDUM.json"
BATCH32_RGB = REPORT_ROOT / "evaluation" / "rgb_yolo.json"
BATCH8_RGB = (
    REPORT_ROOT / "evaluation-recovery-batch8" / "rgb_yolo.json"
)
EVALUATOR = PROJECT_ROOT / "evaluate_standardized_instance_strata.py"
DEFAULT_OUTPUT = REPORT_ROOT / "RUNTIME_CHUNKING_ADDENDUM.json"
ORIGINAL_PROTOCOL_SHA256 = (
    "eab7d649ce0a0b4fddb4e60c58ad9f7060b288f0cee1f842ee5b571826a1fba8"
)
PREVIOUS_ADDENDUM_SHA256 = (
    "8a0a14333f6facd05d074321fc7d2b6fb008292da00181731e75679621955010"
)
PREVIOUS_EVALUATOR_SHA256 = (
    "ebea4b60ca4fcb0355ca9d5e8620648aac0ae164ea16350d607ee04581dfc327"
)
BATCH32_RGB_SHA256 = (
    "5b1eff4cae2f9c80bdd3b9534f12f5712972382f559e5910e4c33f6f675d4050"
)
BATCH8_RGB_SHA256 = (
    "acb8be4455f8149bcbea5ebaf6760e7e85782b344410fb3c4b4cba22a594c8ec"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze long-list chunking recovery addendum"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def build_addendum() -> dict[str, Any]:
    for path in (
        PROTOCOL,
        PREVIOUS_ADDENDUM,
        BATCH32_RGB,
        BATCH8_RGB,
        EVALUATOR,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"chunking evidence is missing: {path}")
    batch32 = json.loads(BATCH32_RGB.read_text(encoding="utf-8-sig"))
    batch8 = json.loads(BATCH8_RGB.read_text(encoding="utf-8-sig"))
    if (
        _sha256(PROTOCOL) != ORIGINAL_PROTOCOL_SHA256
        or _sha256(PREVIOUS_ADDENDUM) != PREVIOUS_ADDENDUM_SHA256
        or _sha256(EVALUATOR) == PREVIOUS_EVALUATOR_SHA256
        or _sha256(BATCH32_RGB) != BATCH32_RGB_SHA256
        or _sha256(BATCH8_RGB) != BATCH8_RGB_SHA256
        or batch32.get("rows") != batch8.get("rows")
        or batch32.get("model", {}).get("prediction_count")
        != batch8.get("model", {}).get("prediction_count")
    ):
        raise ValueError("batch-8 chunking recovery evidence is inconsistent")
    return {
        "schema_version": 1,
        "status": "frozen_after_long_list_batch8_oom_before_chunked_rerun",
        "valid": True,
        "formal_paper_metric": False,
        "failure": {
            "type": "torch.OutOfMemoryError",
            "stage": "v5_rgb_fallback_long_list_prediction",
            "batch_size": 8,
            "long_list_source_image_count": 570,
            "diagnosis": (
                "The same v5 checkpoint completed all 570 images when the "
                "ordered list was split into chunks of 64 with cache release."
            ),
        },
        "authorized_runtime_change": {
            "batch_size": 8,
            "prediction_chunk_size": 64,
            "release_between_chunks": [
                "drop result generator references",
                "Python garbage collection",
                "torch.cuda.empty_cache",
            ],
            "output": (
                "reports/qwen-material/standardized-instance-strata/"
                "evaluation-recovery-batch8-chunk64"
            ),
            "scientific_settings_unchanged": True,
        },
        "batch_invariance_check": {
            "required": True,
            "reference_artifacts": [
                _binding(BATCH32_RGB),
                _binding(BATCH8_RGB),
            ],
            "required_comparison": (
                "all metric rows and prediction counts must exactly match "
                "the chunked rgb_yolo result"
            ),
        },
        "source_revision": {
            "previous_evaluator_sha256": PREVIOUS_EVALUATOR_SHA256,
            "revised_evaluator_sha256": _sha256(EVALUATOR),
        },
        "bindings": {
            "original_protocol": _binding(PROTOCOL),
            "previous_runtime_addendum": _binding(PREVIOUS_ADDENDUM),
            "batch32_rgb_result": _binding(BATCH32_RGB),
            "batch8_rgb_result": _binding(BATCH8_RGB),
            "revised_evaluator": _binding(EVALUATOR),
            "addendum_builder": _binding(Path(__file__)),
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }


def write_addendum(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    addendum = build_addendum()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8-sig"))
        left = dict(existing)
        right = dict(addendum)
        left.pop("frozen_at", None)
        right.pop("frozen_at", None)
        if left != right:
            raise FileExistsError(
                "refusing to replace a drifted chunking addendum"
            )
        return existing
    _atomic_text(
        output,
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(
        output.with_suffix(".md"),
        "\n".join(
            [
                "# 标准化实例分层分块恢复附录",
                "",
                "- batch保持8；有序输入按64张分块。",
                "- 每块后释放生成器引用、执行GC并清理CUDA缓存。",
                "- 阈值、模型、图像顺序、FP32、掩膜和指标均不变。",
                "- batch32与batch8的RGB指标已先验完全一致。",
                "- 最终分块结果必须再次与两个RGB参考完全一致。",
                "",
            ]
        ),
    )
    return addendum


def main() -> None:
    result = write_addendum(parse_args().output)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": result["valid"],
                "status": result["status"],
                "prediction_chunk_size": result[
                    "authorized_runtime_change"
                ]["prediction_chunk_size"],
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
