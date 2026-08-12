"""Audit exact equivalence and timing for workers=4 versus workers=8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_external_sensor_worker_equivalence import (
    _model_state,
    compare_state_dicts,
    normalized_train_args,
)
from mmfc.model import PROJECT_ROOT


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
)
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg-external"
)
BASE_NAME = "yolo11s-seg-joint-external-rule-workers4-benchmark"
CANDIDATE_NAME = "yolo11s-seg-joint-external-rule-workers8-benchmark"
DEFAULT_OUTPUT = REPORT_ROOT / "WORKER_SCALING_4_TO_8.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit workers=4 versus workers=8"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_last_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"training CSV has no rows: {path}")
    return rows[-1]


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


def select_worker_count(
    workers4_seconds: float,
    workers8_seconds: float,
    *,
    exact: bool,
) -> int:
    if not exact:
        return 4
    return 8 if workers8_seconds < workers4_seconds else 4


def audit(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    paths = {
        "auditor_source": Path(__file__),
        "workers4_protocol": REPORT_ROOT / BASE_NAME / "PROTOCOL.json",
        "workers8_protocol": (
            REPORT_ROOT / CANDIDATE_NAME / "PROTOCOL.json"
        ),
        "workers4_evaluation": (
            REPORT_ROOT / BASE_NAME / "evaluation.json"
        ),
        "workers8_evaluation": (
            REPORT_ROOT / CANDIDATE_NAME / "evaluation.json"
        ),
        "workers4_results": (
            CHECKPOINT_ROOT / BASE_NAME / "results.csv"
        ),
        "workers8_results": (
            CHECKPOINT_ROOT / CANDIDATE_NAME / "results.csv"
        ),
        "workers4_checkpoint": (
            CHECKPOINT_ROOT / BASE_NAME / "weights" / "best.pt"
        ),
        "workers8_checkpoint": (
            CHECKPOINT_ROOT / CANDIDATE_NAME / "weights" / "best.pt"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "worker scaling inputs are missing: " + ", ".join(missing)
        )
    p4 = _read_json(paths["workers4_protocol"])
    p8 = _read_json(paths["workers8_protocol"])
    e4 = _read_json(paths["workers4_evaluation"])
    e8 = _read_json(paths["workers8_evaluation"])
    row4 = _read_last_csv_row(paths["workers4_results"])
    row8 = _read_last_csv_row(paths["workers8_results"])
    state = compare_state_dicts(
        _model_state(paths["workers4_checkpoint"]),
        _model_state(paths["workers8_checkpoint"]),
    )
    errors: list[str] = []
    args4 = p4.get("train_args", {})
    args8 = p8.get("train_args", {})
    if (
        args4.get("workers") != 4
        or args8.get("workers") != 8
        or p4.get("seed") != p8.get("seed")
        or p4.get("sensor_source") != p8.get("sensor_source")
        or p4.get("architecture") != p8.get("architecture")
        or p4.get("fusion_args") != p8.get("fusion_args")
        or normalized_train_args(args4) != normalized_train_args(args8)
    ):
        errors.append("protocols differ beyond run name and worker count")
    non_time_differences = [
        key
        for key in sorted(set(row4) | set(row8))
        if key != "time" and row4.get(key) != row8.get(key)
    ]
    if non_time_differences:
        errors.append("non-time training CSV values differ")
    if not state["exact"]:
        errors.append("model state tensors differ")
    metrics4 = {
        "external_sensor": e4["external_sensor"]["metrics"],
        "rgb_fallback": e4["rgb_fallback"]["metrics"],
    }
    metrics8 = {
        "external_sensor": e8["external_sensor"]["metrics"],
        "rgb_fallback": e8["rgb_fallback"]["metrics"],
    }
    if metrics4 != metrics8:
        errors.append("test metrics differ")
    seconds4 = float(row4["time"])
    seconds8 = float(row8["time"])
    exact = not errors
    recommendation = select_worker_count(
        seconds4, seconds8, exact=exact
    )
    report = {
        "schema_version": 1,
        "state": "complete" if exact else "invalid",
        "valid": exact,
        "scope": "external_sensor_v2_worker_scaling_4_to_8",
        "authorized_change": {
            "workers": {"baseline": 4, "candidate": 8},
            "all_scientific_training_arguments_unchanged": True,
        },
        "timing": {
            "workers4_training_seconds": seconds4,
            "workers8_training_seconds": seconds8,
            "workers8_over_workers4_ratio": seconds8 / seconds4,
            "workers8_slowdown_fraction": seconds8 / seconds4 - 1.0,
            "workers4_over_workers8_speed_ratio": seconds4 / seconds8,
        },
        "equivalence": {
            "model_state": state,
            "training_csv_non_time_equal": not non_time_differences,
            "training_csv_non_time_differences": non_time_differences,
            "external_and_fallback_test_metrics_equal": (
                metrics4 == metrics8
            ),
        },
        "recommendation": {
            "formal_training_workers": recommendation,
            "retain_frozen_workers4": recommendation == 4,
            "batch_size_unchanged": 32,
            "reason": (
                "workers=8 is exactly equivalent but slower on the current "
                "Windows/PyTorch/Ultralytics/RTX 4090 environment"
                if exact and recommendation == 4
                else "candidate requires further audit"
            ),
        },
        "boundaries": [
            "Only DataLoader worker concurrency differs.",
            "Batch size, model, optimizer, seed, data, and AMP are unchanged.",
            "The benchmark does not authorize a batch-size change.",
            "Rebenchmark after material environment or storage changes.",
        ],
        "source_bindings": {
            name: _binding(path) for name, path in paths.items()
        },
        "errors": errors,
    }
    _atomic_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# DataLoader workers=4 与 workers=8 等价性及耗时",
        "",
        f"- workers=4：{seconds4:.4f} 秒",
        f"- workers=8：{seconds8:.4f} 秒",
        f"- workers=8 相对变慢：{(seconds8 / seconds4 - 1.0) * 100:.2f}%",
        f"- 模型状态张量逐元素一致：{state['exact']}",
        f"- 除 time 外训练 CSV 一致：{not non_time_differences}",
        f"- external/fallback 测试指标一致：{metrics4 == metrics8}",
        "",
        (
            f"当前环境推荐 workers={recommendation}，batch 保持32。"
            "提高到8不会改变科学结果，但会降低训练速度。"
        ),
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    if errors:
        raise ValueError("; ".join(errors))
    return report


def main() -> None:
    result = audit(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "model_state_exact": result["equivalence"][
                    "model_state"
                ]["exact"],
                "workers8_slowdown_fraction": result["timing"][
                    "workers8_slowdown_fraction"
                ],
                "recommended_workers": result["recommendation"][
                    "formal_training_workers"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
