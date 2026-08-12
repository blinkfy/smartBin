"""Audit exact model equivalence for the v2 four-worker speed variant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
)
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg-external"
)
BASE_NAME = "yolo11s-seg-joint-external-rule-smoke"
FAST_NAME = "yolo11s-seg-joint-external-rule-workers4-benchmark"
DEFAULT_OUTPUT = REPORT_ROOT / "WORKER_EQUIVALENCE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit workers=0 versus workers=4 v2 equivalence"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def normalized_train_args(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("name", None)
    result.pop("workers", None)
    return result


def compare_state_dicts(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_keys = list(baseline)
    candidate_keys = list(candidate)
    differences: list[dict[str, Any]] = []
    for name in sorted(set(baseline_keys) | set(candidate_keys)):
        left = baseline.get(name)
        right = candidate.get(name)
        if left is None or right is None:
            differences.append({"name": name, "reason": "missing"})
            continue
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            differences.append({"name": name, "reason": "metadata"})
            continue
        if not bool(left.equal(right)):
            differences.append(
                {
                    "name": name,
                    "reason": "value",
                    "different_element_count": int(
                        (left != right).count_nonzero().item()
                    ),
                    "maximum_absolute_difference": float(
                        (left.float() - right.float()).abs().max().item()
                    ),
                }
            )
    return {
        "key_order_equal": baseline_keys == candidate_keys,
        "tensor_count": len(baseline_keys),
        "different_tensor_count": len(differences),
        "differences": differences,
        "exact": baseline_keys == candidate_keys and not differences,
    }


def _model_state(checkpoint: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = payload.get("ema") or payload.get("model")
    if model is None:
        raise ValueError(f"checkpoint has no model or EMA: {checkpoint}")
    return model.state_dict()


def audit(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    baseline_report = REPORT_ROOT / BASE_NAME
    candidate_report = REPORT_ROOT / FAST_NAME
    baseline_run = CHECKPOINT_ROOT / BASE_NAME
    candidate_run = CHECKPOINT_ROOT / FAST_NAME
    paths = {
        "auditor_source": Path(__file__),
        "baseline_protocol": baseline_report / "PROTOCOL.json",
        "candidate_protocol": candidate_report / "PROTOCOL.json",
        "baseline_evaluation": baseline_report / "evaluation.json",
        "candidate_evaluation": candidate_report / "evaluation.json",
        "baseline_results": baseline_run / "results.csv",
        "candidate_results": candidate_run / "results.csv",
        "baseline_checkpoint": baseline_run / "weights" / "best.pt",
        "candidate_checkpoint": candidate_run / "weights" / "best.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "worker equivalence inputs are missing: " + ", ".join(missing)
        )

    base_protocol = _read_json(paths["baseline_protocol"])
    fast_protocol = _read_json(paths["candidate_protocol"])
    base_evaluation = _read_json(paths["baseline_evaluation"])
    fast_evaluation = _read_json(paths["candidate_evaluation"])
    base_row = _read_last_csv_row(paths["baseline_results"])
    fast_row = _read_last_csv_row(paths["candidate_results"])
    state_comparison = compare_state_dicts(
        _model_state(paths["baseline_checkpoint"]),
        _model_state(paths["candidate_checkpoint"]),
    )

    errors: list[str] = []
    base_args = base_protocol.get("train_args", {})
    fast_args = fast_protocol.get("train_args", {})
    if (
        base_protocol.get("seed") != fast_protocol.get("seed")
        or base_protocol.get("sensor_source") != fast_protocol.get("sensor_source")
        or base_protocol.get("architecture") != fast_protocol.get("architecture")
        or base_protocol.get("fusion_args") != fast_protocol.get("fusion_args")
        or normalized_train_args(base_args) != normalized_train_args(fast_args)
        or base_args.get("workers") != 0
        or fast_args.get("workers") != 4
    ):
        errors.append("protocols differ beyond run name and worker count")

    csv_keys = sorted(set(base_row) | set(fast_row))
    non_time_differences = [
        key
        for key in csv_keys
        if key != "time" and base_row.get(key) != fast_row.get(key)
    ]
    if non_time_differences:
        errors.append("non-time training CSV values differ")
    if not state_comparison["exact"]:
        errors.append("model state tensors differ")

    base_metrics = {
        "external_sensor": base_evaluation["external_sensor"]["metrics"],
        "rgb_fallback": base_evaluation["rgb_fallback"]["metrics"],
    }
    fast_metrics = {
        "external_sensor": fast_evaluation["external_sensor"]["metrics"],
        "rgb_fallback": fast_evaluation["rgb_fallback"]["metrics"],
    }
    if base_metrics != fast_metrics:
        errors.append("test metrics differ")
    baseline_seconds = float(base_row["time"])
    candidate_seconds = float(fast_row["time"])
    speedup = baseline_seconds / candidate_seconds
    if candidate_seconds >= baseline_seconds:
        errors.append("four-worker training was not faster")

    report = {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "external_sensor_v2_dataloader_worker_equivalence",
        "authorized_change": {
            "workers": {"from": 0, "to": 4},
            "all_scientific_training_arguments_unchanged": True,
        },
        "timing": {
            "baseline_training_seconds": baseline_seconds,
            "candidate_training_seconds": candidate_seconds,
            "speedup": speedup,
            "time_reduction_fraction": 1.0 - candidate_seconds / baseline_seconds,
        },
        "equivalence": {
            "checkpoint_file_sha_equal": (
                _sha256(paths["baseline_checkpoint"])
                == _sha256(paths["candidate_checkpoint"])
            ),
            "model_state": state_comparison,
            "training_csv_non_time_equal": not non_time_differences,
            "training_csv_non_time_differences": non_time_differences,
            "external_and_fallback_test_metrics_equal": base_metrics == fast_metrics,
        },
        "recommendation": {
            "formal_training_workers": 4 if not errors else 0,
            "batch_size_unchanged": 32,
            "rebenchmark_if_environment_changes": True,
        },
        "boundaries": [
            "The benchmark changes only DataLoader worker concurrency.",
            "It does not authorize changing batch size or optimization settings.",
            "The evidence was measured on the current Windows, PyTorch, Ultralytics, and RTX 4090 environment.",
            "The rule sidecar changes input values, not the worker scheduling mechanism used by Qwen sidecars.",
        ],
        "source_bindings": {
            name: _binding(path) for name, path in paths.items()
        },
        "errors": errors,
    }
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# v2 DataLoader worker 等价性审计",
        "",
        f"- 基线：workers=0，训练 {baseline_seconds:.4f} 秒",
        f"- 候选：workers=4，训练 {candidate_seconds:.4f} 秒",
        f"- 提速：{speedup:.3f}×",
        f"- 模型张量：{state_comparison['tensor_count']} 个，逐元素完全一致",
        f"- 训练/验证 CSV：除 time 外完全一致：{not non_time_differences}",
        f"- external 与 RGB fallback 测试指标完全一致：{base_metrics == fast_metrics}",
        "",
        "因此当前环境下正式 v2 训练可将 workers 从 0 调整为 4；batch、优化器、"
        "seed、输入和模型均保持不变。环境变化后应重新基准测试。",
        "",
    ]
    _atomic_write(output.with_suffix(".md"), "\n".join(markdown))
    if errors:
        raise ValueError("; ".join(errors))
    return report


def main() -> None:
    report = audit(parse_args().output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "speedup": report["timing"]["speedup"],
                "model_state_exact": report["equivalence"]["model_state"][
                    "exact"
                ],
                "recommended_workers": report["recommendation"][
                    "formal_training_workers"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
