"""Audit and aggregate independently retrained joint-head branch ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path
from prepare_joint_multimodal_seg import sha256_file


DEFAULT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
    / "retrained-ablations"
)
DEFAULT_FULL = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
    / "MULTISEED_SUMMARY.json"
)
DEFAULT_BASE = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "yolo-seg"
    / "evaluation.json"
)
METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit independently retrained joint-head ablations."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--full-summary", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--base-evaluation", type=Path, default=DEFAULT_BASE)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_loss_columns(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"training history is empty: {path}")
    return {
        "train_sensor_loss": [
            float(row["train/sensor_loss"]) for row in rows
        ],
        "val_sensor_loss": [
            float(row["val/sensor_loss"]) for row in rows
        ],
    }


def _checkpoint_structure(path: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(path)).model
    fusion = model.joint_fusion
    text_norms = [
        float(module.weight.detach().float().norm().item())
        for module in fusion.text_output
    ]
    sensor_norms = [
        float(module.weight.detach().float().norm().item())
        for module in fusion.sensor_output
    ]
    return {
        "fusion_mode": getattr(model, "fusion_mode", "full"),
        "text_enabled": bool(getattr(fusion, "text_enabled", True)),
        "sensor_enabled": bool(getattr(fusion, "sensor_enabled", True)),
        "text_output_l2": text_norms,
        "sensor_output_l2": sensor_norms,
    }


def audit(
    root: Path,
    full_summary_path: Path,
    base_evaluation_path: Path,
) -> dict[str, Any]:
    root = resolve_project_path(root)
    full_summary_path = resolve_project_path(full_summary_path)
    base_evaluation_path = resolve_project_path(base_evaluation_path)
    protocol_path = root / "PROTOCOL.json"
    protocol = _read_object(protocol_path)
    full_summary = _read_object(full_summary_path)
    base_evaluation = _read_object(base_evaluation_path)
    errors: list[str] = []

    if protocol.get("status") != "frozen_before_formal_ablation_training":
        errors.append("formal ablation protocol is not frozen")
    if not full_summary.get("valid"):
        errors.append("full multimodal multiseed summary is invalid")
    for relative, expected_hash in protocol["source_bindings"].items():
        path = resolve_project_path(Path(relative))
        if not path.is_file():
            errors.append(f"missing frozen source: {relative}")
        elif sha256_file(path) != expected_hash:
            errors.append(f"frozen source drift: {relative}")

    base_metrics = {
        metric: float(base_evaluation["metrics"][metric])
        for metric in METRICS
    }
    full_runs = {
        int(run["seed"]): run["metrics"] for run in full_summary["runs"]
    }
    full_metrics = {
        metric: float(full_summary["aggregate"][metric]["mean"])
        for metric in METRICS
    }
    expected_by_mode: dict[str, list[dict[str, Any]]] = {
        "text_only": [],
        "sensor_only": [],
    }
    for run in protocol["formal_runs"]:
        expected_by_mode[str(run["fusion_mode"])].append(run)

    checkpoint_hashes = []
    run_results: dict[str, list[dict[str, Any]]] = {
        "text_only": [],
        "sensor_only": [],
    }
    normalized_protocols: dict[str, list[dict[str, Any]]] = {
        "text_only": [],
        "sensor_only": [],
    }
    for mode, expected_runs in expected_by_mode.items():
        for expected in expected_runs:
            run_name = str(expected["run_name"])
            seed = int(expected["seed"])
            report_run = root.parent / run_name
            run_protocol_path = report_run / "PROTOCOL.json"
            evaluation_path = report_run / "evaluation.json"
            history_path = (
                PROJECT_ROOT
                / "checkpoints"
                / "joint-multimodal-seg"
                / run_name
                / "results.csv"
            )
            checkpoint_path = (
                PROJECT_ROOT
                / "checkpoints"
                / "joint-multimodal-seg"
                / run_name
                / "weights"
                / "best.pt"
            )
            required = (
                run_protocol_path,
                evaluation_path,
                history_path,
                checkpoint_path,
            )
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                errors.append(f"{run_name}: missing {missing}")
                continue

            run_protocol = _read_object(run_protocol_path)
            evaluation = _read_object(evaluation_path)
            checkpoint_hash = sha256_file(checkpoint_path)
            checkpoint_hashes.append(checkpoint_hash)
            policy = protocol["training_policy"]
            train_args = run_protocol.get("train_args", {})
            expected_train_args = {
                "epochs": policy["epochs"],
                "patience": policy["patience"],
                "batch": policy["batch_size"],
                "imgsz": policy["image_size"],
                "workers": policy["workers"],
                "freeze": policy["freeze_layers"],
                "amp": policy["amp"],
                "lr0": policy["learning_rate"],
                "weight_decay": policy["weight_decay"],
            }
            for key, expected_value in expected_train_args.items():
                if train_args.get(key) != expected_value:
                    errors.append(
                        f"{run_name}: frozen train argument mismatch: {key}"
                    )
            fusion_args = run_protocol.get("fusion_args", {})
            expected_fusion_args = {
                "fusion_mode": mode,
                "semantic_dim": policy["semantic_dim"],
                "sensor_hidden_dim": policy["sensor_hidden_dim"],
                "text_temperature": policy["text_temperature"],
                "sensor_reconstruction_weight": policy[
                    "sensor_reconstruction_weight"
                ],
                "sensor_calibration_weight": policy[
                    "sensor_calibration_weight"
                ],
            }
            for key, expected_value in expected_fusion_args.items():
                if fusion_args.get(key) != expected_value:
                    errors.append(
                        f"{run_name}: frozen fusion argument mismatch: {key}"
                    )
            if run_protocol.get("seed") != seed:
                errors.append(f"{run_name}: seed mismatch")
            architecture = run_protocol.get("architecture", {})
            if architecture.get("fusion_mode") != mode:
                errors.append(f"{run_name}: architecture mode mismatch")
            expected_text = mode == "text_only"
            expected_sensor = mode == "sensor_only"
            if architecture.get("text_enabled") is not expected_text:
                errors.append(f"{run_name}: text branch flag mismatch")
            if architecture.get("sensor_enabled") is not expected_sensor:
                errors.append(f"{run_name}: sensor branch flag mismatch")
            if evaluation.get("fusion_mode") != mode:
                errors.append(f"{run_name}: evaluation mode mismatch")
            if evaluation.get("forward_contract", {}).get(
                "metric_forward_inputs"
            ) != ["RGB"]:
                errors.append(f"{run_name}: metric forward is not RGB-only")
            if evaluation["checkpoint"].get("sha256") != checkpoint_hash:
                errors.append(f"{run_name}: checkpoint hash mismatch")
            metrics = {
                metric: float(evaluation["metrics"][metric])
                for metric in METRICS
            }
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in metrics.values()
            ):
                errors.append(f"{run_name}: invalid metric value")

            structure = _checkpoint_structure(checkpoint_path)
            if structure["fusion_mode"] != mode:
                errors.append(f"{run_name}: checkpoint mode mismatch")
            if structure["text_enabled"] is not expected_text:
                errors.append(f"{run_name}: checkpoint text flag mismatch")
            if structure["sensor_enabled"] is not expected_sensor:
                errors.append(f"{run_name}: checkpoint sensor flag mismatch")
            text_nonzero = all(
                value > 1e-8 for value in structure["text_output_l2"]
            )
            sensor_nonzero = all(
                value > 1e-8 for value in structure["sensor_output_l2"]
            )
            if text_nonzero is not expected_text:
                errors.append(f"{run_name}: text output activity mismatch")
            if sensor_nonzero is not expected_sensor:
                errors.append(f"{run_name}: sensor output activity mismatch")

            losses = _read_loss_columns(history_path)
            if mode == "text_only" and any(
                abs(value) > 1e-12
                for values in losses.values()
                for value in values
            ):
                errors.append(f"{run_name}: disabled sensor loss is nonzero")
            if mode == "sensor_only" and not any(
                value > 1e-8 for value in losses["train_sensor_loss"]
            ):
                errors.append(f"{run_name}: active sensor loss is zero")

            normalized_train_args = dict(run_protocol["train_args"])
            normalized_train_args.pop("name", None)
            normalized_train_args.pop("seed", None)
            normalized_protocols[mode].append(
                {
                    "train_args": normalized_train_args,
                    "fusion_args": {
                        key: value
                        for key, value in run_protocol["fusion_args"].items()
                        if key != "fusion_mode"
                    },
                    "source_bindings": run_protocol["source_bindings"],
                }
            )
            run_results[mode].append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "completed_epochs": int(
                        evaluation["training"]["completed_epochs"]
                    ),
                    "checkpoint": _binding(checkpoint_path),
                    "metrics": metrics,
                    "structure": structure,
                    "sensor_loss": {
                        "train_last": losses["train_sensor_loss"][-1],
                        "val_last": losses["val_sensor_loss"][-1],
                    },
                }
            )

    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        errors.append("formal ablation checkpoint hashes are not unique")
    for mode in ("text_only", "sensor_only"):
        seeds = sorted(run["seed"] for run in run_results[mode])
        if seeds != [42, 43, 44]:
            errors.append(f"{mode}: completed seed set differs from protocol")
        protocols = normalized_protocols[mode]
        if protocols and any(item != protocols[0] for item in protocols[1:]):
            errors.append(f"{mode}: normalized protocols differ across seeds")

    aggregate: dict[str, dict[str, Any]] = {}
    for mode, runs in run_results.items():
        mode_metrics = {}
        for metric in METRICS:
            values = [run["metrics"][metric] for run in runs]
            summary = _summary(values) if values else {}
            if summary:
                summary.update(
                    {
                        "rgb_baseline": base_metrics[metric],
                        "mean_minus_rgb_baseline": (
                            summary["mean"] - base_metrics[metric]
                        ),
                        "mean_minus_rgb_baseline_percentage_points": (
                            100.0
                            * (summary["mean"] - base_metrics[metric])
                        ),
                        "full_mean": full_metrics[metric],
                        "full_mean_minus_mode_mean": (
                            full_metrics[metric] - summary["mean"]
                        ),
                        "full_mean_minus_mode_mean_percentage_points": (
                            100.0
                            * (full_metrics[metric] - summary["mean"])
                        ),
                    }
                )
            mode_metrics[metric] = summary

        paired = {}
        by_seed = {run["seed"]: run for run in runs}
        for metric in METRICS:
            differences = [
                float(full_runs[seed][metric])
                - by_seed[seed]["metrics"][metric]
                for seed in sorted(by_seed)
            ]
            paired[metric] = {
                "full_minus_mode_by_seed": differences,
                **_summary(differences),
                "mean_percentage_points": 100.0 * mean(differences),
            }
        aggregate[mode] = {
            "run_count": len(runs),
            "metrics": mode_metrics,
            "paired_full_minus_mode": paired,
        }

    outcome = {
        "schema_version": 1,
        "protocol": "joint_head_independently_retrained_branch_ablation_outcome_v1",
        "valid": not errors,
        "errors": errors,
        "independent_retraining": True,
        "same_checkpoint_masking": False,
        "formal_run_count": sum(len(runs) for runs in run_results.values()),
        "reference": {
            "vision_only": {
                "evaluation": _binding(base_evaluation_path),
                "metrics": base_metrics,
            },
            "full": {
                "summary": _binding(full_summary_path),
                "metrics": full_metrics,
                "seeds": sorted(full_runs),
            },
        },
        "runs": run_results,
        "aggregate": aggregate,
        "protocol_binding": _binding(protocol_path),
        "normalized_protocol_sha256": {
            mode: (
                _canonical_sha256(protocols[0]) if protocols else None
            )
            for mode, protocols in normalized_protocols.items()
        },
        "claim_boundaries": protocol["claim_boundaries"],
    }
    return outcome


def _write_report(outcome: dict[str, Any], path: Path) -> None:
    lines = [
        "# 联合头独立重训分支消融",
        "",
        f"- 审计状态：{'通过' if outcome['valid'] else '未通过'}",
        f"- 正式运行数：{outcome['formal_run_count']}",
        "- 对照性质：独立重训，不是同检查点屏蔽",
        "",
        "| 模型 | Box mAP50-95 | Mask mAP50-95 |",
        "|---|---:|---:|",
    ]
    full = outcome["reference"]["full"]["metrics"]
    base = outcome["reference"]["vision_only"]["metrics"]
    lines.append(
        "| 仅 RGB | "
        f"{base['metrics/mAP50-95(B)']:.4f} | "
        f"{base['metrics/mAP50-95(M)']:.4f} |"
    )
    for mode, label in (
        ("text_only", "仅文本适配器"),
        ("sensor_only", "仅预测软传感器适配器"),
    ):
        metrics = outcome["aggregate"][mode]["metrics"]
        box = metrics["metrics/mAP50-95(B)"]
        mask = metrics["metrics/mAP50-95(M)"]
        lines.append(
            f"| {label} | {box['mean']:.4f}±{box['sample_std']:.4f} | "
            f"{mask['mean']:.4f}±{mask['sample_std']:.4f} |"
        )
    lines.append(
        "| 完整联合头 | "
        f"{full['metrics/mAP50-95(B)']:.4f} | "
        f"{full['metrics/mAP50-95(M)']:.4f} |"
    )
    lines.extend(
        [
            "",
            "均值与样本标准差基于种子 42/43/44。所有正式指标前向只接收 RGB；"
            "预测软传感器分支没有使用测试 sidecar 或硬件测量。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = resolve_project_path(args.root)
    outcome = audit(root, args.full_summary, args.base_evaluation)
    output_path = root / "OUTCOME.json"
    output_path.write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(outcome, root / "REPORT.md")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    if not outcome["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
