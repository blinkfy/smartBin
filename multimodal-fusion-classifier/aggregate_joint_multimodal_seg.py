"""Aggregate audited joint-head runs across independent training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, stdev
from typing import Any, Iterable

from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path


DEFAULT_REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "joint-multimodal-seg"
)
DEFAULT_PROTOCOL = DEFAULT_REPORT_ROOT / "MULTISEED_PROTOCOL.json"
DEFAULT_RUN_NAMES = tuple(
    f"yolo11s-seg-joint-v1-seed{seed}" for seed in (42, 43, 44)
)
PRIMARY_METRICS = (
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
        description="Aggregate audited joint-head runs across seeds."
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=list(DEFAULT_RUN_NAMES),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_REPORT_ROOT / "MULTISEED_SUMMARY.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_REPORT_ROOT / "MULTISEED_SUMMARY.md",
    )
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_training_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    train_args = dict(protocol.get("train_args", {}))
    train_args.pop("name", None)
    train_args.pop("seed", None)
    return {
        "protocol": protocol.get("protocol"),
        "architecture": protocol.get("architecture"),
        "leakage_control": protocol.get("leakage_control"),
        "train_args": train_args,
        "fusion_args": protocol.get("fusion_args"),
        "performance_policy": protocol.get("performance_policy"),
        "claim_boundaries": protocol.get("claim_boundaries"),
        "source_bindings": protocol.get("source_bindings"),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metric_summary(values: Iterable[float]) -> dict[str, float]:
    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("cannot summarize an empty metric")
    return {
        "mean": mean(sample),
        "sample_std": stdev(sample) if len(sample) > 1 else 0.0,
        "min": min(sample),
        "max": max(sample),
    }


def aggregate_runs(
    report_root: Path,
    run_names: list[str],
    multiseed_protocol_path: Path | None = None,
) -> dict[str, Any]:
    report_root = resolve_project_path(report_root)
    multiseed_protocol_path = resolve_project_path(
        multiseed_protocol_path or report_root / DEFAULT_PROTOCOL.name
    )
    multiseed_protocol = (
        _read_object(multiseed_protocol_path)
        if multiseed_protocol_path.is_file()
        else None
    )
    if len(run_names) < 2:
        raise ValueError("at least two independently seeded runs are required")
    if len(run_names) != len(set(run_names)):
        raise ValueError("run names must be unique")
    if multiseed_protocol is not None:
        if multiseed_protocol.get("run_names") != run_names:
            raise ValueError("run names differ from MULTISEED_PROTOCOL.json")

    runs = []
    errors = []
    datasets = []
    base_metrics_by_run = []
    seeds = []
    checkpoint_hashes = []
    normalized_protocols = []
    for run_name in run_names:
        run_dir = report_root / run_name
        run_protocol_path = run_dir / "PROTOCOL.json"
        evaluation_path = run_dir / "evaluation.json"
        outcome_path = run_dir / "OUTCOME.json"
        missing = [
            path.name
            for path in (run_protocol_path, evaluation_path, outcome_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{run_name} is missing audited inputs: {', '.join(missing)}"
            )

        protocol = _read_object(run_protocol_path)
        evaluation = _read_object(evaluation_path)
        outcome = _read_object(outcome_path)
        seed = int(protocol["seed"])
        metrics = evaluation["metrics"]
        checkpoint = evaluation["checkpoint"]
        base_metrics = {
            name: float(comparison["base"])
            for name, comparison in outcome["metric_comparison"].items()
        }
        if protocol.get("run_name") != run_name:
            errors.append(f"{run_name}: protocol run_name mismatch")
        if not outcome.get("valid"):
            errors.append(f"{run_name}: OUTCOME.json is not valid")
        forward_contract = evaluation.get("forward_contract", {})
        metric_inputs = forward_contract.get("metric_forward_inputs")
        rgb_only = metric_inputs == ["RGB"] or forward_contract.get(
            "metric_forward_is_rgb_only"
        ) is True
        if not rgb_only:
            errors.append(f"{run_name}: metric forward is not RGB-only")
        missing_metrics = [name for name in PRIMARY_METRICS if name not in metrics]
        if missing_metrics:
            errors.append(f"{run_name}: missing metrics {missing_metrics}")

        checkpoint_hash = str(checkpoint["sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash) is None:
            errors.append(f"{run_name}: invalid checkpoint SHA-256")
        seeds.append(seed)
        checkpoint_hashes.append(checkpoint_hash)
        normalized_protocols.append(_normalized_training_protocol(protocol))
        datasets.append(outcome["dataset"])
        base_metrics_by_run.append(base_metrics)
        runs.append(
            {
                "run_name": run_name,
                "seed": seed,
                "checkpoint_sha256": checkpoint_hash,
                "completed_epochs": outcome["training"]["completed_epochs"],
                "metrics": {
                    name: float(metrics[name]) for name in PRIMARY_METRICS
                },
            }
        )

    if len(seeds) != len(set(seeds)):
        errors.append("training seeds are not unique")
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        errors.append("best checkpoint hashes are not unique")
    if any(
        protocol != normalized_protocols[0]
        for protocol in normalized_protocols[1:]
    ):
        errors.append("normalized training protocols differ across runs")
    if any(dataset != datasets[0] for dataset in datasets[1:]):
        errors.append("dataset summaries differ across runs")
    if any(
        base_metrics != base_metrics_by_run[0]
        for base_metrics in base_metrics_by_run[1:]
    ):
        errors.append("RGB baseline metrics differ across runs")

    aggregate = {}
    for metric_name in PRIMARY_METRICS:
        summary = _metric_summary(
            run["metrics"][metric_name] for run in runs
        )
        base = base_metrics_by_run[0][metric_name]
        summary["rgb_baseline"] = base
        summary["mean_minus_rgb_baseline"] = summary["mean"] - base
        summary["mean_minus_rgb_baseline_percentage_points"] = (
            100.0 * (summary["mean"] - base)
        )
        aggregate[metric_name] = summary

    return {
        "schema_version": 1,
        "protocol": "audited_joint_head_multiseed_aggregate_v1",
        "valid": not errors,
        "errors": errors,
        "run_count": len(runs),
        "seeds": sorted(seeds),
        "sample_std_definition": "n-1 across independent training seeds",
        "multiseed_protocol": (
            {
                "path": multiseed_protocol_path.resolve().as_posix(),
                "sha256": _sha256_file(multiseed_protocol_path),
                "declaration_status": multiseed_protocol.get(
                    "declaration_status"
                ),
            }
            if multiseed_protocol is not None
            else None
        ),
        "normalized_training_protocol_sha256": _canonical_sha256(
            normalized_protocols[0]
        ),
        "dataset": datasets[0],
        "runs": sorted(runs, key=lambda item: item["seed"]),
        "aggregate": aggregate,
        "claim_boundaries": {
            "internal_source_grouped_test": True,
            "independent_training_seeds": True,
            "external_blind_test": False,
            "hardware_sensor_measurement": False,
            "rgb_d_input": False,
            "full_patent_reproduction": False,
        },
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    box = summary["aggregate"]["metrics/mAP50-95(B)"]
    mask = summary["aggregate"]["metrics/mAP50-95(M)"]
    lines = [
        "# 联合检测/分割头多种子汇总",
        "",
        f"- 独立训练种子：{', '.join(map(str, summary['seeds']))}",
        f"- 审计状态：{'通过' if summary['valid'] else '未通过'}",
        "- 标准差：独立训练种子间样本标准差（n-1）",
        "",
        "| 指标 | 联合头均值 | 样本标准差 | RGB 基线 | 均值变化（百分点） |",
        "|---|---:|---:|---:|---:|",
        (
            "| Box mAP50-95 | "
            f"{box['mean']:.4f} | {box['sample_std']:.4f} | "
            f"{box['rgb_baseline']:.4f} | "
            f"{box['mean_minus_rgb_baseline_percentage_points']:+.2f} |"
        ),
        (
            "| Mask mAP50-95 | "
            f"{mask['mean']:.4f} | {mask['sample_std']:.4f} | "
            f"{mask['rgb_baseline']:.4f} | "
            f"{mask['mean_minus_rgb_baseline_percentage_points']:+.2f} |"
        ),
        "",
        "各运行均须先通过独立 OUTCOME 审计，且正式指标前向只接收 RGB。"
        "本汇总仍属于固定内部 source-grouped 测试集，不代表外部盲测、"
        "RGB-D 或同步硬件传感器实验。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = aggregate_runs(args.report_root, list(args.run_names))
    output_json = resolve_project_path(args.output_json)
    output_markdown = resolve_project_path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, output_markdown)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
