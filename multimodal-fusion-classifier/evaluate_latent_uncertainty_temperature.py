"""Evaluate frozen validation-fit latent variance temperature on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_uncertainty import _source_binding
from fit_latent_uncertainty_temperature import (
    CHECKPOINT,
    CONFIG,
    DEFAULT_OUTPUT as PARAMETERS,
    PROTOCOL,
    PROTOCOL_NAME,
    _atomic_json,
    _read_json,
    _resolve,
    collect_split_latents,
)
from mmfc.training import PROJECT_ROOT
from mmfc.uncertainty_temperature import (
    apply_variance_temperature,
    grouped_bootstrap_mean_difference,
    per_sample_gaussian_nll,
    summarize_uncertainty_scale,
)


DEFAULT_OUTPUT = PROTOCOL.parent / "EVALUATION.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen v3 latent variance scale on test"
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--parameters", type=Path, default=PARAMETERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    return parser.parse_args()


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    raw = report["test"]["raw"]
    calibrated = report["test"]["calibrated"]
    interval = report["test"]["nll_paired_cluster_bootstrap"]["interval"]
    lines = [
        "# v3 潜空间方差温度校准",
        "",
        "- 状态：验证集拟合、测试集一次评估；非正式论文指标。",
        f"- 全局方差尺度：`{report['parameters']['scale']:.8f}`。",
        (
            f"- 测试 Gaussian NLL：`{raw['gaussian_nll_mean']:.6f}` → "
            f"`{calibrated['gaussian_nll_mean']:.6f}`。"
        ),
        (
            "- NLL 配对源图聚类 bootstrap 差值（校准-原始）："
            f"`[{interval[0]:.6f}, {interval[1]:.6f}]`。"
        ),
        (
            f"- |z|≤1 覆盖率：`{raw['coverage_absolute_le_1']:.4f}` → "
            f"`{calibrated['coverage_absolute_le_1']:.4f}`；"
            f"|z|≤1.96：`{raw['coverage_absolute_le_1_96']:.4f}` → "
            f"`{calibrated['coverage_absolute_le_1_96']:.4f}`。"
        ),
        "",
        (
            "该标定只修正图像条件传感器潜表示的方差尺度，不改分类预测、"
            "检查点或训练时可靠性门控，不能解释为物理传感器标定。"
        ),
        "",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = _resolve(args.protocol)
    parameters_path = _resolve(args.parameters)
    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    output = _resolve(args.output)
    if output.is_file():
        raise FileExistsError(
            "refusing to overwrite uncertainty-temperature evaluation"
        )
    for path in (
        protocol_path,
        parameters_path,
        config_path,
        checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"temperature input is missing: {path}")
    protocol = _read_json(protocol_path)
    parameters_report = _read_json(parameters_path)
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or parameters_report.get("state")
        != "frozen_before_test_evaluation"
        or parameters_report.get("valid") is not True
        or parameters_report.get("test_manifest_read") is not False
        or parameters_report.get("bindings", {})
        .get("protocol", {})
        .get("sha256")
        != _source_binding(protocol_path)["sha256"]
    ):
        raise ValueError("frozen variance parameters are inconsistent")

    collected = collect_split_latents(
        config_path,
        checkpoint_path,
        "test",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device_name=str(args.device),
    )
    scale = float(parameters_report["parameters"]["scale"])
    raw_log_variance = collected["log_variance"]
    calibrated_log_variance = apply_variance_temperature(
        raw_log_variance, scale
    )
    raw = summarize_uncertainty_scale(
        collected["actual"],
        collected["predicted"],
        raw_log_variance,
    )
    calibrated = summarize_uncertainty_scale(
        collected["actual"],
        collected["predicted"],
        calibrated_log_variance,
    )
    raw_nll = per_sample_gaussian_nll(
        collected["actual"],
        collected["predicted"],
        raw_log_variance,
    )
    calibrated_nll = per_sample_gaussian_nll(
        collected["actual"],
        collected["predicted"],
        calibrated_log_variance,
    )
    bootstrap = grouped_bootstrap_mean_difference(
        raw_nll,
        calibrated_nll,
        collected["source_image_ids"],
        repeats=int(args.bootstrap_repeats),
        seed=int(args.bootstrap_seed),
    )
    nominal_one = 0.6826894921370859
    nominal_one_96 = 0.95
    scientific_diagnostics = {
        "test_nll_below_raw": (
            calibrated["gaussian_nll_mean"] < raw["gaussian_nll_mean"]
        ),
        "bootstrap_nll_interval_below_zero": bootstrap["interval"][1] < 0.0,
        "coverage_le_1_gap_reduced": (
            abs(calibrated["coverage_absolute_le_1"] - nominal_one)
            < abs(raw["coverage_absolute_le_1"] - nominal_one)
        ),
        "coverage_le_1_96_gap_reduced": (
            abs(
                calibrated["coverage_absolute_le_1_96"]
                - nominal_one_96
            )
            < abs(raw["coverage_absolute_le_1_96"] - nominal_one_96)
        ),
        "elementwise_spearman_preserved": (
            calibrated["variance_error_correlation"]["elementwise"][
                "spearman"
            ]
            == raw["variance_error_correlation"]["elementwise"]["spearman"]
        ),
    }
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "protocol": PROTOCOL_NAME,
        "parameter_fit_split": "val",
        "evaluation_split": "test",
        "test_used_to_fit_parameters": False,
        "parameters": parameters_report["parameters"],
        "test": {
            "sample_count": collected["sample_count"],
            "source_image_count": collected["source_image_count"],
            "raw": raw,
            "calibrated": calibrated,
            "nll_paired_cluster_bootstrap": bootstrap,
        },
        "scientific_diagnostics": scientific_diagnostics,
        "runtime": {
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(args.device),
        },
        "bindings": {
            "protocol": _source_binding(protocol_path),
            "parameters": _source_binding(parameters_path),
            **collected["bindings"],
            "evaluator_source": _source_binding(Path(__file__)),
            "numerical_source": _source_binding(
                PROJECT_ROOT / "mmfc" / "uncertainty_temperature.py"
            ),
        },
        "claim_boundary": (
            "The calibrated scale is a post-hoc latent-space development "
            "diagnostic. It is not hardware uncertainty calibration and is "
            "not fed back into classification or the trained gate."
        ),
    }
    _atomic_json(output, report)
    _write_markdown(output.with_suffix(".md"), report)
    return report


def main() -> None:
    result = evaluate(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "scale": result["parameters"]["scale"],
                "test_raw_nll": result["test"]["raw"][
                    "gaussian_nll_mean"
                ],
                "test_calibrated_nll": result["test"]["calibrated"][
                    "gaussian_nll_mean"
                ],
                "bootstrap_interval": result["test"][
                    "nll_paired_cluster_bootstrap"
                ]["interval"],
                "scientific_diagnostics": result[
                    "scientific_diagnostics"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
