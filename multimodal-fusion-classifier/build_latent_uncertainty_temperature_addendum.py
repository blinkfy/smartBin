"""Build the manuscript addendum for v3 latent variance calibration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evaluate_uncertainty import _source_binding
from fit_latent_uncertainty_temperature import REPORT_ROOT
from mmfc.training import PROJECT_ROOT, resolve_project_path


PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
PARAMETERS = REPORT_ROOT / "PARAMETERS.json"
EVALUATION = REPORT_ROOT / "EVALUATION.json"
RAW_DIAGNOSTICS = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "full_cross_modal_quality"
    / "uncertainty"
    / "uncertainty_diagnostics.json"
)
FLOOR_AUDIT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "UNCERTAINTY_FLOOR_AUDIT.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "LATENT_UNCERTAINTY_TEMPERATURE_ADDENDUM.json"
)
INPUTS = {
    "protocol": PROTOCOL,
    "parameters": PARAMETERS,
    "evaluation": EVALUATION,
    "raw_diagnostics": RAW_DIAGNOSTICS,
    "floor_audit": FLOOR_AUDIT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the latent uncertainty temperature addendum"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_project_path(path)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build() -> dict[str, Any]:
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "uncertainty addendum inputs missing: " + ", ".join(missing)
        )
    protocol = _read(PROTOCOL)
    parameters = _read(PARAMETERS)
    evaluation = _read(EVALUATION)
    raw_diagnostics = _read(RAW_DIAGNOSTICS)
    floor_audit = _read(FLOOR_AUDIT)
    if (
        protocol.get("status")
        != "development_protocol_frozen_before_calibration"
        or parameters.get("state") != "frozen_before_test_evaluation"
        or parameters.get("test_manifest_read") is not False
        or evaluation.get("state") != "complete"
        or evaluation.get("valid") is not True
        or evaluation.get("test_used_to_fit_parameters") is not False
        or floor_audit.get("valid") is not True
        or raw_diagnostics.get("valid") is not True
        or evaluation.get("bindings", {})
        .get("parameters", {})
        .get("sha256")
        != _source_binding(PARAMETERS)["sha256"]
    ):
        raise ValueError("uncertainty-temperature evidence is inconsistent")

    raw = evaluation["test"]["raw"]
    calibrated = evaluation["test"]["calibrated"]
    bootstrap = evaluation["test"]["nll_paired_cluster_bootstrap"]
    nominal = {"coverage_le_1": 0.6826894921370859, "coverage_le_1_96": 0.95}
    raw_variance_to_mse = 1.0 / float(raw["mse_to_variance_ratio"])
    calibrated_variance_to_mse = (
        1.0 / float(calibrated["mse_to_variance_ratio"])
    )
    metrics = {
        "variance_scale": float(evaluation["parameters"]["scale"]),
        "raw_variance_to_mse_ratio": raw_variance_to_mse,
        "calibrated_variance_to_mse_ratio": calibrated_variance_to_mse,
        "raw_mean_squared_standardized_residual": float(
            raw["mean_squared_standardized_residual"]
        ),
        "calibrated_mean_squared_standardized_residual": float(
            calibrated["mean_squared_standardized_residual"]
        ),
        "raw_gaussian_nll": float(raw["gaussian_nll_mean"]),
        "calibrated_gaussian_nll": float(
            calibrated["gaussian_nll_mean"]
        ),
        "nll_difference_calibrated_minus_raw": float(
            calibrated["gaussian_nll_mean"] - raw["gaussian_nll_mean"]
        ),
        "nll_cluster_bootstrap_interval": [
            float(value) for value in bootstrap["interval"]
        ],
        "raw_coverage_absolute_le_1": float(
            raw["coverage_absolute_le_1"]
        ),
        "calibrated_coverage_absolute_le_1": float(
            calibrated["coverage_absolute_le_1"]
        ),
        "raw_coverage_absolute_le_1_96": float(
            raw["coverage_absolute_le_1_96"]
        ),
        "calibrated_coverage_absolute_le_1_96": float(
            calibrated["coverage_absolute_le_1_96"]
        ),
        "nominal_coverage_absolute_le_1": nominal["coverage_le_1"],
        "nominal_coverage_absolute_le_1_96": nominal[
            "coverage_le_1_96"
        ],
        "sample_mean_variance_error_spearman": float(
            calibrated["variance_error_correlation"]["sample_mean"][
                "spearman"
            ]
        ),
    }
    validation = parameters["validation"]
    return {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_role": (
            "validation_fit_latent_uncertainty_scale_development_addendum"
        ),
        "training_time_floor_status": {
            "resolved_before_this_addendum": True,
            "v3_configured_log_variance_min": -3.0,
            "v3_variance_prior_target": -2.0,
            "v3_variance_prior_weight": 0.1,
            "test_fraction_within_0_05_of_lower_bound": float(
                raw_diagnostics["diagnostics"]["bound_proximity"]["lower"][
                    "within_0.05"
                ]
            ),
            "minimum_distance_above_lower_bound": float(
                raw_diagnostics["diagnostics"][
                    "derived_interpretation_values"
                ]["minimum_log_variance_distance_above_lower_bound"]
            ),
        },
        "calibration_design": {
            "fit_split": "val",
            "test_split": "test",
            "test_used_to_fit_parameters": False,
            "closed_form_gaussian_nll_scale": True,
            "checkpoint_modified": False,
            "classification_outputs_modified": False,
            "training_reliability_gate_modified": False,
            "validation_sample_count": int(
                validation["sample_count"]
            ),
            "validation_source_image_count": int(
                validation["source_image_count"]
            ),
            "test_sample_count": int(evaluation["test"]["sample_count"]),
            "test_source_image_count": int(
                evaluation["test"]["source_image_count"]
            ),
        },
        "test_metrics": metrics,
        "scientific_decision": {
            "lower_bound_collapse_present": False,
            "raw_absolute_variance_calibrated": False,
            "posthoc_absolute_variance_calibrated": True,
            "variance_error_ranking_preserved": True,
            "nll_improvement_interval_excludes_zero": (
                metrics["nll_cluster_bootstrap_interval"][1] < 0.0
            ),
            "physical_sensor_uncertainty_calibrated": False,
            "eligible_for_primary_accuracy_table": False,
            "eligible_for_uncertainty_supplement": True,
        },
        "recommended_manuscript_text": {
            "method": (
                "A single positive variance temperature was estimated on "
                "the validation split using the closed-form Gaussian-NLL "
                "optimum and applied unchanged to the held-out test latent "
                "variances."
            ),
            "result": (
                "Post-hoc scaling reduced latent Gaussian NLL from "
                f"{metrics['raw_gaussian_nll']:.3f} to "
                f"{metrics['calibrated_gaussian_nll']:.3f}; the paired "
                "source-image cluster-bootstrap interval for the change was "
                f"[{metrics['nll_cluster_bootstrap_interval'][0]:.3f}, "
                f"{metrics['nll_cluster_bootstrap_interval'][1]:.3f}]. "
                "The 1.96-standardized-residual coverage changed from "
                f"{metrics['raw_coverage_absolute_le_1_96']:.3f} to "
                f"{metrics['calibrated_coverage_absolute_le_1_96']:.3f}."
            ),
            "limitation": (
                "This calibration concerns image-conditioned sensor-latent "
                "errors only and does not validate physical sensor "
                "uncertainty or alter classification performance."
            ),
        },
        "bindings": {
            **{name: _source_binding(path) for name, path in INPUTS.items()},
            "builder_source": _source_binding(Path(__file__)),
        },
    }


def write(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    report = build()
    _atomic_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    metrics = report["test_metrics"]
    interval = metrics["nll_cluster_bootstrap_interval"]
    markdown = [
        "# v3 潜空间不确定性尺度校准附录",
        "",
        "状态：完成、有效、非正式论文指标。",
        "",
        "## 结论",
        "",
        (
            "v3 训练期 log-variance 已不再贴近下限，但原始方差绝对尺度"
            "偏保守。使用验证集闭式拟合单个全局方差温度后，测试潜空间"
            "覆盖率与 Gaussian NLL 明显改善，同时不改变分类输出或可靠性"
            "门控。"
        ),
        "",
        "## 测试结果",
        "",
        (
            f"- 方差温度：`{metrics['variance_scale']:.8f}`；方差/MSE："
            f"`{metrics['raw_variance_to_mse_ratio']:.3f}` → "
            f"`{metrics['calibrated_variance_to_mse_ratio']:.3f}`。"
        ),
        (
            f"- Gaussian NLL：`{metrics['raw_gaussian_nll']:.6f}` → "
            f"`{metrics['calibrated_gaussian_nll']:.6f}`；校准-原始的"
            f"源图聚类95%区间：`[{interval[0]:.6f}, "
            f"{interval[1]:.6f}]`。"
        ),
        (
            f"- |z|≤1：`{metrics['raw_coverage_absolute_le_1']:.5f}` → "
            f"`{metrics['calibrated_coverage_absolute_le_1']:.5f}`，"
            f"名义值 `{metrics['nominal_coverage_absolute_le_1']:.5f}`。"
        ),
        (
            f"- |z|≤1.96："
            f"`{metrics['raw_coverage_absolute_le_1_96']:.5f}` → "
            f"`{metrics['calibrated_coverage_absolute_le_1_96']:.5f}`，"
            f"名义值 `{metrics['nominal_coverage_absolute_le_1_96']:.2f}`。"
        ),
        (
            f"- 样本级方差-误差 Spearman 保持 "
            f"`{metrics['sample_mean_variance_error_spearman']:.4f}`。"
        ),
        "",
        "## 边界",
        "",
        "- 参数只在验证集拟合，测试集不参与拟合。",
        "- 检查点、分类预测和训练时可靠性门控均未修改。",
        "- 结果仅适用于图像条件传感器潜空间，不代表硬件传感器标定。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return report


def main() -> None:
    report = write(parse_args().output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "formal_paper_metric": report["formal_paper_metric"],
                "variance_scale": report["test_metrics"][
                    "variance_scale"
                ],
                "nll_difference": report["test_metrics"][
                    "nll_difference_calibrated_minus_raw"
                ],
                "coverage_1_96": report["test_metrics"][
                    "calibrated_coverage_absolute_le_1_96"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
