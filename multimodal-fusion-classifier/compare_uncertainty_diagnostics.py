"""Compare locked v2b and v3 sensor-latent uncertainty diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_SOURCES = {
    "v2b": (
        "full_quality_aware_prior/uncertainty/"
        "uncertainty_diagnostics.json"
    ),
    "v3": (
        "full_cross_modal_quality/uncertainty/"
        "uncertainty_diagnostics.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较 v2b/v3 固定测试集潜表示不确定性诊断"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"诊断报告必须是 JSON object：{path}")
    return value


def _metric_row(report: Mapping[str, Any]) -> dict[str, float]:
    diagnostics = report["diagnostics"]
    correlations = diagnostics["variance_error_correlation"]
    standardized = diagnostics["standardized_residual"]
    derived = diagnostics["derived_interpretation_values"]
    lower = diagnostics["bound_proximity"]["lower"]
    return {
        "predicted_variance_to_mse_ratio": float(
            derived["predicted_variance_to_mse_ratio"]
        ),
        "elementwise_spearman": float(
            correlations["elementwise"]["spearman"]
        ),
        "sample_mean_spearman": float(
            correlations["sample_mean"]["spearman"]
        ),
        "latent_dimension_mean_spearman": float(
            correlations["latent_dimension_mean"]["spearman"]
        ),
        "standardized_residual_standard_deviation": float(
            standardized["standard_deviation"]
        ),
        "coverage_absolute_le_1": float(
            standardized["coverage_absolute_le_1"]
        ),
        "coverage_absolute_le_1_96": float(
            standardized["coverage_absolute_le_1_96"]
        ),
        "minimum_log_variance_distance_above_lower_bound": float(
            derived[
                "minimum_log_variance_distance_above_lower_bound"
            ]
        ),
        "lower_bound_proximity_within_0_1": float(lower["within_0.1"]),
    }


def _paired_bootstrap_comparison(
    v2b_report: Mapping[str, Any],
    v3_report: Mapping[str, Any],
) -> dict[str, Any]:
    v2b = v2b_report.get("bootstrap")
    v3 = v3_report.get("bootstrap")
    if not isinstance(v2b, Mapping) or not isinstance(v3, Mapping):
        raise ValueError("uncertainty reports require bootstrap evidence")
    identity_keys = (
        "method",
        "repeats",
        "confidence",
        "seed",
        "sample_count",
        "cluster_count",
        "source_group_assignment_sha256",
    )
    for key in identity_keys:
        if v2b.get(key) != v3.get(key):
            raise ValueError(f"uncertainty bootstrap mismatch: {key}")
    confidence = float(v3["confidence"])
    alpha = 1.0 - confidence
    metrics = {}
    specifications = {
        "sample_mean_spearman_increase": (
            "sample_mean_spearman",
            "v3_minus_v2b",
        ),
        "predicted_variance_to_mse_ratio_reduction": (
            "predicted_variance_to_mse_ratio",
            "v2b_minus_v3",
        ),
    }
    for output_name, (source_name, direction) in specifications.items():
        try:
            v2b_values = np.asarray(
                v2b["metrics"][source_name]["replicates"],
                dtype=np.float64,
            )
            v3_values = np.asarray(
                v3["metrics"][source_name]["replicates"],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid bootstrap replicates: {source_name}"
            ) from exc
        repeats = int(v3["repeats"])
        if (
            v2b_values.shape != (repeats,)
            or v3_values.shape != (repeats,)
            or not bool(np.isfinite(v2b_values).all())
            or not bool(np.isfinite(v3_values).all())
        ):
            raise ValueError(
                f"invalid bootstrap replicate shape: {source_name}"
            )
        difference = (
            v3_values - v2b_values
            if direction == "v3_minus_v2b"
            else v2b_values - v3_values
        )
        estimate = (
            float(v3["metrics"][source_name]["estimate"])
            - float(v2b["metrics"][source_name]["estimate"])
            if direction == "v3_minus_v2b"
            else float(v2b["metrics"][source_name]["estimate"])
            - float(v3["metrics"][source_name]["estimate"])
        )
        metrics[output_name] = {
            "estimate": estimate,
            "lower": float(np.quantile(difference, alpha / 2.0)),
            "upper": float(
                np.quantile(difference, 1.0 - alpha / 2.0)
            ),
            "probability_greater_than_zero": float(
                np.mean(difference > 0.0)
            ),
            "preferred_direction": "greater_than_zero",
        }
    return {
        "method": "paired_source_image_cluster_percentile_bootstrap",
        "posthoc_diagnostic": True,
        "repeats": int(v3["repeats"]),
        "confidence": confidence,
        "seed": int(v3["seed"]),
        "sample_count": int(v3["sample_count"]),
        "cluster_count": int(v3["cluster_count"]),
        "source_group_assignment_sha256": v3[
            "source_group_assignment_sha256"
        ],
        "metrics": metrics,
    }


def build_uncertainty_comparison(
    v2b_report: Mapping[str, Any],
    v3_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic comparison without significance overclaims."""

    for name, report in (("v2b", v2b_report), ("v3", v3_report)):
        if (
            report.get("schema_version") != 1
            or report.get("valid") is not True
            or report.get("diagnostic_scope")
            != "sensor_encoder_latent_space"
        ):
            raise ValueError(f"{name} uncertainty report is not valid")
        dataset = report.get("dataset", {})
        if dataset.get("split") != "test" or dataset.get("max_samples") is not None:
            raise ValueError(f"{name} uncertainty report is not a full test run")
    v2b_diagnostics = v2b_report["diagnostics"]
    v3_diagnostics = v3_report["diagnostics"]
    for key in ("sample_count", "latent_dimension", "element_count"):
        if v2b_diagnostics.get(key) != v3_diagnostics.get(key):
            raise ValueError(f"uncertainty report shape mismatch: {key}")
    for binding_name in ("manifest", "sensor_manifest"):
        v2b_binding = v2b_report["source_bindings"][binding_name]
        v3_binding = v3_report["source_bindings"][binding_name]
        if (
            v2b_binding.get("sha256") != v3_binding.get("sha256")
            or v2b_binding.get("size_bytes") != v3_binding.get("size_bytes")
        ):
            raise ValueError(
                f"uncertainty report data source mismatch: {binding_name}"
            )

    v2b = _metric_row(v2b_report)
    v3 = _metric_row(v3_report)
    deltas = {
        key: float(v3[key] - v2b[key])
        for key in v2b
    }
    v2b_overdispersion = v2b["predicted_variance_to_mse_ratio"]
    v3_overdispersion = v3["predicted_variance_to_mse_ratio"]
    return {
        "sample_count": int(v3_diagnostics["sample_count"]),
        "latent_dimension": int(v3_diagnostics["latent_dimension"]),
        "element_count": int(v3_diagnostics["element_count"]),
        "models": {"v2b": v2b, "v3": v3},
        "deltas_v3_minus_v2b": deltas,
        "derived": {
            "predicted_variance_to_mse_ratio_reduction": float(
                v2b_overdispersion - v3_overdispersion
            ),
            "predicted_variance_to_mse_ratio_relative_reduction": float(
                1.0 - v3_overdispersion / v2b_overdispersion
            ),
            "sample_mean_spearman_increase": deltas[
                "sample_mean_spearman"
            ],
            "elementwise_spearman_increase": deltas[
                "elementwise_spearman"
            ],
        },
        "paired_bootstrap": _paired_bootstrap_comparison(
            v2b_report,
            v3_report,
        ),
        "interpretation": (
            "v3 improves relative variance-error ranking and moves the "
            "variance/MSE ratio toward 1 compared with v2b, while both remain "
            "strongly conservative in absolute latent-space scale."
        ),
        "claim_boundary": (
            "The paired intervals are post-hoc diagnostics rather than "
            "preregistered primary endpoints and do not establish physical "
            "sensor calibration or external validity."
        ),
    }


def _source_binding(path: Path, reports_root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(reports_root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    comparison = report["comparison"]
    models = comparison["models"]
    derived = comparison["derived"]
    paired = comparison["paired_bootstrap"]
    lines = [
        "# v2b-v3 Sensor-Latent Uncertainty Comparison",
        "",
        (
            f"Fixed internal test set: {comparison['sample_count']} samples, "
            f"{comparison['latent_dimension']} latent dimensions and "
            f"{comparison['element_count']} latent elements."
        ),
        "",
        "| Metric | v2b | v3 | v3 - v2b |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "predicted_variance_to_mse_ratio": "Predicted variance / MSE",
        "elementwise_spearman": "Elementwise Spearman",
        "sample_mean_spearman": "Sample-mean Spearman",
        "latent_dimension_mean_spearman": "Dimension-mean Spearman",
        "standardized_residual_standard_deviation": "Standardized residual std",
        "coverage_absolute_le_1": "\\|z\\| <= 1 coverage",
        "coverage_absolute_le_1_96": "\\|z\\| <= 1.96 coverage",
        "minimum_log_variance_distance_above_lower_bound": (
            "Minimum distance above lower bound"
        ),
        "lower_bound_proximity_within_0_1": (
            "Lower-bound proximity within 0.1"
        ),
    }
    for key, label in labels.items():
        lines.append(
            f"| {label} | {models['v2b'][key]:.6f} | "
            f"{models['v3'][key]:.6f} | "
            f"{comparison['deltas_v3_minus_v2b'][key]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- {comparison['interpretation']}",
            (
                "- The variance/MSE over-dispersion ratio decreases by "
                f"{derived['predicted_variance_to_mse_ratio_reduction']:.3f} "
                f"({100.0 * derived['predicted_variance_to_mse_ratio_relative_reduction']:.2f}%)."
            ),
            (
                "- Sample-mean Spearman increases by "
                f"{derived['sample_mean_spearman_increase']:.3f}; "
                "the paired source-image cluster bootstrap interval is ["
                f"{paired['metrics']['sample_mean_spearman_increase']['lower']:.3f}, "
                f"{paired['metrics']['sample_mean_spearman_increase']['upper']:.3f}]."
            ),
            (
                "- The paired interval for reduction in predicted "
                "variance/MSE ratio is ["
                f"{paired['metrics']['predicted_variance_to_mse_ratio_reduction']['lower']:.3f}, "
                f"{paired['metrics']['predicted_variance_to_mse_ratio_reduction']['upper']:.3f}]."
            ),
            "- These intervals are post-hoc diagnostics, not preregistered primary endpoints.",
            f"- Boundary: {comparison['claim_boundary']}",
            "",
            "## Evidence Binding",
            "",
        ]
    )
    for name, binding in report["source_reports"].items():
        lines.append(
            f"- `{name}`: `{binding['path']}`, SHA-256 "
            f"`{binding['sha256']}`."
        )
    lines.append("")
    temporary = path.with_suffix(".tmp.md")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    reports_root = _resolve_project_path(args.reports)
    source_paths = {
        name: reports_root / relative
        for name, relative in DEFAULT_SOURCES.items()
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"找不到不确定性诊断报告：{path}")
    source_reports = {
        name: _read_report(path)
        for name, path in source_paths.items()
    }
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "source_reports": {
            name: _source_binding(path, reports_root)
            for name, path in source_paths.items()
        },
        "comparison": build_uncertainty_comparison(
            source_reports["v2b"],
            source_reports["v3"],
        ),
    }
    output_path = _resolve_project_path(
        args.output
        if args.output is not None
        else reports_root / "UNCERTAINTY_DIAGNOSTICS_COMPARISON.json"
    )
    markdown_path = output_path.with_suffix(".md")
    _write_json(output_path, report)
    _write_markdown(report, markdown_path)
    print(
        json.dumps(
            {
                "valid": True,
                "output": output_path.as_posix(),
                "markdown": markdown_path.as_posix(),
                "derived": report["comparison"]["derived"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
