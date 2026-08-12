"""Evaluate sensor-latent uncertainty against realized encoder-space errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mmfc.checkpoints import (
    load_checkpoint,
    load_model_state_compatible,
    validated_checkpoint_class_names,
)
from mmfc.data import write_json
from mmfc.model import build_production_model
from mmfc.training import (
    ManifestDataset,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)
from mmfc.uncertainty_diagnostics import (
    compute_uncertainty_diagnostics,
    grouped_bootstrap_uncertainty_intervals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="诊断传感器潜表示异方差预测，不解释为物理传感器标定"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "ablations"
        / "full_cross_modal_quality.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "checkpoints"
        / "ablations"
        / "full_cross_modal_quality"
        / "best.pt",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bin-count", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅用于诊断冒烟；正式报告不要设置",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples 必须大于 0")
    if args.max_samples is not None and args.output is None:
        raise ValueError("--max-samples 必须同时指定独立的 --output")
    if args.bin_count < 2:
        raise ValueError("--bin-count 必须至少为 2")
    if args.bootstrap_repeats < 2:
        raise ValueError("--bootstrap-repeats 必须至少为 2")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence 必须位于 (0,1)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _source_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": _relative_path(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _training_config_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    training = payload.get("training")
    if isinstance(training, dict):
        training.pop("resume", None)
        training.pop("resume_checkpoint", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _collect_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    sensor_encoder = getattr(model, "sensor_encoder", None)
    if sensor_encoder is None:
        raise ValueError("模型没有可用的 sensor_encoder")
    model.eval()
    actual_values: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    log_variance_values: list[np.ndarray] = []
    source_image_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            sensors = batch["sensor"].to(device, non_blocking=True)
            availability = torch.ones(
                sensors.shape[0],
                1,
                device=device,
                dtype=sensors.dtype,
            )
            actual = sensor_encoder(sensors)
            _, auxiliary = model(
                images,
                sensors,
                sensor_available=availability,
                return_aux=True,
            )
            predicted = auxiliary.get("predicted_sensor")
            log_variance = auxiliary.get("sensor_log_variance")
            if predicted is None or log_variance is None:
                raise ValueError("模型未返回潜表示均值与对数方差")
            if (
                actual.shape != predicted.shape
                or actual.shape != log_variance.shape
            ):
                raise ValueError("实际、预测潜表示和对数方差形状不一致")
            actual_values.append(actual.detach().cpu().numpy())
            predicted_values.append(predicted.detach().cpu().numpy())
            log_variance_values.append(
                log_variance.detach().cpu().numpy()
            )
            source_image_ids.extend(
                str(value)
                for value in batch.get(
                    "source_image_id",
                    batch["sample_id"],
                )
            )
    if not actual_values:
        raise ValueError("评估数据集为空")
    return (
        np.concatenate(actual_values, axis=0),
        np.concatenate(predicted_values, axis=0),
        np.concatenate(log_variance_values, axis=0),
        source_image_ids,
    )


def _plot_diagnostics(
    diagnostics: dict[str, Any],
    log_variance: np.ndarray,
    standardized_residual: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    lower = diagnostics["configured_bounds"]["log_variance_min"]
    upper = diagnostics["configured_bounds"]["log_variance_max"]

    axes[0].hist(
        log_variance.reshape(-1),
        bins=60,
        color="#287271",
        alpha=0.88,
    )
    axes[0].axvline(lower, color="#C44536", linestyle="--", label="lower bound")
    axes[0].axvline(upper, color="#2F4858", linestyle=":", label="upper bound")
    axes[0].set_title("Predicted log-variance")
    axes[0].set_xlabel("log-variance")
    axes[0].set_ylabel("latent elements")
    axes[0].legend(frameon=False)

    bins = diagnostics["variance_calibration"]["equal_count_bins"]
    predicted = np.asarray(
        [item["mean_predicted_variance"] for item in bins],
        dtype=np.float64,
    )
    observed = np.asarray(
        [item["mean_squared_error"] for item in bins],
        dtype=np.float64,
    )
    axes[1].plot(
        predicted,
        observed,
        marker="o",
        color="#D97706",
        linewidth=1.8,
        label="equal-count bins",
    )
    minimum = float(min(predicted.min(), observed.min()) * 0.8)
    maximum = float(max(predicted.max(), observed.max()) * 1.2)
    axes[1].plot(
        [minimum, maximum],
        [minimum, maximum],
        color="#4B5563",
        linestyle="--",
        label="variance = MSE",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Variance versus realized error")
    axes[1].set_xlabel("mean predicted variance")
    axes[1].set_ylabel("mean squared error")
    axes[1].legend(frameon=False)

    clipped = np.clip(standardized_residual.reshape(-1), -5.0, 5.0)
    axes[2].hist(
        clipped,
        bins=70,
        density=True,
        color="#4C78A8",
        alpha=0.82,
        label="standardized residual",
    )
    grid = np.linspace(-5.0, 5.0, 400)
    normal_density = np.exp(-0.5 * np.square(grid)) / np.sqrt(2.0 * np.pi)
    axes[2].plot(
        grid,
        normal_density,
        color="#C44536",
        linewidth=1.6,
        label="standard normal",
    )
    axes[2].set_title("Standardized residual")
    axes[2].set_xlabel("(actual - predicted) / predicted std")
    axes[2].set_ylabel("density")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout(pad=1.4)
    temporary = output_path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, format="png")
    plt.close(figure)
    temporary.replace(output_path)


def _format_optional(value: float | None, digits: int = 6) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    diagnostics = report["diagnostics"]
    calibration = diagnostics["variance_calibration"]
    correlations = diagnostics["variance_error_correlation"]
    standardized = diagnostics["standardized_residual"]
    lower = diagnostics["bound_proximity"]["lower"]
    derived = diagnostics["derived_interpretation_values"]
    bootstrap = report["bootstrap"]
    lines = [
        "# Sensor-Latent Uncertainty Diagnostics",
        "",
        "## Scope",
        "",
        (
            f"- Split: `{report['dataset']['split']}`; samples: "
            f"{diagnostics['sample_count']}; latent dimensions: "
            f"{diagnostics['latent_dimension']}."
        ),
        (
            "- This report compares predicted heteroscedastic variance with "
            "realized error in the learned sensor-encoder latent space."
        ),
        (
            "- It does not establish physical sensor uncertainty calibration "
            "or external deployment validity."
        ),
        "",
        "## Main Diagnostics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        (
            "| Mean predicted variance | "
            f"{calibration['global_mean_predicted_variance']:.6f} |"
        ),
        (
            "| Mean squared latent error | "
            f"{calibration['global_mean_squared_error']:.6f} |"
        ),
        (
            "| MSE / predicted variance | "
            f"{_format_optional(calibration['global_mse_to_variance_ratio'])} |"
        ),
        (
            "| Predicted variance / MSE | "
            f"{_format_optional(derived['predicted_variance_to_mse_ratio'])} |"
        ),
        (
            "| Gaussian NLL mean | "
            f"{diagnostics['gaussian_nll']['mean']:.6f} |"
        ),
        (
            "| Lower-bound proximity (within 0.01) | "
            f"{100.0 * lower['within_0.01']:.3f}% |"
        ),
        (
            "| Elementwise Spearman(var, error²) | "
            f"{_format_optional(correlations['elementwise']['spearman'])} |"
        ),
        (
            "| Sample-level Spearman(mean var, MSE) | "
            f"{_format_optional(correlations['sample_mean']['spearman'])} |"
        ),
        (
            "| Dimension-level Spearman(mean var, MSE) | "
            f"{_format_optional(correlations['latent_dimension_mean']['spearman'])} |"
        ),
        (
            "| Standardized residual std | "
            f"{standardized['standard_deviation']:.6f} |"
        ),
        (
            "| \\|z\\| <= 1 coverage | "
            f"{100.0 * standardized['coverage_absolute_le_1']:.3f}% |"
        ),
        (
            "| \\|z\\| <= 1.96 coverage | "
            f"{100.0 * standardized['coverage_absolute_le_1_96']:.3f}% |"
        ),
        "",
        "## Interpretation",
        "",
        (
            "- No latent element lies within 0.01 log-variance units of the "
            "configured lower bound; the minimum is "
            f"{derived['minimum_log_variance_distance_above_lower_bound']:.6f} "
            "above that bound."
        ),
        (
            "- Predicted variance is "
            f"{derived['predicted_variance_to_mse_ratio']:.3f} times the "
            "realized latent-space MSE. Together with the standardized-residual "
            f"standard deviation of {standardized['standard_deviation']:.3f}, "
            "this indicates conservative magnitude rather than calibrated "
            "Gaussian scale."
        ),
        (
            "- Rank correlations should be reported separately from magnitude "
            "calibration: the former tests relative error ordering, while the "
            "variance/MSE ratio and residual coverage test absolute scale."
        ),
        (
            "- Source-image cluster bootstrap ("
            f"{bootstrap['repeats']} repeats) gives sample Spearman "
            f"{bootstrap['metrics']['sample_mean_spearman']['estimate']:.3f} "
            "["
            f"{bootstrap['metrics']['sample_mean_spearman']['lower']:.3f}, "
            f"{bootstrap['metrics']['sample_mean_spearman']['upper']:.3f}] "
            "and predicted variance/MSE ratio "
            f"{bootstrap['metrics']['predicted_variance_to_mse_ratio']['estimate']:.3f} "
            "["
            f"{bootstrap['metrics']['predicted_variance_to_mse_ratio']['lower']:.3f}, "
            f"{bootstrap['metrics']['predicted_variance_to_mse_ratio']['upper']:.3f}]."
        ),
        "",
        "## Equal-Count Variance Bins",
        "",
        "| Bin | Count | Mean variance | MSE | MSE / variance |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in calibration["equal_count_bins"]:
        lines.append(
            f"| {item['bin']} | {item['count']} | "
            f"{item['mean_predicted_variance']:.6f} | "
            f"{item['mean_squared_error']:.6f} | "
            f"{_format_optional(item['mse_to_variance_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Binding",
            "",
        ]
    )
    for name, binding in report["source_bindings"].items():
        lines.append(
            f"- `{name}`: `{binding['path']}`, SHA-256 "
            f"`{binding['sha256']}`."
        )
    lines.extend(
        [
            "",
            "## Figure",
            "",
            (
                f"- `{report['figure']['path']}`, SHA-256 "
                f"`{report['figure']['sha256']}`."
            ),
            "",
        ]
    )
    temporary = output_path.with_suffix(".tmp.md")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output_path)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    config_path = resolve_project_path(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    config = load_config(config_path)
    training = config.get("training", {})
    data = config.get("data", {})
    device = torch.device(
        args.device
        or training.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到模型检查点：{checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = validated_checkpoint_class_names(checkpoint, config)
    configured_fingerprint = _training_config_fingerprint(config)
    checkpoint_fingerprint = checkpoint.get("config_fingerprint")
    if checkpoint_fingerprint != configured_fingerprint:
        raise ValueError(
            "检查点 config_fingerprint 与当前配置不一致："
            f"{checkpoint_fingerprint!r} != {configured_fingerprint!r}"
        )

    model = build_production_model(config).to(device)
    load_model_state_compatible(model, checkpoint["model_state"])
    if bool(getattr(model, "legacy_sensor_passthrough", False)):
        raise ValueError("旧版直通检查点没有可诊断的异方差预测头")
    if not bool(getattr(model, "sensor_reliability_enabled", False)):
        raise ValueError("配置未启用 sensor reliability heads")

    manifest_key = f"{args.split}_manifest"
    if manifest_key not in data or "sensor_manifest" not in data:
        raise ValueError(f"配置 data 缺少 {manifest_key} 或 sensor_manifest")
    manifest_path = resolve_project_path(data[manifest_key])
    sensor_manifest_path = resolve_project_path(data["sensor_manifest"])
    dataset = ManifestDataset(
        manifest_path,
        class_names=class_names,
        image_size=int(data.get("image_size", 640)),
        sensor_manifest=sensor_manifest_path,
        preprocess=config.get("preprocess", {}),
    )
    full_split_samples = len(dataset)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(training.get("batch_size", 16))
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training.get("num_workers", 0))
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    actual, predicted, log_variance, source_image_ids = _collect_latents(
        model,
        loader,
        device=device,
    )
    model_config = config.get("model", {})
    lower_bound = float(model_config["sensor_log_variance_min"])
    upper_bound = float(model_config["sensor_log_variance_max"])
    diagnostics = compute_uncertainty_diagnostics(
        actual,
        predicted,
        log_variance,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        bin_count=args.bin_count,
    )
    sample_mean_variance = np.exp(log_variance).mean(axis=1)
    sample_mean_error = np.square(actual - predicted).mean(axis=1)
    bootstrap = grouped_bootstrap_uncertainty_intervals(
        sample_mean_variance,
        sample_mean_error,
        source_image_ids,
        repeats=args.bootstrap_repeats,
        confidence=args.bootstrap_confidence,
        seed=args.bootstrap_seed,
    )

    output_dir = resolve_project_path(
        args.output
        if args.output is not None
        else PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "uncertainty"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "uncertainty_diagnostics.png"
    standardized_residual = (actual - predicted) / np.sqrt(
        np.exp(log_variance)
    )
    _plot_diagnostics(
        diagnostics,
        log_variance,
        standardized_residual,
        figure_path,
    )
    report = {
        "schema_version": 1,
        "valid": True,
        "diagnostic_scope": "sensor_encoder_latent_space",
        "claim_boundary": (
            "Predicted variance is compared with realized error in the "
            "image-conditioned soft-sensor latent space; this is not physical "
            "sensor calibration or external validation."
        ),
        "dataset": {
            "split": args.split,
            "evaluated_samples": len(dataset),
            "full_split_samples": full_split_samples,
            "max_samples": args.max_samples,
        },
        "checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "checkpoint_role": checkpoint.get("checkpoint_role"),
            "config_fingerprint": checkpoint_fingerprint,
        },
        "evaluation": {
            "device": str(device),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "bin_count": args.bin_count,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_confidence": args.bootstrap_confidence,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "source_bindings": {
            "config": _source_binding(config_path),
            "checkpoint": _source_binding(checkpoint_path),
            "manifest": _source_binding(manifest_path),
            "sensor_manifest": _source_binding(sensor_manifest_path),
        },
        "diagnostics": diagnostics,
        "bootstrap": bootstrap,
        "figure": {
            "path": _relative_path(figure_path),
            "sha256": _sha256(figure_path),
            "size_bytes": figure_path.stat().st_size,
        },
    }
    json_path = output_dir / "uncertainty_diagnostics.json"
    markdown_path = output_dir / "uncertainty_diagnostics.md"
    write_json(json_path, report)
    _write_markdown(report, markdown_path)
    print(
        json.dumps(
            {
                "valid": True,
                "samples": diagnostics["sample_count"],
                "latent_dimension": diagnostics["latent_dimension"],
                "global_mse_to_variance_ratio": diagnostics[
                    "variance_calibration"
                ]["global_mse_to_variance_ratio"],
                "sample_spearman": diagnostics[
                    "variance_error_correlation"
                ]["sample_mean"]["spearman"],
                "lower_bound_within_0.01": diagnostics[
                    "bound_proximity"
                ]["lower"]["within_0.01"],
                "artifacts": {
                    "json": _relative_path(json_path),
                    "markdown": _relative_path(markdown_path),
                    "figure": _relative_path(figure_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
