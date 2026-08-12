"""Audit aggregate evidence for heteroscedastic variance floor collapse."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VARIANTS = {
    "full_quality_aware_prior": {
        "config": "configs/ablations/full_quality_aware_prior.json",
        "history": (
            "checkpoints/ablations/full_quality_aware_prior/history.json"
        ),
    },
    "full_cross_modal_quality": {
        "config": "configs/ablations/full_cross_modal_quality.json",
        "history": (
            "checkpoints/ablations/full_cross_modal_quality/history.json"
        ),
    },
}
NEAR_FLOOR_MARGIN = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit logged NLL and variance-prior evidence for floor collapse"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "UNCERTAINTY_FLOOR_AUDIT.json"
        ),
    )
    return parser.parse_args()


def _resolve(path: Path | str, *, root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(config: Mapping[str, Any]) -> str:
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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _finite_number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def audit_variant(
    *,
    name: str,
    config_path: Path,
    history_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    config = _read_json(config_path)
    history_document = _read_json(history_path)
    if not isinstance(config, Mapping):
        raise ValueError(f"{name}: config root must be an object")
    if not isinstance(history_document, Mapping):
        raise ValueError(f"{name}: history root must be an object")
    rows = history_document.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name}: training history is empty")
    last = rows[-1]
    if not isinstance(last, Mapping):
        raise ValueError(f"{name}: final history row must be an object")

    experiment_name = config.get("experiment", {}).get("name")
    recorded_name = history_document.get("experiment", {}).get("name")
    if experiment_name != name or recorded_name != name:
        raise ValueError(f"{name}: experiment identity mismatch")
    fingerprint = _config_fingerprint(config)
    if history_document.get("config_fingerprint") != fingerprint:
        raise ValueError(f"{name}: configuration fingerprint mismatch")

    model = config.get("model", {})
    loss = config.get("loss", {})
    lower = _finite_number(
        model.get("sensor_log_variance_min"),
        label=f"{name}.sensor_log_variance_min",
    )
    target = _finite_number(
        loss.get("variance_prior_target"),
        label=f"{name}.variance_prior_target",
    )
    prior_weight = _finite_number(
        loss.get("variance_prior_weight"),
        label=f"{name}.variance_prior_weight",
    )
    reconstruction = _finite_number(
        last.get("val_sensor_reconstruction"),
        label=f"{name}.val_sensor_reconstruction",
    )
    prior_mse = _finite_number(
        last.get("val_sensor_variance_prior"),
        label=f"{name}.val_sensor_variance_prior",
    )
    if prior_weight <= 0.0:
        raise ValueError(f"{name}: variance prior is disabled")
    target_distance_sq = (lower - target) ** 2
    if target_distance_sq <= 0.0:
        raise ValueError(f"{name}: variance target equals lower clamp")

    theoretical_floor = 0.5 * lower
    floor_gap = reconstruction - theoretical_floor
    exact_floor_fraction_upper_bound = min(
        1.0,
        max(0.0, prior_mse / target_distance_sq),
    )
    return {
        "experiment": name,
        "epoch_count": len(rows),
        "final_epoch": int(last.get("epoch", -1)),
        "sensor_log_variance_min": lower,
        "variance_prior_target": target,
        "variance_prior_weight": prior_weight,
        "val_sensor_reconstruction": reconstruction,
        "theoretical_nll_floor": theoretical_floor,
        "nll_floor_gap": floor_gap,
        "near_theoretical_floor": floor_gap <= NEAR_FLOOR_MARGIN,
        "val_sensor_variance_prior_mse": prior_mse,
        "exact_lower_clamp_fraction_upper_bound": (
            exact_floor_fraction_upper_bound
        ),
        "source_config_fingerprint": fingerprint,
        "sources": {
            "config": {
                "path": config_path.relative_to(project_root).as_posix(),
                "size_bytes": config_path.stat().st_size,
                "sha256": _sha256(config_path),
            },
            "history": {
                "path": history_path.relative_to(project_root).as_posix(),
                "size_bytes": history_path.stat().st_size,
                "sha256": _sha256(history_path),
            },
        },
    }


def audit_uncertainty_floor(
    *,
    project_root: Path,
    variants: Mapping[str, Mapping[str, str | Path]] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    errors: list[str] = []
    results: dict[str, Any] = {}
    for name, paths in variants.items():
        try:
            results[name] = audit_variant(
                name=name,
                config_path=_resolve(paths["config"], root=project_root),
                history_path=_resolve(paths["history"], root=project_root),
                project_root=project_root,
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(f"{name}: {exc}")
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "failed",
        "valid": not errors,
        "near_floor_margin": NEAR_FLOOR_MARGIN,
        "error_count": len(errors),
        "errors": errors,
        "variants": results,
        "claim_boundary": (
            "Aggregate logged losses do not replace per-dimension variance "
            "histograms and do not establish physical sensor uncertainty "
            "calibration."
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 异方差下界审计",
        "",
        "本审计只检查训练日志中的聚合损失是否出现全局贴近对数方差下界的现象。",
        "它不把潜空间方差解释为物理传感器误差，也不替代逐样本、逐维方差直方图。",
        "",
        "| 变体 | 末轮 | 验证重建 NLL | 理论下界 | 距下界 | 方差先验 MSE | 恰好卡下界占比上界 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report.get("variants", {}).items():
        lines.append(
            f"| `{name}` | {row['final_epoch']} | "
            f"{row['val_sensor_reconstruction']:.6f} | "
            f"{row['theoretical_nll_floor']:.6f} | "
            f"{row['nll_floor_gap']:.6f} | "
            f"{row['val_sensor_variance_prior_mse']:.6f} | "
            f"{100.0 * row['exact_lower_clamp_fraction_upper_bound']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "占比上界来自当前实现的 "
            "`mean((log_variance - target)^2)`：当下界为 -3、目标为 -2 时，"
            "每个恰好卡在下界的潜变量至少贡献 1，因此其占比不超过记录的均方项。",
            "",
            "结论边界：当前 v2b/v3 日志不支持“整体方差贴底”的判断；"
            "这不等于连续可靠性或真实物理不确定性已经校准。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = _resolve(args.output, root=PROJECT_ROOT)
    report = audit_uncertainty_floor(project_root=PROJECT_ROOT)
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown_path = output.with_suffix(".md")
    _atomic_write(markdown_path, _markdown(report))
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "markdown": markdown_path.as_posix(),
                **report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
