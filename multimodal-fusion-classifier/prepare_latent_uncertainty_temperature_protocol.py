"""Freeze the validation-fit latent variance-temperature protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evaluate_uncertainty import _source_binding
from fit_latent_uncertainty_temperature import (
    CHECKPOINT,
    CONFIG,
    DEFAULT_OUTPUT as PARAMETERS,
    PROTOCOL_NAME,
    REPORT_ROOT,
)
from mmfc.training import PROJECT_ROOT, load_config, resolve_project_path


DEFAULT_OUTPUT = REPORT_ROOT / "PROTOCOL.json"
EXISTING_DIAGNOSTICS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze v3 latent variance-temperature protocol"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_project_path(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def build_protocol() -> dict[str, Any]:
    config = load_config(CONFIG)
    data = config["data"]
    paths = {
        "config": CONFIG,
        "checkpoint": CHECKPOINT,
        "train_manifest": _resolve(Path(data["train_manifest"])),
        "val_manifest": _resolve(Path(data["val_manifest"])),
        "test_manifest": _resolve(Path(data["test_manifest"])),
        "sensor_manifest": _resolve(Path(data["sensor_manifest"])),
        "existing_test_diagnostics": EXISTING_DIAGNOSTICS,
        "uncertainty_floor_audit": FLOOR_AUDIT,
        "numerical_source": (
            PROJECT_ROOT / "mmfc" / "uncertainty_temperature.py"
        ),
        "fitter_source": (
            PROJECT_ROOT / "fit_latent_uncertainty_temperature.py"
        ),
        "evaluator_source": (
            PROJECT_ROOT / "evaluate_latent_uncertainty_temperature.py"
        ),
        "protocol_builder_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "uncertainty-temperature inputs missing: " + ", ".join(missing)
        )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "status": "development_protocol_frozen_before_calibration",
        "formal_paper_metric": False,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Correct the absolute scale of the v3 image-conditioned "
            "sensor-latent Gaussian variance without changing predictions, "
            "checkpoint weights, or the trained reliability gate."
        ),
        "model": {
            "variant": "full_cross_modal_quality_v3",
            "checkpoint": _source_binding(CHECKPOINT),
            "configured_log_variance_bounds": [
                float(config["model"]["sensor_log_variance_min"]),
                float(config["model"]["sensor_log_variance_max"]),
            ],
            "existing_floor_collapse_resolved": True,
            "remaining_issue": (
                "test predicted variance is about 44.3 times latent MSE"
            ),
        },
        "fit_split": "val",
        "test_split": "test",
        "formula": {
            "scale": "mean((actual-predicted)^2 / predicted_variance)",
            "calibrated_variance": "scale * predicted_variance",
            "calibrated_log_variance": "log_variance + log(scale)",
            "objective": "Gaussian NLL",
            "closed_form_global_optimum": True,
        },
        "scale_bounds": [1e-6, 1e6],
        "runtime": {
            "batch_size": 16,
            "num_workers": 0,
            "device": "cuda",
            "workers_zero_reason": (
                "preserve the stable Windows object-model evaluation path"
            ),
        },
        "validation_acceptance": {
            "scale_not_clipped": True,
            "calibrated_nll_below_raw": True,
            "mean_squared_standardized_residual": 1.0,
            "variance_error_rank_preserved": True,
        },
        "test_evaluation": {
            "parameters_frozen_before_test_forward": True,
            "bootstrap_unit": "source_image_id",
            "bootstrap_repeats": 2000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 20260729,
            "test_previously_accessed_in_project": True,
            "classification_metrics_recomputed": False,
        },
        "outputs": {
            "parameters": PARAMETERS.relative_to(PROJECT_ROOT).as_posix(),
            "evaluation": (
                REPORT_ROOT / "EVALUATION.json"
            ).relative_to(PROJECT_ROOT).as_posix(),
        },
        "claim_boundaries": [
            "This is a development diagnostic, not a formal paper metric.",
            (
                "The scale is fit only on validation latents, but the test "
                "split has been accessed by earlier project experiments."
            ),
            (
                "Calibration applies only to image-conditioned latent "
                "variance reporting and is not physical sensor calibration."
            ),
            (
                "The calibrated variance is not fed into classification or "
                "the already-trained reliability gate."
            ),
        ],
        "bindings": {
            name: _source_binding(path) for name, path in paths.items()
        },
    }


def write_protocol(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    protocol = build_protocol()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8-sig"))
        old = dict(existing)
        new = dict(protocol)
        old.pop("frozen_at", None)
        new.pop("frozen_at", None)
        if old != new:
            raise FileExistsError(
                "refusing to replace drifted uncertainty protocol"
            )
        return existing
    _atomic_text(
        output,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# v3 潜空间方差温度校准协议",
        "",
        "- 状态：校准前冻结；非正式论文指标。",
        "- 拟合集：val；一次评估集：test。",
        "- 参数：单个全局正方差尺度，使用 Gaussian NLL 闭式解。",
        "- batch=16，workers=0，保持稳定的 Windows 对象模型路径。",
        "- 不修改检查点、分类输出或训练时可靠性门控。",
        "- 只解释为图像条件潜空间尺度校准，不解释为硬件标定。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return protocol


def main() -> None:
    protocol = write_protocol(parse_args().output)
    print(
        json.dumps(
            {
                "state": "complete",
                "valid": True,
                "protocol": protocol["protocol"],
                "status": protocol["status"],
                "fit_split": protocol["fit_split"],
                "test_split": protocol["test_split"],
                "binding_count": len(protocol["bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
