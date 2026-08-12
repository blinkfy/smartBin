"""Audit internal paper readiness separately from full patent reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from audit_image_recovery import (
    audit_image_recovery,
)
from audit_source_image_integrity import (
    audit_source_image_integrity,
)
from audit_bibliography import (
    DOMAIN_COVERAGE as EXPECTED_BIBLIOGRAPHY_COVERAGE,
    EXPECTED_KEYS as EXPECTED_BIBLIOGRAPHY_KEYS,
)
from audit_variant_lineage import (
    COMPARISON_ORDER as EXPECTED_VARIANT_COMPARISONS,
    VARIANT_FILES as EXPECTED_VARIANT_FILES,
    build_variant_lineage,
)
from analyze_error_patterns import (
    DEFAULT_GATE_SOURCES as EXPECTED_GATE_ERROR_ANALYSIS_SOURCES,
    DEFAULT_SOURCES as EXPECTED_ERROR_ANALYSIS_SOURCES,
    build_error_analysis,
)
from analyze_class_metric_intervals import (
    DEFAULT_SOURCES as EXPECTED_CLASS_INTERVAL_SOURCES,
    build_class_metric_intervals,
)
from compare_uncertainty_diagnostics import (
    DEFAULT_SOURCES as EXPECTED_UNCERTAINTY_COMPARISON_SOURCES,
    build_uncertainty_comparison,
)
from audit_manuscript_stable_sections import (
    audit_manuscript_stable_sections,
)
from audit_formal_queue_interruption_outcome import (
    DEFAULT_FILES as EXPECTED_FORMAL_INTERRUPTION_OUTCOME_FILES,
    build_interruption_outcome,
)


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_PAPER_TABLES = (
    "main_classification.csv",
    "formal_ablations.csv",
    "formal_ablation_statistics.csv",
    "yolo_segmentation.csv",
    "scene_pipeline.csv",
    "scene_strata.csv",
    "sensor_source_transfer.csv",
    "sensor_source_statistics.csv",
    "sensor_source_retraining.csv",
    "sensor_source_retraining_statistics.csv",
)
EXPECTED_PAPER_FIGURES = (
    "formal_ablation_effects.png",
    "yolo_segmentation_metrics.png",
    "scene_pipeline_comparison.png",
    "scene_strata_diagnostics.png",
    "sensor_source_comparison.png",
)
EXPECTED_PAPER_SOURCE_REPORTS = {
    "final_evidence_audit": "FINAL_EVIDENCE_AUDIT.json",
    "main_full": "test/evaluation.json",
    "main_full_quality_aware_prior": (
        "full_quality_aware_prior/test/evaluation.json"
    ),
    "main_full_cross_modal_quality": (
        "full_cross_modal_quality/test/evaluation.json"
    ),
    "main_full_cross_modal_quality_balanced": (
        "full_cross_modal_quality_balanced/test/evaluation.json"
    ),
    "formal_ablation_summary": "ablations/ablation_summary.json",
    "formal_ablation_statistics": (
        "ablations/ablation_pairwise_statistics.json"
    ),
    "v3_vs_v2b_paired": (
        "full_cross_modal_quality/PAIRED_COMPARISON.json"
    ),
    "v3_gate_paired": (
        "full_cross_modal_quality/GATE_PAIRED_COMPARISON.json"
    ),
    "yolo_segmentation": "yolo-seg/evaluation.json",
    "scene_pipeline": "yolo-seg/integrated/evaluation.json",
    "scene_strata": (
        "yolo-seg/integrated/stratified/evaluation.json"
    ),
    "sensor_source_transfer": "sensor-source-transfer/evaluation.json",
    "sensor_retraining_candidate": (
        "sensor-source-retraining/rule-v1/test/evaluation.json"
    ),
    "sensor_retraining_comparison": (
        "sensor-source-retraining/"
        "RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json"
    ),
}
EXPECTED_METHOD_SOURCES = (
    "audit_method_contract.py",
    "mmfc/model.py",
    "mmfc/losses.py",
    "mmfc/training.py",
    "configs/ablations/full_cross_modal_quality.json",
)
EXPECTED_UNCERTAINTY_SOURCES = {
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
EXPECTED_LATENT_UNCERTAINTY_SOURCES = {
    "config": "configs/ablations/full_cross_modal_quality.json",
    "checkpoint": (
        "checkpoints/ablations/full_cross_modal_quality/best.pt"
    ),
    "manifest": "data/qwen-material/test.jsonl",
    "sensor_manifest": "data/qwen-material/sensors.jsonl",
}
EXPECTED_V2B_LATENT_UNCERTAINTY_SOURCES = {
    "config": "configs/ablations/full_quality_aware_prior.json",
    "checkpoint": (
        "checkpoints/ablations/full_quality_aware_prior/best.pt"
    ),
    "manifest": "data/qwen-material/test.jsonl",
    "sensor_manifest": "data/qwen-material/sensors.jsonl",
}
EXPECTED_LATENT_BOOTSTRAP_REPEATS = 2000
EXPECTED_LATENT_BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_LATENT_BOOTSTRAP_SEED = 20260727
FOUNDATIONAL_REQUIREMENTS = {
    "dataset_governance": (
        "DATASET_CARD.md",
        "DATA_LEAKAGE_AUDIT.json",
        "EVALUATION_GOVERNANCE.md",
        "IMAGE_READ_RECOVERY_AUDIT.json",
        "IMAGE_READ_RECOVERY_AUDIT.md",
        "IMAGE_READ_RECOVERY_INCIDENT.json",
        "IMAGE_READ_RECOVERY_INCIDENT.md",
        "IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.json",
        "IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.md",
        "IMAGE_READ_RECOVERY_OUTCOME.json",
        "IMAGE_READ_RECOVERY_OUTCOME.md",
        "SOURCE_IMAGE_INTEGRITY_AUDIT.json",
        "SOURCE_IMAGE_INTEGRITY_AUDIT.md",
        "SOURCE_IMAGE_INTEGRITY_CURRENT.json",
        "SOURCE_IMAGE_INTEGRITY_CURRENT.md",
        "SOURCE_IMAGE_REPAIR_TRANSITION.json",
        "SOURCE_IMAGE_REPAIR_TRANSITION.md",
        "FORMAL_QUEUE_INTERRUPTION.json",
        "FORMAL_QUEUE_INTERRUPTION.md",
        "FORMAL_QUEUE_INTERRUPTION_ADDENDUM.json",
        "FORMAL_QUEUE_INTERRUPTION_ADDENDUM.md",
        "FORMAL_QUEUE_INTERRUPTION_OUTCOME.json",
        "FORMAL_QUEUE_INTERRUPTION_OUTCOME.md",
    ),
    "method_definition": (
        "METHOD_SPEC.md",
        "METHOD_IMPLEMENTATION_TRACE.md",
        "METHOD_CONTRACT_AUDIT.json",
        "method_architecture.png",
        "method_architecture.pdf",
        "PATENT_ALIGNMENT.md",
        "UNCERTAINTY_FLOOR_AUDIT.json",
        "UNCERTAINTY_FLOOR_AUDIT.md",
        "VARIANT_LINEAGE_AUDIT.json",
        "VARIANT_LINEAGE_AUDIT.md",
    ),
    "statistical_governance": (
        "STATISTICAL_PROTOCOL.md",
        "ENVIRONMENT.md",
        "REPRODUCIBILITY.md",
        "YOLO_SEG_DEPENDENCY_INCIDENT.json",
        "YOLO_SEG_DEPENDENCY_INCIDENT.md",
        "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.json",
        "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.md",
        "YOLO_SEG_DEPENDENCY_OUTCOME.json",
        "YOLO_SEG_DEPENDENCY_OUTCOME.md",
        "yolo-seg/INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json",
        "yolo-seg/INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.md",
        "yolo-seg/STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json",
        "yolo-seg/STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.md",
        "yolo-seg/SCENE_STRATA_REPORTING_INCIDENT.json",
        "yolo-seg/SCENE_STRATA_REPORTING_INCIDENT.md",
        "yolo-seg/SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json",
        "yolo-seg/SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.md",
        "yolo-seg/SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json",
        "yolo-seg/SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.md",
        "SENSOR_RETRAINING_EXIT_INCIDENT.json",
        "SENSOR_RETRAINING_EXIT_INCIDENT.md",
        "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json",
        "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.md",
        "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.json",
        "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.md",
        "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json",
        "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.md",
        "SENSOR_RETRAINING_EXIT_OUTCOME.json",
        "SENSOR_RETRAINING_EXIT_OUTCOME.md",
    ),
    "claim_governance": (
        "PAPER_CLAIM_REGISTER.md",
        "MANUSCRIPT_DECISION_BRIEF.md",
        "MANUSCRIPT_STABLE_SECTIONS.md",
        "MANUSCRIPT_STABLE_SECTIONS_AUDIT.json",
        "MANUSCRIPT_STABLE_SECTIONS_AUDIT.md",
        "THREATS_TO_VALIDITY.md",
        "ARTIFACT_RELEASE_CHECKLIST.md",
        "reference_candidates.bib",
        "BIBLIOGRAPHY_AUDIT.json",
        "BIBLIOGRAPHY_AUDIT.md",
    ),
    "object_level_evidence": (
        "full_cross_modal_quality/CHECKPOINT_AUDIT.json",
        "full_cross_modal_quality/test/evaluation.json",
        "full_cross_modal_quality/GATE_PAIRED_COMPARISON.json",
        "full_cross_modal_quality/ROBUSTNESS_COMPARISON.json",
        (
            "full_cross_modal_quality/uncertainty/"
            "uncertainty_diagnostics.json"
        ),
        (
            "full_cross_modal_quality/uncertainty/"
            "uncertainty_diagnostics.md"
        ),
        (
            "full_cross_modal_quality/uncertainty/"
            "uncertainty_diagnostics.png"
        ),
        (
            "full_quality_aware_prior/uncertainty/"
            "uncertainty_diagnostics.json"
        ),
        (
            "full_quality_aware_prior/uncertainty/"
            "uncertainty_diagnostics.md"
        ),
        (
            "full_quality_aware_prior/uncertainty/"
            "uncertainty_diagnostics.png"
        ),
        "UNCERTAINTY_DIAGNOSTICS_COMPARISON.json",
        "UNCERTAINTY_DIAGNOSTICS_COMPARISON.md",
        "ERROR_ANALYSIS.json",
        "ERROR_ANALYSIS.md",
        "GATE_ERROR_ANALYSIS.json",
        "GATE_ERROR_ANALYSIS.md",
        "CLASS_METRIC_INTERVALS.json",
        "CLASS_METRIC_INTERVALS.md",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit paper-ready internal evidence without conflating it with "
            "full patent or deployment reproduction"
        )
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--stable-manuscript-audit",
        type=Path,
        default=None,
        help=(
            "Use an alternate hash-bound stable-manuscript audit without "
            "overwriting the immutable base audit"
        ),
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow unfinished experiment queues while auditing static evidence",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_evidence_audit_path(
    reports_root: Path,
    *,
    allow_pending: bool,
) -> Path:
    final_path = reports_root / "FINAL_EVIDENCE_AUDIT.json"
    pending_path = reports_root / "FINAL_EVIDENCE_AUDIT.pending.json"
    if final_path.is_file():
        if not allow_pending:
            return final_path
        try:
            final = _read_json(final_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            final = {}
        if (
            final.get("state") == "complete"
            and final.get("valid") is True
        ):
            return final_path
    if allow_pending and pending_path.is_file():
        return pending_path
    return final_path


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_requirement(
    reports_root: Path,
    paths: Sequence[str],
) -> dict[str, Any]:
    artifacts = []
    missing = []
    for relative in paths:
        path = reports_root / relative
        exists = path.is_file()
        artifacts.append(
            {
                "path": path.as_posix(),
                "exists": exists,
                "sha256": _sha256(path) if exists else None,
            }
        )
        if not exists:
            missing.append(path.as_posix())
    return {
        "status": "complete" if not missing else "missing",
        "artifacts": artifacts,
        "missing": missing,
    }


def audit_method_contract_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"method contract audit is missing: {report_path}"],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                f"method contract audit cannot be parsed: {type(exc).__name__}"
            ],
        }
    if report.get("state") != "complete" or report.get("valid") is not True:
        errors.append("method contract audit is not complete and valid")
    if report.get("config") != EXPECTED_METHOD_SOURCES[-1]:
        errors.append("method contract config path mismatch")
    recorded_errors = report.get("errors")
    if recorded_errors != []:
        errors.append("method contract audit records errors")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("method contract checks are missing")
    elif any(item.get("passed") is not True for item in checks):
        errors.append("method contract contains a failed check")

    tensor_contract = report.get("tensor_contract", {})
    expected_tensor_values = {
        "logits_shape": [2, 19],
        "raw_grid_quality_shape": [2, 7],
        "cross_modal_grid_quality_shape": [2, 22],
        "predicted_grid_shape": [2, 1, 4, 4],
        "sensor_grid_quality_dim": 7,
        "sensor_cross_modal_quality_dim": 22,
        "film_gamma_shape": [3, 512],
        "film_beta_shape": [3, 512],
        "film_affine_parameter_count": 3072,
        "scale_prior_is_trainable": True,
        "scale_prior_initial": [0.3, 0.5, 0.2],
        "modality_weight_row_sums": [1.0, 1.0],
        "missing_reliability": 0.0,
    }
    for key, expected in expected_tensor_values.items():
        if tensor_contract.get(key) != expected:
            errors.append(f"method tensor contract mismatch: {key}")
    for key in (
        "effective_sensor_formula_max_abs_error",
        "visual_query_context_max_abs_error",
        "fusion_query_weights_max_abs_error",
        "modality_weight_formula_max_abs_error",
        "missing_fallback_max_abs_error",
    ):
        value = tensor_contract.get(key)
        if not isinstance(value, (int, float)) or abs(float(value)) > 1e-6:
            errors.append(f"method tensor contract error is too large: {key}")

    sources = report.get("sources")
    if not isinstance(sources, list):
        sources = []
        errors.append("method contract source list is missing")
    recorded_paths = {
        item.get("path")
        for item in sources
        if isinstance(item, Mapping)
    }
    if recorded_paths != set(EXPECTED_METHOD_SOURCES):
        errors.append("method contract source set mismatch")
    fingerprint_payload = json.dumps(
        sources,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if report.get("source_fingerprint_sha256") != hashlib.sha256(
        fingerprint_payload
    ).hexdigest():
        errors.append("method contract source fingerprint mismatch")

    root = project_root.resolve()
    for item in sources:
        if not isinstance(item, Mapping):
            errors.append("method contract source entry is malformed")
            continue
        relative = item.get("path")
        if relative not in EXPECTED_METHOD_SOURCES:
            continue
        source_path = (root / str(relative)).resolve()
        try:
            source_path.relative_to(root)
        except ValueError:
            errors.append(f"method contract source escapes project: {relative}")
            continue
        if not source_path.is_file():
            errors.append(f"method contract source is missing: {relative}")
            continue
        if item.get("size_bytes") != source_path.stat().st_size:
            errors.append(f"method contract source size mismatch: {relative}")
        if item.get("sha256") != _sha256(source_path):
            errors.append(f"method contract source hash mismatch: {relative}")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "source_count": len(sources),
        "errors": errors,
    }


def _training_config_fingerprint(config: Mapping[str, Any]) -> str:
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


def _sequence_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_uncertainty_floor_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"uncertainty floor audit is missing: {report_path}"],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                f"uncertainty floor audit cannot be parsed: {type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["uncertainty floor audit must be a JSON object"],
        }
    if report.get("state") != "complete" or report.get("valid") is not True:
        errors.append("uncertainty floor audit is not complete and valid")
    if report.get("schema_version") != 1:
        errors.append("uncertainty floor audit schema version mismatch")
    if report.get("near_floor_margin") != 0.05:
        errors.append("uncertainty floor audit margin mismatch")
    variants = report.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(
        EXPECTED_UNCERTAINTY_SOURCES
    ):
        errors.append("uncertainty floor variant set mismatch")
        variants = {}

    project_root = project_root.resolve()
    checked_sources = 0
    for name, expected_sources in EXPECTED_UNCERTAINTY_SOURCES.items():
        row = variants.get(name)
        if not isinstance(row, Mapping):
            errors.append(f"uncertainty floor variant is missing: {name}")
            continue
        sources = row.get("sources")
        if not isinstance(sources, Mapping) or set(sources) != {
            "config",
            "history",
        }:
            errors.append(f"uncertainty floor source set mismatch: {name}")
            continue
        resolved: dict[str, Path] = {}
        for source_name, relative in expected_sources.items():
            path = (project_root / relative).resolve()
            resolved[source_name] = path
            entry = sources.get(source_name)
            if not isinstance(entry, Mapping):
                errors.append(
                    f"uncertainty floor source is missing: {name}.{source_name}"
                )
                continue
            if entry.get("path") != relative:
                errors.append(
                    f"uncertainty floor source path mismatch: "
                    f"{name}.{source_name}"
                )
            if not path.is_file():
                errors.append(
                    f"uncertainty floor source file is missing: {path}"
                )
                continue
            if entry.get("size_bytes") != path.stat().st_size:
                errors.append(
                    f"uncertainty floor source size mismatch: "
                    f"{name}.{source_name}"
                )
            if entry.get("sha256") != _sha256(path):
                errors.append(
                    f"uncertainty floor source hash mismatch: "
                    f"{name}.{source_name}"
                )
            checked_sources += 1
        if not all(path.is_file() for path in resolved.values()):
            continue
        try:
            config = _read_json(resolved["config"])
            history_document = _read_json(resolved["history"])
            history = history_document["history"]
            last = history[-1]
            fingerprint = _training_config_fingerprint(config)
            lower = float(config["model"]["sensor_log_variance_min"])
            target = float(config["loss"]["variance_prior_target"])
            reconstruction = float(last["val_sensor_reconstruction"])
            prior_mse = float(last["val_sensor_variance_prior"])
            theoretical_floor = 0.5 * lower
            floor_gap = reconstruction - theoretical_floor
            fraction_bound = min(
                1.0,
                max(0.0, prior_mse / ((lower - target) ** 2)),
            )
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(
                f"uncertainty floor source calculation failed: {name}: {exc}"
            )
            continue
        if history_document.get("config_fingerprint") != fingerprint:
            errors.append(
                f"uncertainty floor history fingerprint mismatch: {name}"
            )
        if row.get("source_config_fingerprint") != fingerprint:
            errors.append(
                f"uncertainty floor report fingerprint mismatch: {name}"
            )
        expected_numbers = {
            "epoch_count": len(history),
            "final_epoch": int(last["epoch"]),
            "val_sensor_reconstruction": reconstruction,
            "theoretical_nll_floor": theoretical_floor,
            "nll_floor_gap": floor_gap,
            "val_sensor_variance_prior_mse": prior_mse,
            "exact_lower_clamp_fraction_upper_bound": fraction_bound,
        }
        for key, expected in expected_numbers.items():
            actual = row.get(key)
            if isinstance(expected, float):
                try:
                    matches = math.isclose(
                        float(actual),
                        expected,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                errors.append(
                    f"uncertainty floor metric mismatch: {name}.{key}"
                )
        if row.get("near_theoretical_floor") is not (
            floor_gap <= float(report["near_floor_margin"])
        ):
            errors.append(
                f"uncertainty floor flag mismatch: {name}"
            )
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "variant_count": len(variants),
        "source_count": checked_sources,
        "errors": errors,
    }


def _is_close(actual: Any, expected: float) -> bool:
    try:
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        return False


def audit_latent_uncertainty_bundle(
    report_path: Path,
    *,
    project_root: Path,
    expected_sources: Mapping[str, str] = (
        EXPECTED_LATENT_UNCERTAINTY_SOURCES
    ),
    expected_figure: str = (
        "reports/qwen-material/full_cross_modal_quality/uncertainty/"
        "uncertainty_diagnostics.png"
    ),
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"latent uncertainty report is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "latent uncertainty report cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["latent uncertainty report must be a JSON object"],
        }
    if report.get("schema_version") != 1 or report.get("valid") is not True:
        errors.append("latent uncertainty report is not schema-valid")
    if report.get("diagnostic_scope") != "sensor_encoder_latent_space":
        errors.append("latent uncertainty diagnostic scope mismatch")
    claim_boundary = report.get("claim_boundary")
    if (
        not isinstance(claim_boundary, str)
        or "not physical sensor calibration" not in claim_boundary
        or "external validation" not in claim_boundary
    ):
        errors.append("latent uncertainty claim boundary is incomplete")

    root = project_root.resolve()
    bindings = report.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        expected_sources
    ):
        errors.append("latent uncertainty source binding set mismatch")
        bindings = {}
    checked_sources = 0
    resolved_sources: dict[str, Path] = {}
    for name, relative in expected_sources.items():
        source_path = (root / relative).resolve()
        resolved_sources[name] = source_path
        try:
            source_path.relative_to(root)
        except ValueError:
            errors.append(
                f"latent uncertainty source escapes project: {name}"
            )
            continue
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            errors.append(f"latent uncertainty source is missing: {name}")
            continue
        if binding.get("path") != relative:
            errors.append(
                f"latent uncertainty source path mismatch: {name}"
            )
        if not source_path.is_file():
            errors.append(
                f"latent uncertainty source file is missing: {source_path}"
            )
            continue
        if binding.get("size_bytes") != source_path.stat().st_size:
            errors.append(
                f"latent uncertainty source size mismatch: {name}"
            )
        if binding.get("sha256") != _sha256(source_path):
            errors.append(
                f"latent uncertainty source hash mismatch: {name}"
            )
        checked_sources += 1

    config: Mapping[str, Any] = {}
    manifest_count: int | None = None
    manifest_groups: list[str] | None = None
    if resolved_sources.get("config", Path()).is_file():
        try:
            loaded_config = _read_json(resolved_sources["config"])
            if isinstance(loaded_config, Mapping):
                config = loaded_config
            else:
                errors.append("latent uncertainty config must be an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                "latent uncertainty config cannot be parsed: "
                f"{type(exc).__name__}"
            )
    if resolved_sources.get("manifest", Path()).is_file():
        try:
            records = [
                json.loads(line)
                for line in resolved_sources["manifest"].read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ]
            if not all(isinstance(item, Mapping) for item in records):
                raise ValueError("manifest row is not an object")
            manifest_groups = [
                str(
                    item.get(
                        "source_image_id",
                        item.get(
                            "sample_id",
                            item.get("id", ""),
                        ),
                    )
                )
                for item in records
            ]
            if any(not value for value in manifest_groups):
                raise ValueError("manifest row lacks a stable group id")
            manifest_count = len(records)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            errors.append(
                "latent uncertainty manifest cannot be counted: "
                f"{type(exc).__name__}"
            )

    checkpoint = report.get("checkpoint")
    dataset = report.get("dataset")
    diagnostics = report.get("diagnostics")
    if not isinstance(checkpoint, Mapping):
        errors.append("latent uncertainty checkpoint metadata is malformed")
        checkpoint = {}
    if not isinstance(dataset, Mapping):
        errors.append("latent uncertainty dataset metadata is malformed")
        dataset = {}
    if not isinstance(diagnostics, Mapping):
        errors.append("latent uncertainty diagnostics are malformed")
        diagnostics = {}
    if config:
        fingerprint = _training_config_fingerprint(config)
        if checkpoint.get("config_fingerprint") != fingerprint:
            errors.append(
                "latent uncertainty config fingerprint mismatch"
            )
        try:
            expected_lower = float(
                config["model"]["sensor_log_variance_min"]
            )
            expected_upper = float(
                config["model"]["sensor_log_variance_max"]
            )
        except (KeyError, TypeError, ValueError):
            errors.append("latent uncertainty variance bounds are unavailable")
        else:
            bounds = diagnostics.get("configured_bounds", {})
            if (
                not isinstance(bounds, Mapping)
                or not _is_close(
                    bounds.get("log_variance_min"),
                    expected_lower,
                )
                or not _is_close(
                    bounds.get("log_variance_max"),
                    expected_upper,
                )
            ):
                errors.append(
                    "latent uncertainty configured bounds mismatch"
                )
    if (
        dataset.get("split") != "test"
        or dataset.get("max_samples") is not None
    ):
        errors.append("latent uncertainty report is not a full test run")
    evaluated_samples = dataset.get("evaluated_samples")
    if (
        not isinstance(evaluated_samples, int)
        or evaluated_samples <= 0
        or dataset.get("full_split_samples") != evaluated_samples
        or (
            manifest_count is not None
            and evaluated_samples != manifest_count
        )
    ):
        errors.append("latent uncertainty sample count mismatch")
    latent_dimension = diagnostics.get("latent_dimension")
    if (
        diagnostics.get("sample_count") != evaluated_samples
        or not isinstance(latent_dimension, int)
        or latent_dimension <= 0
        or diagnostics.get("element_count")
        != evaluated_samples * latent_dimension
    ):
        errors.append("latent uncertainty tensor dimensions mismatch")

    calibration = diagnostics.get("variance_calibration", {})
    derived = diagnostics.get("derived_interpretation_values", {})
    log_variance = diagnostics.get("log_variance", {})
    standardized = diagnostics.get("standardized_residual", {})
    bounds = diagnostics.get("configured_bounds", {})
    if not all(
        isinstance(value, Mapping)
        for value in (
            calibration,
            derived,
            log_variance,
            standardized,
            bounds,
        )
    ):
        errors.append("latent uncertainty metric groups are malformed")
    else:
        try:
            mean_variance = float(
                calibration["global_mean_predicted_variance"]
            )
            mean_error = float(
                calibration["global_mean_squared_error"]
            )
            ratio = mean_error / mean_variance
            inverse_ratio = mean_variance / mean_error
            lower_distance = float(log_variance["minimum"]) - float(
                bounds["log_variance_min"]
            )
            coverage_one = float(
                standardized["coverage_absolute_le_1"]
            )
            coverage_one_ninety_six = float(
                standardized["coverage_absolute_le_1_96"]
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            errors.append(
                "latent uncertainty core metrics cannot be recomputed"
            )
        else:
            expected_values = {
                "global_mse_to_variance_ratio": (
                    calibration.get("global_mse_to_variance_ratio"),
                    ratio,
                ),
                "predicted_variance_to_mse_ratio": (
                    derived.get("predicted_variance_to_mse_ratio"),
                    inverse_ratio,
                ),
                "minimum_log_variance_distance_above_lower_bound": (
                    derived.get(
                        "minimum_log_variance_distance_above_lower_bound"
                    ),
                    lower_distance,
                ),
                "coverage_gap_absolute_le_1_vs_standard_normal": (
                    derived.get(
                        "coverage_gap_absolute_le_1_vs_standard_normal"
                    ),
                    coverage_one - 0.6826894921370859,
                ),
                "coverage_gap_absolute_le_1_96_vs_standard_normal": (
                    derived.get(
                        "coverage_gap_absolute_le_1_96_vs_standard_normal"
                    ),
                    coverage_one_ninety_six - 0.950004209703559,
                ),
            }
            for name, (actual, expected) in expected_values.items():
                if not _is_close(actual, expected):
                    errors.append(
                        f"latent uncertainty metric mismatch: {name}"
                    )

    bootstrap = report.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        errors.append("latent uncertainty bootstrap is malformed")
    else:
        bootstrap_identity = {
            "method": "source_image_cluster_percentile_bootstrap",
            "repeats": EXPECTED_LATENT_BOOTSTRAP_REPEATS,
            "confidence": EXPECTED_LATENT_BOOTSTRAP_CONFIDENCE,
            "seed": EXPECTED_LATENT_BOOTSTRAP_SEED,
            "sample_count": evaluated_samples,
            "cluster_count": (
                len(set(manifest_groups))
                if manifest_groups is not None
                else None
            ),
            "source_group_assignment_sha256": (
                _sequence_sha256(manifest_groups)
                if manifest_groups is not None
                else None
            ),
        }
        for name, expected in bootstrap_identity.items():
            actual = bootstrap.get(name)
            matches = (
                _is_close(actual, expected)
                if isinstance(expected, float)
                else actual == expected
            )
            if not matches:
                errors.append(
                    f"latent uncertainty bootstrap mismatch: {name}"
                )
        bootstrap_metrics = bootstrap.get("metrics")
        expected_bootstrap_metrics = {
            "sample_mean_spearman": (
                diagnostics.get("variance_error_correlation", {})
                .get("sample_mean", {})
                .get("spearman")
            ),
            "predicted_variance_to_mse_ratio": derived.get(
                "predicted_variance_to_mse_ratio"
            ),
        }
        if (
            not isinstance(bootstrap_metrics, Mapping)
            or set(bootstrap_metrics) != set(expected_bootstrap_metrics)
        ):
            errors.append(
                "latent uncertainty bootstrap metric set mismatch"
            )
        else:
            alpha = 1.0 - EXPECTED_LATENT_BOOTSTRAP_CONFIDENCE
            for name, expected_estimate in (
                expected_bootstrap_metrics.items()
            ):
                metric = bootstrap_metrics.get(name)
                if not isinstance(metric, Mapping):
                    errors.append(
                        f"latent uncertainty bootstrap metric malformed: {name}"
                    )
                    continue
                try:
                    replicates = [
                        float(value)
                        for value in metric["replicates"]
                    ]
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"latent uncertainty bootstrap replicates malformed: {name}"
                    )
                    continue
                if (
                    len(replicates) != EXPECTED_LATENT_BOOTSTRAP_REPEATS
                    or not all(math.isfinite(value) for value in replicates)
                ):
                    errors.append(
                        f"latent uncertainty bootstrap replicates invalid: {name}"
                    )
                    continue
                ordered = sorted(replicates)
                try:
                    import numpy as np

                    lower = float(np.quantile(ordered, alpha / 2.0))
                    upper = float(
                        np.quantile(ordered, 1.0 - alpha / 2.0)
                    )
                except (TypeError, ValueError):
                    errors.append(
                        f"latent uncertainty bootstrap interval failed: {name}"
                    )
                    continue
                expected_metric_values = {
                    "estimate": expected_estimate,
                    "lower": lower,
                    "upper": upper,
                    "probability_greater_than_zero": (
                        sum(value > 0.0 for value in replicates)
                        / len(replicates)
                    ),
                }
                for field, expected in expected_metric_values.items():
                    if field == "estimate":
                        try:
                            matches = math.isclose(
                                float(metric.get(field)),
                                float(expected),
                                rel_tol=1e-6,
                                abs_tol=1e-8,
                            )
                        except (TypeError, ValueError):
                            matches = False
                    else:
                        matches = _is_close(metric.get(field), expected)
                    if not matches:
                        errors.append(
                            "latent uncertainty bootstrap metric mismatch: "
                            f"{name}.{field}"
                        )

    figure = report.get("figure")
    figure_path = (root / expected_figure).resolve()
    if not isinstance(figure, Mapping):
        errors.append("latent uncertainty figure metadata is malformed")
    elif figure.get("path") != expected_figure:
        errors.append("latent uncertainty figure path mismatch")
    elif not figure_path.is_file():
        errors.append("latent uncertainty figure is missing")
    else:
        if figure.get("size_bytes") != figure_path.stat().st_size:
            errors.append("latent uncertainty figure size mismatch")
        if figure.get("sha256") != _sha256(figure_path):
            errors.append("latent uncertainty figure hash mismatch")
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("latent uncertainty Markdown report is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "source_count": checked_sources,
        "errors": errors,
    }


def audit_uncertainty_comparison_bundle(
    report_path: Path,
    *,
    reports_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"uncertainty comparison is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "uncertainty comparison cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["uncertainty comparison must be a JSON object"],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("uncertainty comparison is not complete and valid")
    bindings = report.get("source_reports")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_UNCERTAINTY_COMPARISON_SOURCES
    ):
        errors.append("uncertainty comparison source set mismatch")
        bindings = {}
    root = reports_root.resolve()
    source_reports: dict[str, Mapping[str, Any]] = {}
    for name, relative in EXPECTED_UNCERTAINTY_COMPARISON_SOURCES.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(
                f"uncertainty comparison source escapes reports: {name}"
            )
            continue
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            errors.append(
                f"uncertainty comparison source is missing: {name}"
            )
            continue
        if binding.get("path") != relative:
            errors.append(
                f"uncertainty comparison source path mismatch: {name}"
            )
        if not path.is_file():
            errors.append(
                f"uncertainty comparison source file is missing: {path}"
            )
            continue
        if binding.get("size_bytes") != path.stat().st_size:
            errors.append(
                f"uncertainty comparison source size mismatch: {name}"
            )
        if binding.get("sha256") != _sha256(path):
            errors.append(
                f"uncertainty comparison source hash mismatch: {name}"
            )
        try:
            source = _read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                f"uncertainty comparison source cannot be parsed: "
                f"{name}: {type(exc).__name__}"
            )
            continue
        if not isinstance(source, Mapping):
            errors.append(
                f"uncertainty comparison source is malformed: {name}"
            )
            continue
        source_reports[name] = source
    if set(source_reports) == set(
        EXPECTED_UNCERTAINTY_COMPARISON_SOURCES
    ):
        try:
            expected = build_uncertainty_comparison(
                source_reports["v2b"],
                source_reports["v3"],
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(
                f"uncertainty comparison cannot be recomputed: {exc}"
            )
        else:
            if report.get("comparison") != expected:
                errors.append(
                    "uncertainty comparison metrics do not match sources"
                )
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("uncertainty comparison Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "source_count": len(source_reports),
        "errors": errors,
    }


def audit_manuscript_stable_sections_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"stable manuscript audit is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "stable manuscript audit cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["stable manuscript audit must be a JSON object"],
        }
    reports_root = project_root / "reports" / "qwen-material"
    try:
        expected = audit_manuscript_stable_sections(
            sources={
                "manuscript": reports_root / "MANUSCRIPT_STABLE_SECTIONS.md",
                "bibliography": reports_root / "reference_candidates.bib",
                "method_spec": reports_root / "METHOD_SPEC.md",
                "dataset_card": reports_root / "DATASET_CARD.md",
                "statistical_protocol": reports_root / "STATISTICAL_PROTOCOL.md",
                "patent_alignment": reports_root / "PATENT_ALIGNMENT.md",
                "release_checklist": (
                    reports_root / "ARTIFACT_RELEASE_CHECKLIST.md"
                ),
                "image_recovery_outcome": (
                    reports_root / "IMAGE_READ_RECOVERY_OUTCOME.json"
                ),
                "source_repair_transition": (
                    reports_root / "SOURCE_IMAGE_REPAIR_TRANSITION.json"
                ),
                "formal_interruption_addendum": (
                    reports_root
                    / "FORMAL_QUEUE_INTERRUPTION_ADDENDUM.json"
                ),
                "formal_interruption_outcome": (
                    reports_root
                    / "FORMAL_QUEUE_INTERRUPTION_OUTCOME.json"
                ),
                "yolo_dependency_incident": (
                    reports_root / "YOLO_SEG_DEPENDENCY_INCIDENT.json"
                ),
                "yolo_dependency_incident_addendum": (
                    reports_root
                    / "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.json"
                ),
                "yolo_dependency_outcome": (
                    reports_root / "YOLO_SEG_DEPENDENCY_OUTCOME.json"
                ),
                "integrated_scene_protocol_addendum": (
                    reports_root
                    / "yolo-seg"
                    / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
                ),
                "stratified_scene_protocol_addendum": (
                    reports_root
                    / "yolo-seg"
                    / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
                ),
                "scene_strata_reporting_incident": (
                    reports_root
                    / "yolo-seg"
                    / "SCENE_STRATA_REPORTING_INCIDENT.json"
                ),
                "scene_strata_reporting_addendum": (
                    reports_root
                    / "yolo-seg"
                    / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
                ),
                "scene_strata_reporting_outcome": (
                    reports_root
                    / "yolo-seg"
                    / "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json"
                ),
                "sensor_retraining_exit_incident": (
                    reports_root / "SENSOR_RETRAINING_EXIT_INCIDENT.json"
                ),
                "sensor_retraining_exit_addendum": (
                    reports_root
                    / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json"
                ),
                "sensor_retraining_exit_encoding_addendum": (
                    reports_root
                    / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.json"
                ),
                "sensor_retraining_log_encoding_repair": (
                    reports_root
                    / "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json"
                ),
                "sensor_retraining_exit_outcome": (
                    reports_root / "SENSOR_RETRAINING_EXIT_OUTCOME.json"
                ),
                "v3_config": (
                    project_root
                    / "configs"
                    / "ablations"
                    / "full_cross_modal_quality.json"
                ),
                "yolo_run_manifest": (
                    reports_root / "yolo-seg" / "RUN_MANIFEST.json"
                ),
            }
        )
    except (
        OSError,
        UnicodeDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            "stable manuscript audit cannot be recomputed: "
            f"{type(exc).__name__}: {exc}"
        )
        expected = None
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("stable manuscript audit is not complete and valid")
    if expected is not None:
        for key in (
            "state",
            "valid",
            "scope",
            "summary",
            "citations",
            "missing_citations",
            "missing_headings",
            "present_forbidden_headings",
            "missing_boundaries",
            "missing_method_contracts",
            "percentage_literals",
            "source_bindings",
            "errors",
            "boundaries",
        ):
            if report.get(key) != expected.get(key):
                errors.append(f"stable manuscript audit drift: {key}")
    if not report_path.with_suffix(".md").is_file():
        errors.append("stable manuscript audit Markdown is missing")
    summary = report.get("summary", {})
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "citation_count": (
            summary.get("citation_count")
            if isinstance(summary, Mapping)
            else None
        ),
        "required_heading_count": (
            summary.get("required_heading_count")
            if isinstance(summary, Mapping)
            else None
        ),
        "required_boundary_count": (
            summary.get("required_boundary_count")
            if isinstance(summary, Mapping)
            else None
        ),
        "required_method_contract_count": (
            summary.get("required_method_contract_count")
            if isinstance(summary, Mapping)
            else None
        ),
        "source_binding_count": (
            len(report.get("source_bindings", {}))
            if isinstance(report.get("source_bindings"), Mapping)
            else 0
        ),
        "errors": errors,
    }


def audit_bibliography_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"bibliography audit is missing: {report_path}"],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                f"bibliography audit cannot be parsed: {type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["bibliography audit must be a JSON object"],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
        or report.get("error_count") != 0
    ):
        errors.append("bibliography audit is not complete and valid")
    if (
        report.get("raw_entry_count") != len(EXPECTED_BIBLIOGRAPHY_KEYS)
        or report.get("parsed_entry_count")
        != len(EXPECTED_BIBLIOGRAPHY_KEYS)
        or set(report.get("citation_keys", []))
        != set(EXPECTED_BIBLIOGRAPHY_KEYS)
    ):
        errors.append("bibliography audit citation key set mismatch")
    coverage = report.get("domain_coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != set(
        EXPECTED_BIBLIOGRAPHY_COVERAGE
    ):
        errors.append("bibliography audit coverage set mismatch")
    else:
        for name, expected_keys in EXPECTED_BIBLIOGRAPHY_COVERAGE.items():
            row = coverage.get(name)
            if (
                not isinstance(row, Mapping)
                or row.get("keys") != list(expected_keys)
                or row.get("complete") is not True
            ):
                errors.append(
                    f"bibliography audit coverage mismatch: {name}"
                )
    source = report.get("source")
    expected_relative = (
        "reports/qwen-material/reference_candidates.bib"
    )
    source_path = (project_root.resolve() / expected_relative).resolve()
    if not isinstance(source, Mapping):
        errors.append("bibliography audit source binding is malformed")
    elif source.get("path") != expected_relative:
        errors.append("bibliography audit source path mismatch")
    elif not source_path.is_file():
        errors.append("bibliography source file is missing")
    else:
        if source.get("size_bytes") != source_path.stat().st_size:
            errors.append("bibliography source size mismatch")
        if source.get("sha256") != _sha256(source_path):
            errors.append("bibliography source hash mismatch")
    parser = report.get("parser")
    if (
        not isinstance(parser, Mapping)
        or parser.get("name") != "pandoc"
        or parser.get("input_format") != "biblatex"
        or parser.get("output_format") != "csljson"
    ):
        errors.append("bibliography parser metadata mismatch")
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("bibliography audit Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "entry_count": report.get("parsed_entry_count"),
        "errors": errors,
    }


def audit_variant_lineage_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"variant lineage audit is missing: {report_path}"],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "variant lineage audit cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["variant lineage audit must be a JSON object"],
        }
    expected = build_variant_lineage(
        project_root / "configs" / "ablations"
    )
    if not expected["valid"]:
        errors.extend(
            f"current variant config lineage is invalid: {error}"
            for error in expected["errors"]
        )
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
        or report.get("error_count") != 0
    ):
        errors.append("variant lineage audit is not complete and valid")
    for key in (
        "lineage",
        "variants",
        "comparisons",
        "shared_invariants",
    ):
        if report.get(key) != expected.get(key):
            errors.append(f"variant lineage audit drift: {key}")
    if report.get("lineage") != list(EXPECTED_VARIANT_FILES):
        errors.append("variant lineage alias order mismatch")
    comparisons = report.get("comparisons")
    if isinstance(comparisons, list):
        observed_pairs = [
            (row.get("reference"), row.get("candidate"))
            for row in comparisons
            if isinstance(row, Mapping)
        ]
        if observed_pairs != list(EXPECTED_VARIANT_COMPARISONS):
            errors.append("variant lineage comparison order mismatch")
    else:
        errors.append("variant lineage comparisons are malformed")
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("variant lineage audit Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "variant_count": len(report.get("variants", {})),
        "comparison_count": len(report.get("comparisons", [])),
        "errors": errors,
    }


def audit_error_analysis_bundle(
    report_path: Path,
    *,
    project_root: Path,
    expected_sources: Mapping[str, str] = EXPECTED_ERROR_ANALYSIS_SOURCES,
    reference_name: str = "v2b",
    candidate_name: str = "v3",
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"error analysis is missing: {report_path}"],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                f"error analysis cannot be parsed: {type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["error analysis must be a JSON object"],
        }
    try:
        expected = build_error_analysis(
            config_path=project_root
            / expected_sources["config"],
            manifest_path=project_root
            / expected_sources["manifest"],
            reference_path=project_root
            / expected_sources["reference_predictions"],
            candidate_path=project_root
            / expected_sources["candidate_predictions"],
            reference_name=reference_name,
            candidate_name=candidate_name,
            top_k=12,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            f"error analysis cannot be recomputed: {type(exc).__name__}: {exc}"
        )
        expected = None
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("error analysis is not complete and valid")
    if expected is not None and report != expected:
        for key in (
            "source_bindings",
            "model_aliases",
            "analysis",
            "claim_boundary",
        ):
            if report.get(key) != expected.get(key):
                errors.append(f"error analysis drift: {key}")
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("error analysis Markdown is missing")
    analysis = report.get("analysis")
    dataset = analysis.get("dataset", {}) if isinstance(analysis, Mapping) else {}
    paired = (
        analysis.get("paired_outcomes", {})
        if isinstance(analysis, Mapping)
        else {}
    )
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "sample_count": dataset.get("sample_count"),
        "source_image_count": dataset.get("source_image_count"),
        "candidate_only_correct": paired.get("candidate_only_correct"),
        "reference_only_correct": paired.get("reference_only_correct"),
        "errors": errors,
    }


def audit_image_recovery_bundle(
    report_path: Path,
    *,
    project_root: Path,
    historical_transition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"image recovery audit is missing: {report_path}"],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                f"image recovery audit cannot be parsed: {type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["image recovery audit must be a JSON object"],
        }
    historical = False
    if isinstance(historical_transition, Mapping):
        transition = historical_transition.get("transition", {})
        binding = historical_transition.get("source_bindings", {}).get(
            "recovery_audit", {}
        )
        historical = (
            historical_transition.get("valid") is True
            and isinstance(transition, Mapping)
            and isinstance(binding, Mapping)
            and binding.get("sha256") == _sha256(report_path)
            and transition.get("historical_source_sha256")
            == report.get("source", {}).get("sha256")
            and transition.get("locked_fallback_sha256")
            == report.get("fallback", {}).get("sha256")
        )
    try:
        expected = None if historical else audit_image_recovery(
            source=(
                project_root.parent
                / "data"
                / "images"
                / "s_cabblestone_6.jpg"
            ),
            fallback=(
                project_root.parent
                / "data"
                / "combination"
                / "s_cabblestone_6.jpg"
            ),
            manifests=(
                project_root / "data" / "qwen-material" / "train.jsonl",
                project_root / "data" / "qwen-material" / "val.jsonl",
                project_root / "data" / "qwen-material" / "test.jsonl",
            ),
            shim=project_root / "runtime_recovery" / "sitecustomize.py",
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            "image recovery audit cannot be recomputed: "
            f"{type(exc).__name__}: {exc}"
        )
        expected = None
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("image recovery audit is not complete and valid")
    if expected is not None and report != expected:
        for key in (
            "source",
            "fallback",
            "comparison",
            "manifest_occurrences",
            "runtime_shim",
            "original_data_write_performed",
            "boundaries",
        ):
            if report.get(key) != expected.get(key):
                errors.append(f"image recovery audit drift: {key}")
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("image recovery audit Markdown is missing")
    comparison = report.get("comparison", {})
    byte_comparison = (
        comparison.get("bytes", {})
        if isinstance(comparison, Mapping)
        else {}
    )
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "source_sha256": (
            report.get("source", {}).get("sha256")
            if isinstance(report.get("source"), Mapping)
            else None
        ),
        "fallback_sha256": (
            report.get("fallback", {}).get("sha256")
            if isinstance(report.get("fallback"), Mapping)
            else None
        ),
        "differing_byte_count": byte_comparison.get(
            "differing_byte_count"
        ),
        "historical_snapshot": historical,
        "errors": errors,
    }


def audit_image_recovery_incident_bundle(
    report_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"image recovery incident snapshot is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "image recovery incident cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["image recovery incident must be a JSON object"],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("image recovery incident is not complete and valid")
    snapshot = report.get("snapshot")
    if not isinstance(snapshot, Mapping):
        errors.append("image recovery incident snapshot is malformed")
        snapshot = {}
    calculated_snapshot_hash = hashlib.sha256(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if report.get("snapshot_sha256") != calculated_snapshot_hash:
        errors.append("image recovery incident snapshot hash mismatch")
    status = snapshot.get("formal_status", {})
    partial = snapshot.get("partial_training", {})
    if (
        not isinstance(status, Mapping)
        or status.get("state") != "failed"
        or "full_no_degradation_training failed"
        not in str(status.get("message", ""))
    ):
        errors.append("image recovery incident failed status is missing")
    if (
        not isinstance(partial, Mapping)
        or partial.get("completed_epochs") != 5
        or partial.get("last_completed_epoch") != 5
    ):
        errors.append("image recovery incident partial epoch record mismatch")
    invariants = report.get("recovery_invariants", {})
    if not isinstance(invariants, Mapping):
        errors.append("image recovery incident invariants are malformed")
        invariants = {}
    for key in (
        "training_core_modified",
        "training_config_modified",
        "data_manifest_modified",
        "original_image_modified",
    ):
        if invariants.get(key) is not False:
            errors.append(f"image recovery invariant mismatch: {key}")
    bindings = report.get("source_bindings")
    expected_binding_names = {
        "formal_status",
        "history",
        "best_checkpoint",
        "last_checkpoint",
        "config",
        "train_manifest",
        "formal_run_manifest",
        "recovery_audit",
        "runtime_shim",
        "recovery_chain",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != (
        expected_binding_names
    ):
        errors.append("image recovery incident source binding set mismatch")
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("image recovery incident Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "snapshot_sha256": report.get("snapshot_sha256"),
        "completed_epochs_before_resume": partial.get("completed_epochs"),
        "source_binding_count": (
            len(bindings) if isinstance(bindings, Mapping) else 0
        ),
        "errors": errors,
    }


def audit_source_image_repair_transition_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"source image repair transition is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "source image repair transition cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "source image repair transition must be a JSON object"
            ],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append(
            "source image repair transition is not complete and valid"
        )
    transition = report.get("transition", {})
    if not isinstance(transition, Mapping):
        errors.append("source image repair transition block is malformed")
        transition = {}
    if (
        transition.get("cause") != "unknown_external_state_change"
        or transition.get("source_now_equals_locked_fallback") is not True
        or transition.get("source_and_fallback_byte_identical") is not True
        or transition.get("current_source_sha256")
        != transition.get("locked_fallback_sha256")
        or transition.get("current_fallback_sha256")
        != transition.get("locked_fallback_sha256")
        or transition.get("current_raw_inventory_sha256")
        != transition.get("previous_effective_inventory_sha256")
        or transition.get("current_effective_inventory_sha256")
        != transition.get("current_raw_inventory_sha256")
    ):
        errors.append("source image repair transition invariants mismatch")

    bindings = report.get("source_bindings")
    expected_names = {
        "recovery_audit",
        "incident",
        "addendum",
        "outcome",
        "previous_integrity",
        "current_integrity",
        "source",
        "fallback",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_names:
        errors.append("source image repair binding set mismatch")
        bindings = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(f"source image repair binding malformed: {name}")
            continue
        source = Path(str(binding.get("path", "")))
        if not source.is_absolute():
            source = project_root / source
        if not source.is_file():
            errors.append(f"source image repair bound source missing: {name}")
            continue
        if (
            binding.get("size_bytes") != source.stat().st_size
            or binding.get("sha256") != _sha256(source)
        ):
            errors.append(f"source image repair binding drift: {name}")
    if not report_path.with_suffix(".md").is_file():
        errors.append("source image repair transition Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "cause": transition.get("cause"),
        "historical_source_sha256": transition.get(
            "historical_source_sha256"
        ),
        "current_source_sha256": transition.get("current_source_sha256"),
        "current_inventory_sha256": transition.get(
            "current_raw_inventory_sha256"
        ),
        "source_binding_count": len(bindings),
        "errors": errors,
    }


def audit_source_image_integrity_bundle(
    report_path: Path,
    *,
    project_root: Path,
    expected_recovery_state: str = "damaged",
    historical_transition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"source image integrity audit is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "source image integrity audit cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["source image integrity audit must be a JSON object"],
        }
    historical = False
    if isinstance(historical_transition, Mapping):
        transition = historical_transition.get("transition", {})
        binding = historical_transition.get("source_bindings", {}).get(
            "previous_integrity", {}
        )
        summary = report.get("summary", {})
        historical = (
            historical_transition.get("valid") is True
            and isinstance(transition, Mapping)
            and isinstance(binding, Mapping)
            and isinstance(summary, Mapping)
            and binding.get("sha256") == _sha256(report_path)
            and transition.get("previous_raw_inventory_sha256")
            == summary.get("raw_inventory_sha256")
            and transition.get("previous_effective_inventory_sha256")
            == summary.get("effective_inventory_sha256")
        )
    try:
        expected = None if historical else audit_source_image_integrity(
            manifests=(
                project_root / "data" / "qwen-material" / "train.jsonl",
                project_root / "data" / "qwen-material" / "val.jsonl",
                project_root / "data" / "qwen-material" / "test.jsonl",
            ),
            recovery_audit_path=(
                project_root
                / "reports"
                / "qwen-material"
                / "IMAGE_READ_RECOVERY_AUDIT.json"
            ),
            expected_recovery_state=expected_recovery_state,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            "source image integrity audit cannot be recomputed: "
            f"{type(exc).__name__}: {exc}"
        )
        expected = None
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append(
            "source image integrity audit is not complete and valid"
        )
    if expected is not None and report != expected:
        for key in (
            "manifest_bindings",
            "recovery_audit_binding",
            "summary",
            "images",
            "boundaries",
        ):
            if report.get(key) != expected.get(key):
                errors.append(f"source image integrity audit drift: {key}")
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("source image integrity audit Markdown is missing")
    summary = report.get("summary", {})
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "source_image_count": (
            summary.get("source_image_count")
            if isinstance(summary, Mapping)
            else None
        ),
        "decode_failure_count": (
            summary.get("decode_failure_count")
            if isinstance(summary, Mapping)
            else None
        ),
        "documented_recovery_count": (
            summary.get("documented_recovery_count")
            if isinstance(summary, Mapping)
            else None
        ),
        "raw_inventory_sha256": (
            summary.get("raw_inventory_sha256")
            if isinstance(summary, Mapping)
            else None
        ),
        "effective_inventory_sha256": (
            summary.get("effective_inventory_sha256")
            if isinstance(summary, Mapping)
            else None
        ),
        "expected_recovery_state": expected_recovery_state,
        "historical_snapshot": historical,
        "errors": errors,
    }


def audit_image_recovery_outcome_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"image recovery outcome is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "image recovery outcome cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["image recovery outcome must be a JSON object"],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("image recovery outcome is not complete and valid")

    recovery = report.get("recovery", {})
    training = report.get("training", {})
    test = report.get("test", {})
    if not isinstance(recovery, Mapping):
        errors.append("image recovery outcome recovery block is malformed")
        recovery = {}
    if not isinstance(training, Mapping):
        errors.append("image recovery outcome training block is malformed")
        training = {}
    if not isinstance(test, Mapping):
        errors.append("image recovery outcome test block is malformed")
        test = {}
    if (
        recovery.get("completed_epochs_before_resume") != 5
        or recovery.get("completed_epochs_after_resume") != 100
        or recovery.get("resume_stderr_empty") is not True
    ):
        errors.append("image recovery outcome resume boundary mismatch")
    resume_event = recovery.get("resume_event", {})
    if (
        not isinstance(resume_event, Mapping)
        or resume_event.get("event") != "resumed"
        or resume_event.get("completed_epochs") != 5
        or resume_event.get("target_epochs") != 100
    ):
        errors.append("image recovery outcome resume event mismatch")
    best_epoch = training.get("best_epoch")
    if (
        training.get("experiment") != "full_no_degradation"
        or training.get("checkpoint_audit_valid") is not True
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= 100
        or training.get("last_epoch") != 100
    ):
        errors.append("image recovery outcome training record mismatch")
    if (
        test.get("sample_count") != 826
        or test.get("checkpoint_epoch") != best_epoch
    ):
        errors.append("image recovery outcome test record mismatch")

    bindings = report.get("source_bindings")
    expected_names = {
        "incident",
        "addendum",
        "recovery_audit",
        "source_integrity",
        "history",
        "checkpoint_audit",
        "evaluation",
        "resume_stdout",
        "resume_stderr",
        "runtime_shim",
        "recovery_chain",
        "config",
        "train_manifest",
        "test_manifest",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_names:
        errors.append("image recovery outcome source binding set mismatch")
        bindings = {}
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(
                f"image recovery outcome source binding is malformed: {name}"
            )
            continue
        source = Path(str(binding.get("path", "")))
        if not source.is_absolute():
            source = project_root / source
        if not source.is_file():
            errors.append(
                f"image recovery outcome bound source is missing: {name}"
            )
            continue
        if (
            binding.get("size_bytes") != source.stat().st_size
            or binding.get("sha256") != _sha256(source)
        ):
            errors.append(
                f"image recovery outcome source binding drift: {name}"
            )
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("image recovery outcome Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "completed_epochs_before_resume": recovery.get(
            "completed_epochs_before_resume"
        ),
        "completed_epochs_after_resume": recovery.get(
            "completed_epochs_after_resume"
        ),
        "best_epoch": best_epoch,
        "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "source_binding_count": len(bindings),
        "errors": errors,
    }


def audit_formal_queue_interruption_bundle(
    report_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"formal queue interruption is missing: {report_path}"],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "formal queue interruption cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["formal queue interruption must be a JSON object"],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("formal queue interruption is not complete and valid")
    partial = report.get("partial_training", {})
    input_state = report.get("input_state", {})
    if (
        not isinstance(partial, Mapping)
        or partial.get("experiment") != "full_no_reconstruction"
        or partial.get("completed_epochs") != 68
        or partial.get("last_completed_epoch") != 68
        or partial.get("resume_checkpoint_role") != "last.pt"
        or partial.get("training_summary_exists") is not False
    ):
        errors.append("formal queue interruption partial record mismatch")
    if (
        not isinstance(input_state, Mapping)
        or input_state.get(
            "first_68_epochs_used_hash_locked_redirection"
        )
        is not True
        or input_state.get(
            "subsequent_epochs_use_repaired_source_without_shim"
        )
        is not True
        or input_state.get("effective_image_bytes_unchanged_across_resume")
        is not True
    ):
        errors.append("formal queue interruption input-state mismatch")
    bindings = report.get("source_bindings")
    if not isinstance(bindings, Mapping) or len(bindings) != 11:
        errors.append("formal queue interruption binding count mismatch")
        bindings = {}
    if not report_path.with_suffix(".md").is_file():
        errors.append("formal queue interruption Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "completed_epochs": partial.get("completed_epochs"),
        "best_epoch": partial.get("best_checkpoint_epoch"),
        "source_binding_count": len(bindings),
        "errors": errors,
    }


def audit_formal_queue_interruption_addendum_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"formal queue interruption addendum is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "formal queue interruption addendum cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "formal queue interruption addendum must be a JSON object"
            ],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append(
            "formal queue interruption addendum is not complete and valid"
        )
    snapshot = report.get("snapshot", {})
    failed = report.get("failed_resume", {})
    archive = report.get("archive", {})
    snapshot_path = project_root / str(snapshot.get("path", ""))
    if (
        not snapshot_path.is_file()
        or snapshot.get("sha256") != _sha256(snapshot_path)
    ):
        errors.append("formal queue interruption snapshot hash mismatch")
    if (
        not isinstance(failed, Mapping)
        or failed.get("last_checkpoint_zip_readable") is not False
        or failed.get("best_checkpoint_zip_readable") is not True
        or failed.get("optimizer_state_available_from_best") is not False
        or failed.get("decision")
        != "restart_same_formal_config_from_epoch_1"
    ):
        errors.append("formal queue interruption recovery decision mismatch")
    archived_files = (
        archive.get("files", {}) if isinstance(archive, Mapping) else {}
    )
    if not isinstance(archived_files, Mapping) or set(archived_files) != {
        "history",
        "best_checkpoint",
        "last_checkpoint",
    }:
        errors.append("formal queue interrupted archive set mismatch")
        archived_files = {}
    for name, binding in archived_files.items():
        source = project_root / str(binding.get("path", ""))
        if not source.is_file():
            errors.append(f"formal queue archived file missing: {name}")
        elif (
            binding.get("size_bytes") != source.stat().st_size
            or binding.get("sha256") != _sha256(source)
        ):
            errors.append(f"formal queue archived file drift: {name}")
    if not report_path.with_suffix(".md").is_file():
        errors.append("formal queue interruption addendum Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "decision": failed.get("decision"),
        "last_checkpoint_zip_readable": failed.get(
            "last_checkpoint_zip_readable"
        ),
        "archived_file_count": len(archived_files),
        "errors": errors,
    }


def audit_formal_queue_interruption_outcome_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"formal queue interruption outcome is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "formal queue interruption outcome cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "formal queue interruption outcome must be a JSON object"
            ],
        }
    try:
        expected = build_interruption_outcome(
            files={
                name: project_root / relative
                for name, relative in (
                    EXPECTED_FORMAL_INTERRUPTION_OUTCOME_FILES.items()
                )
            }
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            "formal queue interruption outcome cannot be recomputed: "
            f"{type(exc).__name__}: {exc}"
        )
        expected = None
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append(
            "formal queue interruption outcome is not complete and valid"
        )
    if expected is not None:
        for key in (
            "state",
            "valid",
            "scope",
            "interruption",
            "restart",
            "training",
            "test",
            "source_bindings",
            "errors",
            "boundaries",
        ):
            if report.get(key) != expected.get(key):
                errors.append(
                    f"formal queue interruption outcome drift: {key}"
                )
    if not report_path.with_suffix(".md").is_file():
        errors.append(
            "formal queue interruption outcome Markdown is missing"
        )
    restart = report.get("restart", {})
    training = report.get("training", {})
    test = report.get("test", {})
    bindings = report.get("source_bindings", {})
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "completed_epochs": (
            restart.get("completed_epochs")
            if isinstance(restart, Mapping)
            else None
        ),
        "best_epoch": (
            training.get("best_epoch")
            if isinstance(training, Mapping)
            else None
        ),
        "test_accuracy": (
            test.get("accuracy") if isinstance(test, Mapping) else None
        ),
        "test_macro_f1": (
            test.get("macro_f1") if isinstance(test, Mapping) else None
        ),
        "source_binding_count": (
            len(bindings) if isinstance(bindings, Mapping) else 0
        ),
        "errors": errors,
    }


def audit_yolo_seg_dependency_incident_bundle(
    incident_path: Path,
    addendum_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        incident = _read_json(incident_path)
        addendum = _read_json(addendum_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "YOLO-seg dependency incident cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(incident, Mapping) or not isinstance(addendum, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["YOLO-seg dependency incident roots must be objects"],
        }
    if (
        incident.get("schema_version") != 1
        or incident.get("state") != "complete"
        or incident.get("valid") is not True
    ):
        errors.append("YOLO-seg dependency incident is not complete and valid")
    if (
        addendum.get("schema_version") != 1
        or addendum.get("state") != "complete"
        or addendum.get("valid") is not True
    ):
        errors.append(
            "YOLO-seg dependency incident addendum is not complete and valid"
        )
    failure = incident.get("failure", {})
    remediation = incident.get("remediation", {})
    if (
        not isinstance(failure, Mapping)
        or failure.get("completed_epochs") != 1
        or failure.get("last_completed_epoch") != 1
        or failure.get("checkpoint_saved") is not False
        or failure.get("missing_module") != "polars"
        or failure.get("failure_point")
        != "Ultralytics BaseTrainer.save_model/read_results_csv"
    ):
        errors.append("YOLO-seg dependency failure record mismatch")
    if (
        not isinstance(remediation, Mapping)
        or remediation.get("decision")
        != "clean_restart_same_frozen_protocol_from_epoch_1"
        or remediation.get("archived_file_count") != 5
    ):
        errors.append("YOLO-seg dependency remediation record mismatch")

    incident_binding = addendum.get("incident", {})
    if (
        not isinstance(incident_binding, Mapping)
        or incident_binding.get("sha256") != _sha256(incident_path)
        or incident_binding.get("size_bytes") != incident_path.stat().st_size
    ):
        errors.append("YOLO-seg incident addendum does not bind the incident")
    original_bindings = incident.get("source_bindings", {})
    source_mirror = addendum.get("source_mirror", {})
    mirrored_bindings = (
        source_mirror.get("bindings", {})
        if isinstance(source_mirror, Mapping)
        else {}
    )
    if (
        not isinstance(original_bindings, Mapping)
        or not isinstance(mirrored_bindings, Mapping)
        or set(original_bindings) != set(mirrored_bindings)
        or source_mirror.get("file_count") != len(mirrored_bindings)
    ):
        errors.append("YOLO-seg incident source mirror set mismatch")
        original_bindings = {}
        mirrored_bindings = {}
    for name, mirrored in mirrored_bindings.items():
        original = original_bindings.get(name, {})
        if not isinstance(mirrored, Mapping) or not isinstance(
            original, Mapping
        ):
            errors.append(f"YOLO-seg incident mirror binding malformed: {name}")
            continue
        source = project_root / str(mirrored.get("path", ""))
        if not source.is_file():
            errors.append(f"YOLO-seg incident mirror file missing: {name}")
            continue
        if (
            mirrored.get("original_path") != original.get("path")
            or mirrored.get("size_bytes") != source.stat().st_size
            or mirrored.get("size_bytes") != original.get("size_bytes")
            or mirrored.get("sha256") != _sha256(source)
            or mirrored.get("sha256") != original.get("sha256")
        ):
            errors.append(f"YOLO-seg incident mirror binding drift: {name}")

    checkpoint_archive = addendum.get("checkpoint_archive", {})
    archived_bindings = (
        checkpoint_archive.get("bindings", {})
        if isinstance(checkpoint_archive, Mapping)
        else {}
    )
    if (
        not isinstance(archived_bindings, Mapping)
        or set(archived_bindings)
        != {"args.yaml", "results.csv", "train_batch0.jpg",
            "train_batch1.jpg", "train_batch2.jpg"}
        or checkpoint_archive.get("file_count") != 5
    ):
        errors.append("YOLO-seg failed checkpoint archive set mismatch")
        archived_bindings = {}
    checkpoint_root = project_root / str(
        checkpoint_archive.get("path", "")
    )
    for name, binding in archived_bindings.items():
        source = checkpoint_root / name
        if not source.is_file():
            errors.append(f"YOLO-seg failed archive file missing: {name}")
        elif (
            binding.get("size_bytes") != source.stat().st_size
            or binding.get("sha256") != _sha256(source)
        ):
            errors.append(f"YOLO-seg failed archive file drift: {name}")
    results_path = checkpoint_root / "results.csv"
    if results_path.is_file():
        rows = list(
            csv.DictReader(
                results_path.read_text(encoding="utf-8-sig").splitlines()
            )
        )
        if len(rows) != 1 or rows[0].get("epoch") != "1":
            errors.append("YOLO-seg failed archive epoch record mismatch")

    if mirrored_bindings:
        stderr_path = project_root / str(
            mirrored_bindings["yolo_stderr"].get("path", "")
        )
        if stderr_path.is_file():
            stderr = stderr_path.read_text(encoding="utf-8-sig")
            if (
                "ModuleNotFoundError: No module named 'polars'" not in stderr
                or "read_results_csv" not in stderr
            ):
                errors.append("YOLO-seg mirrored traceback mismatch")
    for path in (incident_path.with_suffix(".md"), addendum_path.with_suffix(".md")):
        if not path.is_file():
            errors.append(f"YOLO-seg incident Markdown is missing: {path.name}")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "incident": incident_path.as_posix(),
        "incident_sha256": _sha256(incident_path),
        "addendum": addendum_path.as_posix(),
        "addendum_sha256": _sha256(addendum_path),
        "completed_epochs": failure.get("completed_epochs"),
        "checkpoint_saved": failure.get("checkpoint_saved"),
        "source_mirror_count": len(mirrored_bindings),
        "checkpoint_archive_count": len(archived_bindings),
        "errors": errors,
    }


def _audit_current_source_bindings(
    report: Mapping[str, Any],
    *,
    project_root: Path,
    prefix: str,
    errors: list[str],
) -> int:
    bindings = report.get("source_bindings", {})
    if not isinstance(bindings, Mapping) or not bindings:
        errors.append(f"{prefix} source bindings are missing")
        return 0
    root = project_root.resolve()
    checked = 0
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(f"{prefix} source binding is malformed: {name}")
            continue
        source = Path(str(binding.get("path", "")))
        source = (
            source.resolve()
            if source.is_absolute()
            else (root / source).resolve()
        )
        try:
            source.relative_to(root)
        except ValueError:
            errors.append(f"{prefix} source escapes project: {name}")
            continue
        if not source.is_file():
            errors.append(f"{prefix} source is missing: {name}")
            continue
        if (
            binding.get("size_bytes") != source.stat().st_size
            or binding.get("sha256") != _sha256(source)
        ):
            errors.append(f"{prefix} source binding drift: {name}")
        checked += 1
    return checked


def _validated_regression_source_transition(
    *,
    project_root: Path,
    relative_path: str,
) -> dict[str, Any] | None:
    root = project_root.resolve()
    regression_path = (
        root / "reports" / "qwen-material" / "FULL_REGRESSION_AUDIT.json"
    )
    try:
        regression = _read_json(regression_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(regression, Mapping):
        return None
    combined = regression.get("combined", {})
    bindings = regression.get("source_bindings", {})
    if (
        regression.get("schema_version") != 1
        or regression.get("state") != "complete"
        or regression.get("valid") is not True
        or regression.get("errors") != []
        or not isinstance(combined, Mapping)
        or combined.get("all_collected_tests_passed") is not True
        or combined.get("unresolved_skips") != 0
        or combined.get("unresolved_failures") != 0
        or not isinstance(bindings, Mapping)
    ):
        return None

    allowed_transition_paths = {
        "build_evidence_archive.py",
        "audit_paper_readiness.py",
    }
    if relative_path not in allowed_transition_paths:
        return None
    required_paths = tuple(
        dict.fromkeys(
            (
                relative_path,
                "audit_paper_readiness.py",
                "tests/test_paper_readiness_audit.py",
            )
        )
    )
    current_bindings: dict[str, Any] = {}
    for required_path in required_paths:
        matches = [
            binding
            for binding in bindings.values()
            if isinstance(binding, Mapping)
            and binding.get("path") == required_path
        ]
        source = (root / required_path).resolve()
        if (
            len(matches) != 1
            or not source.is_file()
            or matches[0].get("size_bytes") != source.stat().st_size
            or matches[0].get("sha256") != _sha256(source)
        ):
            return None
        current_bindings[required_path] = dict(matches[0])
    return {
        "scope": "non_algorithmic_historical_source_transition",
        "current_bindings": current_bindings,
        "full_regression_audit": {
            "path": regression_path.relative_to(root).as_posix(),
            "size_bytes": regression_path.stat().st_size,
            "sha256": _sha256(regression_path),
        },
    }


def audit_yolo_seg_dependency_outcome_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "YOLO-seg dependency outcome cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["YOLO-seg dependency outcome must be an object"],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
        or report.get("scope")
        != "yolo_seg_clean_restart_after_dependency_interruption"
        or report.get("errors") != []
    ):
        errors.append("YOLO-seg dependency outcome is not complete and valid")
    restart = report.get("restart", {})
    if (
        not isinstance(restart, Mapping)
        or restart.get("failed_run_completed_epochs") != 1
        or restart.get("failed_run_checkpoint_saved") is not False
        or restart.get("formal_completed_epochs") != 100
        or restart.get("formal_epoch_sequence") != [1, 100]
        or restart.get("resume") is not False
        or restart.get("same_frozen_protocol") is not True
    ):
        errors.append("YOLO-seg clean-restart outcome mismatch")
    detector = report.get("detector_test", {})
    integrated = report.get("integrated_scene", {})
    if (
        not isinstance(detector, Mapping)
        or detector.get("source_image_count") != 570
        or detector.get("object_count") != 825
        or not isinstance(integrated, Mapping)
        or integrated.get("source_image_count") != 570
        or integrated.get("ground_truth_object_count") != 825
        or integrated.get("scenario_count") != 2
    ):
        errors.append("YOLO-seg outcome dataset counts mismatch")
    checked = _audit_current_source_bindings(
        report,
        project_root=project_root,
        prefix="YOLO-seg dependency outcome",
        errors=errors,
    )
    if checked != 23:
        errors.append(
            f"YOLO-seg dependency outcome source count mismatch: {checked}"
        )
    if not report_path.with_suffix(".md").is_file():
        errors.append("YOLO-seg dependency outcome Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "formal_completed_epochs": (
            restart.get("formal_completed_epochs")
            if isinstance(restart, Mapping)
            else None
        ),
        "source_binding_count": checked,
        "errors": errors,
    }


def audit_scene_protocol_addenda_bundle(
    integrated_path: Path,
    stratified_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        integrated = _read_json(integrated_path)
        stratified = _read_json(stratified_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "scene protocol addenda cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(integrated, Mapping) or not isinstance(
        stratified, Mapping
    ):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["scene protocol addenda must be objects"],
        }
    if (
        integrated.get("schema_version") != 1
        or integrated.get("status")
        != "frozen_after_path_identity_failure_before_recovery_rerun"
        or integrated.get("valid") is not True
        or integrated.get("errors") != []
    ):
        errors.append("integrated scene protocol addendum is invalid")
    if (
        stratified.get("schema_version") != 1
        or stratified.get("status")
        != "frozen_after_integrated_path_addendum_before_stratified_run"
        or stratified.get("valid") is not True
        or stratified.get("errors") != []
    ):
        errors.append("stratified scene protocol addendum is invalid")
    if (
        stratified.get("dependency_protocol_addendum", {}).get("sha256")
        != _sha256(integrated_path)
    ):
        errors.append("stratified addendum does not bind integrated addendum")
    integrated_count = _audit_current_source_bindings(
        integrated,
        project_root=project_root,
        prefix="integrated scene protocol addendum",
        errors=errors,
    )
    stratified_count = _audit_current_source_bindings(
        stratified,
        project_root=project_root,
        prefix="stratified scene protocol addendum",
        errors=errors,
    )
    if integrated_count != 13:
        errors.append(
            "integrated scene addendum source count mismatch: "
            f"{integrated_count}"
        )
    if stratified_count != 12:
        errors.append(
            "stratified scene addendum source count mismatch: "
            f"{stratified_count}"
        )
    invariants = stratified.get("invariants", {})
    if (
        not isinstance(invariants, Mapping)
        or invariants.get("confidence_thresholds") != [0.25, 0.8]
        or invariants.get("bootstrap_repeats") != 2000
        or invariants.get("bootstrap_seed") != 20260727
        or invariants.get("expected_row_count") != 42
        or invariants.get("metric_definitions_unchanged") is not True
    ):
        errors.append("stratified scene addendum invariants mismatch")
    for path in (
        integrated_path.with_suffix(".md"),
        stratified_path.with_suffix(".md"),
    ):
        if not path.is_file():
            errors.append(f"scene protocol addendum Markdown is missing: {path}")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "integrated_addendum": integrated_path.as_posix(),
        "integrated_addendum_sha256": _sha256(integrated_path),
        "stratified_addendum": stratified_path.as_posix(),
        "stratified_addendum_sha256": _sha256(stratified_path),
        "source_binding_count": integrated_count + stratified_count,
        "errors": errors,
    }


def audit_scene_strata_reporting_recovery_bundle(
    incident_path: Path,
    addendum_path: Path,
    outcome_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        incident = _read_json(incident_path)
        addendum = _read_json(addendum_path)
        outcome = _read_json(outcome_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "scene-strata reporting recovery cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not all(
        isinstance(value, Mapping)
        for value in (incident, addendum, outcome)
    ):
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "scene-strata reporting recovery artifacts must be objects"
            ],
        }
    if (
        incident.get("schema_version") != 1
        or incident.get("state") != "complete"
        or incident.get("valid") is not True
        or incident.get("failure", {}).get("exception_type") != "KeyError"
        or incident.get("partial_outputs", {}).get("row_count") != 42
        or incident.get("scope", {}).get("metric_computation_completed")
        is not True
        or incident.get("scope", {}).get("metric_values_invalidated")
        is not False
    ):
        errors.append("scene-strata reporting incident is invalid")
    if (
        addendum.get("schema_version") != 1
        or addendum.get("status")
        != "frozen_after_metrics_complete_before_report_recovery"
        or addendum.get("valid") is not True
        or addendum.get("errors") != []
    ):
        errors.append("scene-strata reporting addendum is invalid")
    if addendum.get("incident", {}).get("sha256") != _sha256(incident_path):
        errors.append("scene-strata reporting addendum incident hash mismatch")
    invariants = addendum.get("invariants", {})
    if (
        not isinstance(invariants, Mapping)
        or invariants.get("metric_recomputation") is not False
        or invariants.get("training_rerun") is not False
        or invariants.get("detector_rerun") is not False
        or invariants.get("matching_rerun") is not False
        or invariants.get("bootstrap_rerun") is not False
        or invariants.get("expected_row_count") != 42
        or invariants.get("confidence_thresholds") != [0.25, 0.8]
        or invariants.get("bootstrap_repeats") != 2000
        or invariants.get("bootstrap_seed") != 20260727
    ):
        errors.append("scene-strata reporting addendum invariants mismatch")
    archive_count = _audit_current_source_bindings(
        {"source_bindings": incident.get("archive", {}).get("bindings", {})},
        project_root=project_root,
        prefix="scene-strata reporting incident archive",
        errors=errors,
    )
    if archive_count != 12:
        errors.append(
            "scene-strata reporting incident archive count mismatch: "
            f"{archive_count}"
        )
    addendum_for_current_audit = addendum
    historical_source_transitions = []
    addendum_bindings = addendum.get("source_bindings", {})
    allowed_historical_transitions = {
        "evidence_archive_builder": "build_evidence_archive.py",
        "paper_readiness_audit": "audit_paper_readiness.py",
    }
    if isinstance(addendum_bindings, Mapping):
        current_bindings = dict(addendum_bindings)
        for name, expected_relative in allowed_historical_transitions.items():
            historical_binding = addendum_bindings.get(name)
            if not isinstance(historical_binding, Mapping):
                continue
            relative_path = str(historical_binding.get("path", ""))
            source_path = (project_root.resolve() / relative_path).resolve()
            drifted = (
                not source_path.is_file()
                or historical_binding.get("size_bytes")
                != source_path.stat().st_size
                or historical_binding.get("sha256") != _sha256(source_path)
            )
            if not drifted:
                continue
            if relative_path != expected_relative:
                errors.append(
                    "scene-strata reporting addendum historical source "
                    f"path mismatch: {name}"
                )
                continue
            transition = _validated_regression_source_transition(
                project_root=project_root,
                relative_path=relative_path,
            )
            if transition is None:
                errors.append(
                    "scene-strata reporting addendum non-algorithmic "
                    f"source transition is not regression-bound: {name}"
                )
            else:
                current_bindings[name] = (
                    transition["current_bindings"][relative_path]
                )
                historical_source_transitions.append(
                    {
                        "name": name,
                        "historical_binding": dict(historical_binding),
                        **transition,
                    }
                )
        if historical_source_transitions:
            addendum_for_current_audit = dict(addendum)
            addendum_for_current_audit["source_bindings"] = current_bindings
    addendum_count = _audit_current_source_bindings(
        addendum_for_current_audit,
        project_root=project_root,
        prefix="scene-strata reporting addendum",
        errors=errors,
    )
    if addendum_count != 17:
        errors.append(
            "scene-strata reporting addendum source count mismatch: "
            f"{addendum_count}"
        )
    if (
        outcome.get("schema_version") != 1
        or outcome.get("state") != "complete"
        or outcome.get("valid") is not True
        or outcome.get("errors") != []
        or outcome.get("incident", {}).get("sha256")
        != _sha256(incident_path)
        or outcome.get("addendum", {}).get("sha256")
        != _sha256(addendum_path)
    ):
        errors.append("scene-strata reporting recovery outcome is invalid")
    recovery = outcome.get("recovery", {})
    if (
        not isinstance(recovery, Mapping)
        or recovery.get("scope")
        != "generate_missing_markdown_from_persisted_metrics"
        or recovery.get("metric_recomputation") is not False
        or recovery.get("training_rerun") is not False
        or recovery.get("detector_rerun") is not False
        or recovery.get("row_count") != 42
        or recovery.get("metric_hashes_unchanged") is not True
        or recovery.get("before_hashes") != recovery.get("after_hashes")
    ):
        errors.append("scene-strata reporting recovery invariants mismatch")
    outcome_count = _audit_current_source_bindings(
        outcome,
        project_root=project_root,
        prefix="scene-strata reporting outcome",
        errors=errors,
    )
    if outcome_count != 6:
        errors.append(
            "scene-strata reporting outcome source count mismatch: "
            f"{outcome_count}"
        )
    for path in (
        incident_path.with_suffix(".md"),
        addendum_path.with_suffix(".md"),
        outcome_path.with_suffix(".md"),
    ):
        if not path.is_file():
            errors.append(
                f"scene-strata reporting recovery Markdown is missing: {path}"
            )
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "incident": incident_path.as_posix(),
        "incident_sha256": _sha256(incident_path),
        "addendum": addendum_path.as_posix(),
        "addendum_sha256": _sha256(addendum_path),
        "outcome": outcome_path.as_posix(),
        "outcome_sha256": _sha256(outcome_path),
        "metric_hashes_unchanged": (
            recovery.get("metric_hashes_unchanged")
            if isinstance(recovery, Mapping)
            else None
        ),
        "source_binding_count": (
            archive_count + addendum_count + outcome_count
        ),
        "historical_source_transitions": historical_source_transitions,
        "errors": errors,
    }


def audit_sensor_retraining_exit_bundle(
    incident_path: Path,
    addendum_path: Path,
    encoding_addendum_path: Path,
    encoding_repair_path: Path,
    outcome_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        incident = _read_json(incident_path)
        addendum = _read_json(addendum_path)
        encoding_addendum = _read_json(encoding_addendum_path)
        encoding_repair = _read_json(encoding_repair_path)
        outcome = _read_json(outcome_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "sensor retraining exit bundle cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not all(
        isinstance(value, Mapping)
        for value in (
            incident,
            addendum,
            encoding_addendum,
            encoding_repair,
            outcome,
        )
    ):
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "sensor retraining exit bundle roots must be objects"
            ],
        }
    if (
        incident.get("schema_version") != 1
        or incident.get("state") != "complete"
        or incident.get("valid") is not True
        or incident.get("failure", {}).get("exit_code") != -1073741819
        or incident.get("failure", {}).get("stderr_empty") is not True
        or incident.get("completed_training", {}).get("completed_epochs")
        != 100
        or incident.get("completed_training", {}).get("best_epoch") != 91
        or incident.get("completed_training", {}).get("last_epoch") != 100
        or incident.get("recovery", {}).get("decision")
        != "reuse_valid_complete_training_bundle_and_skip_retraining"
        or incident.get("errors") != []
    ):
        errors.append("sensor retraining exit incident mismatch")
    if (
        addendum.get("schema_version") != 1
        or addendum.get("status")
        != "frozen_after_recovery_queue_path_failure_before_recovery"
        or addendum.get("valid") is not True
        or addendum.get("incident", {}).get("sha256")
        != _sha256(incident_path)
        or addendum.get("invariants", {}).get("training_rerun") is not False
        or addendum.get("errors") != []
    ):
        errors.append("sensor retraining exit addendum mismatch")
    if (
        outcome.get("schema_version") != 1
        or outcome.get("state") != "complete"
        or outcome.get("valid") is not True
        or outcome.get("incident", {}).get("sha256")
        != _sha256(incident_path)
        or outcome.get("recovery", {}).get("training_rerun") is not False
        or outcome.get("recovery", {}).get("training_skip_verified")
        is not True
        or outcome.get("recovery", {}).get("completed_epochs") != 100
        or outcome.get("recovery", {}).get("best_epoch") != 91
        or outcome.get("recovery", {}).get("last_epoch") != 100
        or outcome.get("recovery", {}).get("test_sample_count") != 826
        or outcome.get("recovery", {}).get("paired_bootstrap_repeats")
        != 5000
        or outcome.get("errors") != []
    ):
        errors.append("sensor retraining exit outcome mismatch")
    if (
        encoding_repair.get("schema_version") != 1
        or encoding_repair.get("state") != "complete"
        or encoding_repair.get("valid") is not True
        or encoding_repair.get("decoded_text_equal") is not True
        or encoding_repair.get("original", {}).get("encoding")
        != "utf-16-with-bom"
        or encoding_repair.get("repaired", {}).get("encoding") != "utf-8"
    ):
        errors.append("sensor retraining log encoding repair mismatch")
    if (
        encoding_addendum.get("schema_version") != 1
        or encoding_addendum.get("status")
        != "frozen_after_outcome_log_encoding_failure_before_retry"
        or encoding_addendum.get("valid") is not True
        or encoding_addendum.get("incident", {}).get("sha256")
        != _sha256(incident_path)
        or encoding_addendum.get("first_addendum", {}).get("sha256")
        != _sha256(addendum_path)
        or encoding_addendum.get("invariants", {}).get("training_rerun")
        is not False
        or encoding_addendum.get("invariants", {}).get(
            "checkpoint_audit_rerun"
        )
        is not False
        or encoding_addendum.get("invariants", {}).get("fixed_test_rerun")
        is not False
        or encoding_addendum.get("invariants", {}).get(
            "paired_comparison_rerun"
        )
        is not False
        or encoding_addendum.get("errors") != []
    ):
        errors.append("sensor retraining encoding addendum mismatch")
    incident_count = _audit_current_source_bindings(
        incident,
        project_root=project_root,
        prefix="sensor retraining exit incident",
        errors=errors,
    )
    addendum_count = _audit_current_source_bindings(
        addendum,
        project_root=project_root,
        prefix="sensor retraining exit addendum",
        errors=errors,
    )
    encoding_addendum_count = _audit_current_source_bindings(
        encoding_addendum,
        project_root=project_root,
        prefix="sensor retraining encoding addendum",
        errors=errors,
    )
    outcome_count = _audit_current_source_bindings(
        outcome,
        project_root=project_root,
        prefix="sensor retraining exit outcome",
        errors=errors,
    )
    if incident_count != 16:
        errors.append(
            f"sensor retraining incident source count mismatch: {incident_count}"
        )
    if addendum_count != 10:
        errors.append(
            f"sensor retraining addendum source count mismatch: {addendum_count}"
        )
    if outcome_count != 17:
        errors.append(
            f"sensor retraining outcome source count mismatch: {outcome_count}"
        )
    if encoding_addendum_count != 18:
        errors.append(
            "sensor retraining encoding addendum source count mismatch: "
            f"{encoding_addendum_count}"
        )
    repair_original = Path(
        str(encoding_repair.get("original", {}).get("path", ""))
    )
    repair_repaired = Path(
        str(encoding_repair.get("repaired", {}).get("path", ""))
    )
    for name, path, binding in (
        (
            "original",
            repair_original,
            encoding_repair.get("original", {}),
        ),
        (
            "repaired",
            repair_repaired,
            encoding_repair.get("repaired", {}),
        ),
    ):
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (project_root / path).resolve()
        )
        if not resolved.is_file():
            errors.append(
                f"sensor retraining encoding repair {name} log is missing"
            )
        elif (
            binding.get("size_bytes") != resolved.stat().st_size
            or binding.get("sha256") != _sha256(resolved)
        ):
            errors.append(
                f"sensor retraining encoding repair {name} log drift"
            )
    for path in (
        incident_path.with_suffix(".md"),
        addendum_path.with_suffix(".md"),
        encoding_addendum_path.with_suffix(".md"),
        encoding_repair_path.with_suffix(".md"),
        outcome_path.with_suffix(".md"),
    ):
        if not path.is_file():
            errors.append(
                f"sensor retraining exit Markdown is missing: {path.name}"
            )
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "incident": incident_path.as_posix(),
        "incident_sha256": _sha256(incident_path),
        "addendum": addendum_path.as_posix(),
        "addendum_sha256": _sha256(addendum_path),
        "encoding_addendum": encoding_addendum_path.as_posix(),
        "encoding_addendum_sha256": _sha256(encoding_addendum_path),
        "encoding_repair": encoding_repair_path.as_posix(),
        "encoding_repair_sha256": _sha256(encoding_repair_path),
        "outcome": outcome_path.as_posix(),
        "outcome_sha256": _sha256(outcome_path),
        "completed_epochs": outcome.get("recovery", {}).get(
            "completed_epochs"
        ),
        "training_rerun": outcome.get("recovery", {}).get("training_rerun"),
        "source_binding_count": (
            incident_count
            + addendum_count
            + encoding_addendum_count
            + outcome_count
        ),
        "errors": errors,
    }


def audit_image_recovery_addendum_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                f"image recovery addendum is missing: {report_path}"
            ],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "image recovery addendum cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["image recovery addendum must be a JSON object"],
        }
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("image recovery addendum is not complete and valid")
    incident = report.get("incident_snapshot", {})
    attempt = report.get("resume_attempt_1", {})
    audit_revision = report.get("audit_compatibility_revision", {})
    functional_revision = report.get("functional_revision", {})
    attempt_2 = report.get("resume_attempt_2", {})
    reconstruction = report.get("reconstruction", {})
    boundary = report.get("behavioral_boundary", {})
    incident_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "IMAGE_READ_RECOVERY_INCIDENT.json"
    )
    shim_path = project_root / "runtime_recovery" / "sitecustomize.py"
    recovery_chain_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "training"
        / "run_image_recovery_chain.ps1"
    )
    attempt_stdout_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "training"
        / "image_recovery_chain.full_no_degradation_resume.stdout.log"
    )
    attempt_stderr_path = attempt_stdout_path.with_name(
        "image_recovery_chain.full_no_degradation_resume.stderr.log"
    )
    if (
        not isinstance(incident, Mapping)
        or not incident_path.is_file()
        or incident.get("sha256") != _sha256(incident_path)
    ):
        errors.append("image recovery addendum incident hash mismatch")
    if (
        not isinstance(functional_revision, Mapping)
        or not shim_path.is_file()
        or functional_revision.get("sha256") != _sha256(shim_path)
    ):
        errors.append(
            "image recovery addendum functional shim hash mismatch"
        )
    else:
        if not isinstance(reconstruction, Mapping):
            errors.append(
                "image recovery addendum reconstruction is malformed"
            )
            reconstruction = {}
        audit_line = (
            str(
                reconstruction.get(
                    "line_to_remove_for_attempt_1",
                    "",
                )
            ).encode("utf-8")
            + b"\n"
        )
        block_rows = reconstruction.get("current_block_to_replace", [])
        block = (
            ("\n".join(str(row) for row in block_rows) + "\n").encode(
                "utf-8"
            )
            if isinstance(block_rows, list) and block_rows
            else b""
        )
        replacement = (
            str(
                reconstruction.get(
                    "attempt_1_replacement_line",
                    "",
                )
            ).encode("utf-8")
            + b"\n"
        )
        shim_bytes = shim_path.read_bytes()
        if (
            not audit_line.strip()
            or shim_bytes.count(audit_line) != 1
            or not block.strip()
            or shim_bytes.count(block) != 1
            or not replacement.strip()
        ):
            errors.append(
                "image recovery addendum reconstruction operation mismatch"
            )
        else:
            attempt_bytes = shim_bytes.replace(audit_line, b"", 1).replace(
                block,
                replacement,
                1,
            )
            audit_bytes = shim_bytes.replace(block, replacement, 1)
            attempt_hash = hashlib.sha256(attempt_bytes).hexdigest()
            audit_hash = hashlib.sha256(audit_bytes).hexdigest()
            if (
                not isinstance(attempt, Mapping)
                or attempt.get("startup_runtime_shim_sha256")
                != attempt_hash
            ):
                errors.append(
                    "image recovery addendum attempt-1 shim hash mismatch"
                )
            if (
                not isinstance(audit_revision, Mapping)
                or audit_revision.get("sha256") != audit_hash
            ):
                errors.append(
                    "image recovery addendum audit shim hash mismatch"
                )
    if (
        not recovery_chain_path.is_file()
        or not isinstance(functional_revision, Mapping)
        or functional_revision.get("recovery_chain_sha256")
        != _sha256(recovery_chain_path)
    ):
        errors.append(
            "image recovery addendum recovery-chain hash mismatch"
        )
    if (
        not isinstance(attempt_2, Mapping)
        or attempt_2.get("startup_runtime_shim_sha256")
        != functional_revision.get("sha256")
        or attempt_2.get("recovery_chain_sha256")
        != functional_revision.get("recovery_chain_sha256")
        or attempt_2.get("completed_epochs_before") != 5
    ):
        errors.append(
            "image recovery addendum attempt-2 binding mismatch"
        )
    if not isinstance(attempt, Mapping):
        errors.append("image recovery addendum attempt-1 is malformed")
        attempt = {}
    else:
        for path, key, label in (
            (attempt_stdout_path, "stdout_sha256", "stdout"),
            (attempt_stderr_path, "stderr_sha256", "stderr"),
        ):
            if not path.is_file() or attempt.get(key) != _sha256(path):
                errors.append(
                    f"image recovery addendum attempt-1 {label} hash mismatch"
                )
        failed_status = attempt.get("failed_status_payload", {})
        if (
            not isinstance(failed_status, Mapping)
            or failed_status.get("state") != "failed"
            or "full_no_degradation_resume failed"
            not in str(failed_status.get("message", ""))
        ):
            errors.append(
                "image recovery addendum attempt-1 status is malformed"
            )
        if (
            attempt.get("completed_epochs_before") != 5
            or attempt.get("completed_epochs_after") != 5
            or attempt.get("checkpoint_write_after_resume") is not False
        ):
            errors.append(
                "image recovery addendum attempt-1 no-progress boundary "
                "mismatch"
            )
    expected_false = (
        "attempt_1_completed_new_epoch",
        "attempt_1_changed_checkpoint",
        "functional_revision_changes_model_code",
        "functional_revision_changes_training_config",
        "functional_revision_changes_manifest",
    )
    if not isinstance(boundary, Mapping):
        errors.append("image recovery addendum boundary is malformed")
    else:
        for key in expected_false:
            if boundary.get(key) is not False:
                errors.append(
                    f"image recovery addendum boundary mismatch: {key}"
                )
        if boundary.get(
            "functional_revision_changes_only_path_argument_resolution"
        ) is not True:
            errors.append(
                "image recovery addendum functional boundary is missing"
            )
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("image recovery addendum Markdown is missing")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "attempt_1_runtime_shim_sha256": (
            attempt.get("startup_runtime_shim_sha256")
            if isinstance(attempt, Mapping)
            else None
        ),
        "audit_runtime_shim_sha256": (
            audit_revision.get("sha256")
            if isinstance(audit_revision, Mapping)
            else None
        ),
        "functional_runtime_shim_sha256": (
            functional_revision.get("sha256")
            if isinstance(functional_revision, Mapping)
            else None
        ),
        "errors": errors,
    }


def audit_class_metric_intervals_bundle(
    report_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if not report_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"class metric intervals are missing: {report_path}"],
        }
    try:
        report = _read_json(report_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                "class metric intervals cannot be parsed: "
                f"{type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["class metric intervals must be a JSON object"],
        }
    try:
        expected = build_class_metric_intervals(
            evaluation_path=(
                project_root / EXPECTED_CLASS_INTERVAL_SOURCES["evaluation"]
            ),
            predictions_path=(
                project_root / EXPECTED_CLASS_INTERVAL_SOURCES["predictions"]
            ),
            repeats=2000,
            confidence=0.95,
            seed=20260727,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            "class metric intervals cannot be recomputed: "
            f"{type(exc).__name__}: {exc}"
        )
        expected = None
    if (
        report.get("schema_version") != 1
        or report.get("state") != "complete"
        or report.get("valid") is not True
    ):
        errors.append("class metric intervals are not complete and valid")
    if expected is not None and report != expected:
        for key in (
            "input_sources",
            "dataset",
            "method",
            "metrics",
            "boundaries",
        ):
            if report.get(key) != expected.get(key):
                errors.append(f"class metric intervals drift: {key}")
    markdown_path = report_path.with_suffix(".md")
    if not markdown_path.is_file():
        errors.append("class metric intervals Markdown is missing")
    dataset = report.get("dataset", {})
    metrics = report.get("metrics", [])
    supports = [
        int(row["support"])
        for row in metrics
        if isinstance(row, Mapping) and row.get("support") is not None
    ]
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "report": report_path.as_posix(),
        "report_sha256": _sha256(report_path),
        "sample_count": (
            dataset.get("sample_count")
            if isinstance(dataset, Mapping)
            else None
        ),
        "source_image_count": (
            dataset.get("source_image_count")
            if isinstance(dataset, Mapping)
            else None
        ),
        "class_count": (
            dataset.get("class_count")
            if isinstance(dataset, Mapping)
            else None
        ),
        "minimum_class_support": min(supports) if supports else None,
        "errors": errors,
    }


def audit_paper_table_bundle(
    table_root: Path,
    *,
    final_audit_path: Path,
) -> dict[str, Any]:
    manifest_path = table_root / "manifest.json"
    markdown_path = table_root / "TABLES.md"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [f"paper table manifest is missing: {manifest_path}"],
        }
    try:
        manifest = _read_json(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                f"paper table manifest cannot be parsed: {type(exc).__name__}"
            ],
        }
    if not isinstance(manifest, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["paper table manifest must be a JSON object"],
        }
    if manifest.get("source_audit_valid") is not True:
        errors.append("paper table manifest does not record a valid source audit")
    if manifest.get("schema_version") != 2:
        errors.append("paper table manifest schema version mismatch")
    if manifest.get("formal_ablation_scope") != (
        "legacy_full_reference_not_v3_component_ablation"
    ):
        errors.append("paper table formal ablation scope is missing")
    if manifest.get("main_paired_evidence_count") != 2:
        errors.append("paper table main paired evidence count mismatch")
    if not final_audit_path.is_file():
        errors.append(f"final evidence audit is missing: {final_audit_path}")
    elif manifest.get("source_audit_sha256") != _sha256(final_audit_path):
        errors.append("paper table source audit hash does not match")
    if manifest.get("figures") != list(EXPECTED_PAPER_FIGURES):
        errors.append("paper figure set mismatch")
    expected_artifacts = (
        *EXPECTED_PAPER_TABLES,
        "TABLES.md",
        *EXPECTED_PAPER_FIGURES,
    )
    table_counts = manifest.get("tables")
    if not isinstance(table_counts, Mapping) or set(table_counts) != set(
        EXPECTED_PAPER_TABLES
    ):
        errors.append("paper table set mismatch")
    recorded_hashes = manifest.get("artifact_sha256", {})
    for filename in expected_artifacts:
        path = table_root / filename
        if not path.is_file():
            errors.append(f"paper table artifact is missing: {path}")
            continue
        if recorded_hashes.get(filename) != _sha256(path):
            errors.append(f"paper table artifact hash mismatch: {filename}")
    if not markdown_path.is_file():
        errors.append(f"paper table Markdown is missing: {markdown_path}")

    source_reports = manifest.get("source_reports")
    if not isinstance(source_reports, Mapping) or set(source_reports) != set(
        EXPECTED_PAPER_SOURCE_REPORTS
    ):
        errors.append("paper table source report set mismatch")
        source_reports = {}
    reports_root = table_root.parent.resolve()
    for name, relative in EXPECTED_PAPER_SOURCE_REPORTS.items():
        expected_path = (reports_root / relative).resolve()
        entry = source_reports.get(name)
        if not isinstance(entry, Mapping):
            errors.append(f"paper table source report is missing: {name}")
            continue
        recorded_path = entry.get("path")
        try:
            recorded = Path(str(recorded_path))
            resolved_recorded = (
                recorded.resolve()
                if recorded.is_absolute()
                else (reports_root / recorded).resolve()
            )
        except (OSError, RuntimeError, ValueError):
            errors.append(f"paper table source path is invalid: {name}")
            continue
        if resolved_recorded != expected_path:
            errors.append(f"paper table source path mismatch: {name}")
        if not expected_path.is_file():
            errors.append(f"paper table source report is missing: {name}")
            continue
        if entry.get("sha256") != _sha256(expected_path):
            errors.append(f"paper table source report hash mismatch: {name}")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "manifest": manifest_path.as_posix(),
        "artifact_count": len(expected_artifacts),
        "source_report_count": len(EXPECTED_PAPER_SOURCE_REPORTS),
        "errors": errors,
    }


def audit_final_claim_bundle(
    reports_root: Path,
    *,
    final_audit_path: Path,
    paper_manifest_path: Path,
) -> dict[str, Any]:
    json_path = reports_root / "FINAL_CLAIM_EVIDENCE.json"
    markdown_path = reports_root / "FINAL_CLAIM_EVIDENCE.md"
    errors: list[str] = []
    if not json_path.is_file() or not markdown_path.is_file():
        return {
            "status": "missing",
            "valid": False,
            "errors": [
                "final claim evidence JSON/Markdown bundle is missing"
            ],
        }
    try:
        report = _read_json(json_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "valid": False,
            "errors": [
                f"final claim evidence cannot be parsed: {type(exc).__name__}"
            ],
        }
    if not isinstance(report, Mapping):
        return {
            "status": "invalid",
            "valid": False,
            "errors": ["final claim evidence must be a JSON object"],
        }
    if report.get("state") != "complete":
        errors.append("final claim evidence state is not complete")
    if not final_audit_path.is_file():
        errors.append("final evidence audit is missing")
    elif report.get("source_final_audit_sha256") != _sha256(
        final_audit_path
    ):
        errors.append("final claim evidence audit hash does not match")
    if not paper_manifest_path.is_file():
        errors.append("paper table manifest is missing")
    elif report.get(
        "source_paper_table_manifest_sha256"
    ) != _sha256(paper_manifest_path):
        errors.append("final claim evidence table hash does not match")
    expected_claims = {
        "formal_ablations",
        "C13_same_checkpoint_sensor_transfer",
        "C14_scene_strata",
        "C15_independent_sensor_retraining",
        "target_domain_system",
    }
    claims = report.get("claims", {})
    if set(claims) != expected_claims:
        errors.append("final claim evidence groups are incomplete")
    for name, claim in claims.items():
        if claim.get("status") != "evidence_complete":
            errors.append(f"final claim group is not complete: {name}")
    return {
        "status": "complete" if not errors else "invalid",
        "valid": not errors,
        "json": json_path.as_posix(),
        "markdown": markdown_path.as_posix(),
        "errors": errors,
    }


def patent_capabilities(
    final_sections: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    detector_validated = "yolo_seg" in final_sections
    scene_validated = "scene_pipeline" in final_sections
    return {
        "object_level_multimodal_fusion": {
            "status": "implemented_and_evaluated",
            "boundary": "object crop classification",
        },
        "target_domain_yolo_seg": {
            "status": "validated" if detector_validated else "queued",
            "boundary": "independent RGB detector; no MMFC head injection",
        },
        "two_stage_scene_pipeline": {
            "status": "validated" if scene_validated else "queued",
            "boundary": "detector proposals followed by MMFC reclassification",
        },
        "joint_multimodal_detector_head": {
            "status": "not_implemented",
            "boundary": "fused features are not injected into Detect/Segment heads",
        },
        "rgbd_input": {
            "status": "not_implemented",
            "boundary": "current image tensor is RGB only",
        },
        "synchronized_hardware_sensor_array": {
            "status": "not_implemented",
            "boundary": "current sidecars are image-conditioned soft sensors",
        },
        "human_small_target_occlusion_labels": {
            "status": "not_implemented",
            "boundary": "relative bbox quartiles and overlap are proxies only",
        },
    }


def audit_paper_readiness(
    *,
    reports_root: Path,
    allow_pending: bool,
    stable_manuscript_audit: Path | None = None,
) -> dict[str, Any]:
    reports_root = _resolve(reports_root)
    stable_manuscript_path = (
        reports_root / "MANUSCRIPT_STABLE_SECTIONS_AUDIT.json"
        if stable_manuscript_audit is None
        else _resolve(stable_manuscript_audit)
    )
    try:
        stable_manuscript_relative = stable_manuscript_path.relative_to(
            reports_root
        ).as_posix()
        stable_manuscript_markdown_relative = (
            stable_manuscript_path.with_suffix(".md")
            .relative_to(reports_root)
            .as_posix()
        )
    except ValueError as exc:
        raise ValueError(
            "stable manuscript audit must remain inside reports root"
        ) from exc
    errors: list[str] = []
    pending: list[str] = []
    requirements = {}
    for name, paths in FOUNDATIONAL_REQUIREMENTS.items():
        selected_paths = paths
        if name == "claim_governance":
            selected_paths = tuple(
                stable_manuscript_relative
                if path == "MANUSCRIPT_STABLE_SECTIONS_AUDIT.json"
                else stable_manuscript_markdown_relative
                if path == "MANUSCRIPT_STABLE_SECTIONS_AUDIT.md"
                else path
                for path in paths
            )
        requirements[name] = _artifact_requirement(
            reports_root,
            selected_paths,
        )
    for name, requirement in requirements.items():
        if requirement["status"] != "complete":
            errors.append(
                f"{name}: missing {len(requirement['missing'])} artifacts"
            )
    source_repair_transition = audit_source_image_repair_transition_bundle(
        reports_root / "SOURCE_IMAGE_REPAIR_TRANSITION.json",
        project_root=reports_root.parents[1],
    )
    requirements[
        "source_image_repair_transition_validation"
    ] = source_repair_transition
    if not source_repair_transition["valid"]:
        errors.extend(source_repair_transition["errors"])
        source_repair_report = None
    else:
        source_repair_report = _read_json(
            reports_root / "SOURCE_IMAGE_REPAIR_TRANSITION.json"
        )
    image_recovery = audit_image_recovery_bundle(
        reports_root / "IMAGE_READ_RECOVERY_AUDIT.json",
        project_root=reports_root.parents[1],
        historical_transition=source_repair_report,
    )
    requirements["image_recovery_validation"] = image_recovery
    if not image_recovery["valid"]:
        errors.extend(image_recovery["errors"])
    image_recovery_incident = audit_image_recovery_incident_bundle(
        reports_root / "IMAGE_READ_RECOVERY_INCIDENT.json"
    )
    requirements[
        "image_recovery_incident_validation"
    ] = image_recovery_incident
    if not image_recovery_incident["valid"]:
        errors.extend(image_recovery_incident["errors"])
    image_recovery_addendum = audit_image_recovery_addendum_bundle(
        reports_root / "IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.json",
        project_root=reports_root.parents[1],
    )
    requirements[
        "image_recovery_addendum_validation"
    ] = image_recovery_addendum
    if not image_recovery_addendum["valid"]:
        errors.extend(image_recovery_addendum["errors"])
    image_recovery_outcome = audit_image_recovery_outcome_bundle(
        reports_root / "IMAGE_READ_RECOVERY_OUTCOME.json",
        project_root=reports_root.parents[1],
    )
    requirements[
        "image_recovery_outcome_validation"
    ] = image_recovery_outcome
    if not image_recovery_outcome["valid"]:
        errors.extend(image_recovery_outcome["errors"])
    formal_queue_interruption = audit_formal_queue_interruption_bundle(
        reports_root / "FORMAL_QUEUE_INTERRUPTION.json"
    )
    requirements[
        "formal_queue_interruption_validation"
    ] = formal_queue_interruption
    if not formal_queue_interruption["valid"]:
        errors.extend(formal_queue_interruption["errors"])
    formal_queue_interruption_addendum = (
        audit_formal_queue_interruption_addendum_bundle(
            reports_root / "FORMAL_QUEUE_INTERRUPTION_ADDENDUM.json",
            project_root=reports_root.parents[1],
        )
    )
    requirements[
        "formal_queue_interruption_addendum_validation"
    ] = formal_queue_interruption_addendum
    if not formal_queue_interruption_addendum["valid"]:
        errors.extend(formal_queue_interruption_addendum["errors"])
    formal_queue_interruption_outcome = (
        audit_formal_queue_interruption_outcome_bundle(
            reports_root / "FORMAL_QUEUE_INTERRUPTION_OUTCOME.json",
            project_root=reports_root.parents[1],
        )
    )
    requirements[
        "formal_queue_interruption_outcome_validation"
    ] = formal_queue_interruption_outcome
    if not formal_queue_interruption_outcome["valid"]:
        errors.extend(formal_queue_interruption_outcome["errors"])
    yolo_dependency_incident = audit_yolo_seg_dependency_incident_bundle(
        reports_root / "YOLO_SEG_DEPENDENCY_INCIDENT.json",
        reports_root / "YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.json",
        project_root=reports_root.parents[1],
    )
    requirements[
        "yolo_seg_dependency_incident_validation"
    ] = yolo_dependency_incident
    if not yolo_dependency_incident["valid"]:
        errors.extend(yolo_dependency_incident["errors"])
    yolo_dependency_outcome = audit_yolo_seg_dependency_outcome_bundle(
        reports_root / "YOLO_SEG_DEPENDENCY_OUTCOME.json",
        project_root=reports_root.parents[1],
    )
    requirements[
        "yolo_seg_dependency_outcome_validation"
    ] = yolo_dependency_outcome
    if not yolo_dependency_outcome["valid"]:
        errors.extend(yolo_dependency_outcome["errors"])
    scene_protocol_addenda = audit_scene_protocol_addenda_bundle(
        (
            reports_root
            / "yolo-seg"
            / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
        ),
        (
            reports_root
            / "yolo-seg"
            / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
        ),
        project_root=reports_root.parents[1],
    )
    requirements[
        "scene_protocol_addenda_validation"
    ] = scene_protocol_addenda
    if not scene_protocol_addenda["valid"]:
        errors.extend(scene_protocol_addenda["errors"])
    scene_strata_reporting_recovery = (
        audit_scene_strata_reporting_recovery_bundle(
            (
                reports_root
                / "yolo-seg"
                / "SCENE_STRATA_REPORTING_INCIDENT.json"
            ),
            (
                reports_root
                / "yolo-seg"
                / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
            ),
            (
                reports_root
                / "yolo-seg"
                / "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json"
            ),
            project_root=reports_root.parents[1],
        )
    )
    requirements[
        "scene_strata_reporting_recovery_validation"
    ] = scene_strata_reporting_recovery
    if not scene_strata_reporting_recovery["valid"]:
        errors.extend(scene_strata_reporting_recovery["errors"])
    sensor_retraining_exit = audit_sensor_retraining_exit_bundle(
        reports_root / "SENSOR_RETRAINING_EXIT_INCIDENT.json",
        reports_root / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM.json",
        reports_root / "SENSOR_RETRAINING_EXIT_INCIDENT_ADDENDUM_2.json",
        reports_root / "SENSOR_RETRAINING_LOG_ENCODING_REPAIR.json",
        reports_root / "SENSOR_RETRAINING_EXIT_OUTCOME.json",
        project_root=reports_root.parents[1],
    )
    requirements[
        "sensor_retraining_exit_validation"
    ] = sensor_retraining_exit
    if not sensor_retraining_exit["valid"]:
        errors.extend(sensor_retraining_exit["errors"])
    source_image_integrity = audit_source_image_integrity_bundle(
        reports_root / "SOURCE_IMAGE_INTEGRITY_AUDIT.json",
        project_root=reports_root.parents[1],
        historical_transition=source_repair_report,
    )
    requirements[
        "source_image_integrity_validation"
    ] = source_image_integrity
    if not source_image_integrity["valid"]:
        errors.extend(source_image_integrity["errors"])
    current_source_image_integrity = audit_source_image_integrity_bundle(
        reports_root / "SOURCE_IMAGE_INTEGRITY_CURRENT.json",
        project_root=reports_root.parents[1],
        expected_recovery_state="repaired",
    )
    requirements[
        "current_source_image_integrity_validation"
    ] = current_source_image_integrity
    if not current_source_image_integrity["valid"]:
        errors.extend(current_source_image_integrity["errors"])
    method_contract = audit_method_contract_bundle(
        reports_root / "METHOD_CONTRACT_AUDIT.json",
        project_root=reports_root.parents[1],
    )
    requirements["method_contract_validation"] = method_contract
    if not method_contract["valid"]:
        errors.extend(method_contract["errors"])
    uncertainty_floor = audit_uncertainty_floor_bundle(
        reports_root / "UNCERTAINTY_FLOOR_AUDIT.json",
        project_root=reports_root.parents[1],
    )
    requirements["uncertainty_floor_validation"] = uncertainty_floor
    if not uncertainty_floor["valid"]:
        errors.extend(uncertainty_floor["errors"])
    latent_uncertainty = audit_latent_uncertainty_bundle(
        (
            reports_root
            / "full_cross_modal_quality"
            / "uncertainty"
            / "uncertainty_diagnostics.json"
        ),
        project_root=reports_root.parents[1],
    )
    requirements["latent_uncertainty_validation"] = latent_uncertainty
    if not latent_uncertainty["valid"]:
        errors.extend(latent_uncertainty["errors"])
    v2b_latent_uncertainty = audit_latent_uncertainty_bundle(
        (
            reports_root
            / "full_quality_aware_prior"
            / "uncertainty"
            / "uncertainty_diagnostics.json"
        ),
        project_root=reports_root.parents[1],
        expected_sources=EXPECTED_V2B_LATENT_UNCERTAINTY_SOURCES,
        expected_figure=(
            "reports/qwen-material/full_quality_aware_prior/"
            "uncertainty/uncertainty_diagnostics.png"
        ),
    )
    requirements[
        "v2b_latent_uncertainty_validation"
    ] = v2b_latent_uncertainty
    if not v2b_latent_uncertainty["valid"]:
        errors.extend(v2b_latent_uncertainty["errors"])
    uncertainty_comparison = audit_uncertainty_comparison_bundle(
        reports_root / "UNCERTAINTY_DIAGNOSTICS_COMPARISON.json",
        reports_root=reports_root,
    )
    requirements[
        "uncertainty_comparison_validation"
    ] = uncertainty_comparison
    if not uncertainty_comparison["valid"]:
        errors.extend(uncertainty_comparison["errors"])
    stable_manuscript = audit_manuscript_stable_sections_bundle(
        stable_manuscript_path,
        project_root=reports_root.parents[1],
    )
    requirements["stable_manuscript_validation"] = stable_manuscript
    if not stable_manuscript["valid"]:
        errors.extend(stable_manuscript["errors"])
    bibliography = audit_bibliography_bundle(
        reports_root / "BIBLIOGRAPHY_AUDIT.json",
        project_root=reports_root.parents[1],
    )
    requirements["bibliography_validation"] = bibliography
    if not bibliography["valid"]:
        errors.extend(bibliography["errors"])
    variant_lineage = audit_variant_lineage_bundle(
        reports_root / "VARIANT_LINEAGE_AUDIT.json",
        project_root=reports_root.parents[1],
    )
    requirements["variant_lineage_validation"] = variant_lineage
    if not variant_lineage["valid"]:
        errors.extend(variant_lineage["errors"])
    error_analysis = audit_error_analysis_bundle(
        reports_root / "ERROR_ANALYSIS.json",
        project_root=reports_root.parents[1],
    )
    requirements["error_analysis_validation"] = error_analysis
    if not error_analysis["valid"]:
        errors.extend(error_analysis["errors"])
    gate_error_analysis = audit_error_analysis_bundle(
        reports_root / "GATE_ERROR_ANALYSIS.json",
        project_root=reports_root.parents[1],
        expected_sources=EXPECTED_GATE_ERROR_ANALYSIS_SOURCES,
        reference_name="gate_disabled",
        candidate_name="gate_enabled",
    )
    requirements["gate_error_analysis_validation"] = gate_error_analysis
    if not gate_error_analysis["valid"]:
        errors.extend(gate_error_analysis["errors"])
    class_metric_intervals = audit_class_metric_intervals_bundle(
        reports_root / "CLASS_METRIC_INTERVALS.json",
        project_root=reports_root.parents[1],
    )
    requirements[
        "class_metric_intervals_validation"
    ] = class_metric_intervals
    if not class_metric_intervals["valid"]:
        errors.extend(class_metric_intervals["errors"])

    evidence_path = _select_evidence_audit_path(
        reports_root,
        allow_pending=allow_pending,
    )
    if not evidence_path.is_file():
        errors.append(f"evidence audit is missing: {evidence_path}")
        evidence: dict[str, Any] = {}
    else:
        try:
            evidence = _read_json(evidence_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                f"evidence audit cannot be parsed: {type(exc).__name__}"
            )
            evidence = {}
        if not isinstance(evidence, Mapping):
            errors.append("evidence audit must be a JSON object")
            evidence = {}
    evidence_complete = (
        evidence.get("state") == "complete"
        and evidence.get("valid") is True
        and evidence.get("error_count") == 0
    )
    if not evidence_complete:
        message = (
            "final evidence audit is not complete"
            f" (state={evidence.get('state')!r})"
        )
        if allow_pending and evidence.get("state") == "pending":
            pending.append(message)
        else:
            errors.append(message)

    sections = evidence.get("sections", {})
    experiment_sections = {
        "formal_retrained_ablations": "formal_ablations",
        "target_domain_yolo_seg": "yolo_seg",
        "integrated_scene_pipeline": "scene_pipeline",
        "scene_stratified_diagnostics": "scene_strata",
        "same_checkpoint_sensor_transfer": "sensor_source_transfer",
        "independent_sensor_source_retraining": (
            "sensor_source_retraining"
        ),
    }
    for requirement_name, section_name in experiment_sections.items():
        if section_name in sections:
            requirements[requirement_name] = {
                "status": "complete",
                "evidence_section": section_name,
            }
        else:
            requirements[requirement_name] = {
                "status": "pending",
                "evidence_section": section_name,
            }
            pending.append(
                f"{requirement_name}: final evidence section is pending"
            )

    final_path = reports_root / "FINAL_EVIDENCE_AUDIT.json"
    table_root = reports_root / "paper-tables"
    if table_root.is_dir():
        table_audit = audit_paper_table_bundle(
            table_root,
            final_audit_path=final_path,
        )
        if not table_audit["valid"]:
            errors.extend(table_audit["errors"])
    else:
        table_audit = {
            "status": "pending",
            "valid": False,
            "errors": [],
        }
        pending.append("paper table bundle is pending")
    requirements["paper_table_bundle"] = table_audit
    if evidence_complete and table_audit["valid"]:
        claim_audit = audit_final_claim_bundle(
            reports_root,
            final_audit_path=final_path,
            paper_manifest_path=table_root / "manifest.json",
        )
        if not claim_audit["valid"]:
            errors.extend(claim_audit["errors"])
    else:
        claim_audit = {
            "status": "pending",
            "valid": False,
            "errors": [],
        }
        pending.append("final dynamic claim evidence is pending")
    requirements["final_dynamic_claim_evidence"] = claim_audit

    capabilities = patent_capabilities(sections)
    full_patent_ready = all(
        value["status"] in {"implemented_and_evaluated", "validated"}
        for value in capabilities.values()
    )
    internal_paper_ready = (
        not errors
        and evidence_complete
        and table_audit["valid"]
        and all(
            value.get("status") == "complete"
            for value in requirements.values()
        )
    )
    state = (
        "complete"
        if internal_paper_ready
        else ("invalid" if errors else "pending")
    )
    warnings = [
        "Internal paper readiness does not mean external blind validation.",
        "Qwen and rule-v1 sidecars are not synchronized hardware observations.",
        "Target-domain YOLO-seg remains separate from the MMFC feature head.",
        "Full patent reproduction remains false without RGB-D, hardware sensors, and joint detector-head injection.",
        "Relative-size and bbox-overlap strata are not human small-target or occlusion labels.",
    ]
    return {
        "schema_version": 1,
        "state": state,
        "valid": state != "invalid",
        "internal_paper_evidence_ready": internal_paper_ready,
        "full_patent_reproduction_ready": full_patent_ready,
        "external_deployment_claim_ready": False,
        "error_count": len(errors),
        "pending_count": len(pending),
        "warning_count": len(warnings),
        "errors": errors,
        "pending": pending,
        "warnings": warnings,
        "source_evidence_audit": evidence_path.as_posix(),
        "source_evidence_audit_sha256": (
            _sha256(evidence_path) if evidence_path.is_file() else None
        ),
        "requirements": requirements,
        "patent_capabilities": capabilities,
    }


def main() -> None:
    args = parse_args()
    report = audit_paper_readiness(
        reports_root=args.reports,
        allow_pending=args.allow_pending,
        stable_manuscript_audit=args.stable_manuscript_audit,
    )
    output = args.output
    if output is None:
        filename = (
            "PAPER_READINESS_AUDIT.pending.json"
            if args.allow_pending
            else "PAPER_READINESS_AUDIT.json"
        )
        output = _resolve(args.reports) / filename
    else:
        output = _resolve(output)
    _atomic_write(output, report)
    print(
        json.dumps(
            {"output": output.as_posix(), **report},
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["state"] == "invalid":
        raise SystemExit(1)
    if not args.allow_pending and report["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
