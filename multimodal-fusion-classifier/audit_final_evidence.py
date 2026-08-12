"""Audit the completed paper/patent evidence bundle for internal consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
FORMAL_EXPERIMENTS = (
    "vision_only",
    "vision_text",
    "vision_sensor",
    "trimodal_mean",
    "trimodal_attention_no_reliability",
    "full_no_degradation",
    "full_no_reconstruction",
    "full_no_calibration",
)
ALL_ABLATIONS = ("full", *FORMAL_EXPERIMENTS)
SCENE_SOURCES = ("detector", "missing", "rule-v1")
SCENE_COMPARISONS = (
    "missing_vs_detector",
    "rule_v1_vs_detector",
    "rule_v1_vs_missing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all final experiment evidence before paper use"
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=PROJECT_ROOT / "reports" / "qwen-material",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Report unfinished background stages as pending instead of errors",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    resolved_root = root.resolve()
    for path in sorted(
        (item.resolve() for item in root.rglob("*.txt")),
        key=str,
    ):
        relative = path.relative_to(resolved_root).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        file_size = path.stat().st_size
        size_bytes += file_size
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        file_count += 1
    return {
        "file_count": file_count,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _expect_equal(
    errors: list[str],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        errors.append(f"{name}: expected {expected!r}, got {actual!r}")


def _expect_probability(
    errors: list[str],
    name: str,
    value: Any,
) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name}: expected numeric probability, got {value!r}")
        return
    if not 0.0 <= number <= 1.0:
        errors.append(f"{name}: probability is outside [0,1]: {number}")


def _audit_protocol_input_hashes(
    protocol: Mapping[str, Any],
    *,
    prefix: str,
    errors: list[str],
    expected_hash_overrides: Mapping[str, str] | None = None,
) -> int:
    checked = 0
    for name, entry in protocol.get("inputs", {}).items():
        if not isinstance(entry, Mapping) or "sha256" not in entry:
            continue
        path = Path(str(entry.get("path", "")))
        path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            errors.append(f"{prefix}.{name}: protocol input is missing: {path}")
            continue
        actual = _sha256(path)
        expected = str(
            (expected_hash_overrides or {}).get(name, entry["sha256"])
        ).lower()
        if actual != expected:
            errors.append(
                f"{prefix}.{name}: protocol input hash mismatch: "
                f"{actual} != {expected}"
            )
        checked += 1
    return checked


def _audit_file_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    errors: list[str],
    expected_hash_overrides: Mapping[str, str] | None = None,
) -> int:
    checked = 0
    for index, entry in enumerate(entries):
        name = f"{prefix}[{index}]"
        path = Path(str(entry.get("path", "")))
        path = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            errors.append(f"{name}: file is missing: {path}")
            continue
        if "size_bytes" in entry and path.stat().st_size != int(entry["size_bytes"]):
            errors.append(
                f"{name}: size mismatch: {path.stat().st_size} != "
                f"{entry['size_bytes']}"
            )
        actual = _sha256(path)
        key = Path(str(entry.get("path", ""))).as_posix()
        expected = str(
            (expected_hash_overrides or {}).get(
                key, entry.get("sha256", "")
            )
        ).lower()
        if actual != expected:
            errors.append(f"{name}: hash mismatch: {actual} != {expected}")
        checked += 1
    return checked


def _dependency_state(
    path: Path,
    name: str,
    *,
    allow_pending: bool,
    errors: list[str],
    pending: list[str],
    defer_failed: bool = False,
    defer_reason: str = "",
) -> bool:
    if not path.is_file():
        message = f"{name}: status file is missing: {path}"
        (pending if allow_pending else errors).append(message)
        return False
    document = _read_json(path)
    state = document.get("state")
    if state == "complete":
        return True
    if state == "failed":
        if allow_pending and defer_failed:
            pending.append(
                f"{name}: prior failed state is superseded while "
                f"{defer_reason} is active"
            )
            return False
        errors.append(f"{name}: failed: {document.get('message', '')}")
        return False
    message = f"{name}: current state is {state!r}"
    (pending if allow_pending else errors).append(message)
    return False


def _status_is_active(state: Any) -> bool:
    return state not in (None, "complete", "failed")


def _audit_dataset(
    reports_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    stats_path = PROJECT_ROOT / "data" / "qwen-material" / "stats.json"
    leakage_path = reports_root / "DATA_LEAKAGE_AUDIT.json"
    if not stats_path.is_file():
        errors.append(f"dataset stats are missing: {stats_path}")
        return {}
    if not leakage_path.is_file():
        errors.append(f"dataset leakage audit is missing: {leakage_path}")
        return {}

    stats = _read_json(stats_path)
    leakage = _read_json(leakage_path)
    expected_stats = {
        "samples": 8654,
        "sensor_records": 8654,
        "source_images": 5990,
        "qwen_success_records": 8639,
        "fallback_sensor_records": 15,
        "split_counts": {"train": 7002, "val": 826, "test": 826},
    }
    for key, expected in expected_stats.items():
        _expect_equal(errors, f"dataset.{key}", stats.get(key), expected)
    expected_leakage = {
        "sample_count": 8654,
        "source_image_count": 5990,
        "unique_annotated_object_count": 8646,
        "exact_object_duplicate_group_count": 8,
        "exact_object_duplicate_record_count": 8,
        "cross_split_exact_duplicate_group_count": 0,
        "high_confidence_pair_count": 24,
    }
    for key, expected in expected_leakage.items():
        _expect_equal(errors, f"leakage.{key}", leakage.get(key), expected)
    return {
        "sample_count": stats.get("samples"),
        "source_image_count": stats.get("source_images"),
        "qwen_success_count": stats.get("qwen_success_records"),
        "fallback_count": stats.get("fallback_sensor_records"),
        "unique_object_count": leakage.get("unique_annotated_object_count"),
        "test_object_count": stats.get("split_counts", {}).get("test"),
        "test_unique_object_count": 825,
        "near_duplicate_pair_count": leakage.get("high_confidence_pair_count"),
    }


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def _audit_ablation_evaluation(
    name: str,
    path: Path,
    expected_classes: Sequence[str] | None,
    errors: list[str],
) -> list[str] | None:
    if not path.is_file():
        errors.append(f"ablation {name}: evaluation is missing: {path}")
        return expected_classes if expected_classes is not None else None
    report = _read_json(path)
    metrics = report.get("metrics", {})
    dataset = report.get("dataset", {})
    evaluation = report.get("evaluation_config", {})
    classes = [str(value) for value in report.get("class_names", [])]
    _expect_equal(errors, f"ablation {name}.sample_count", metrics.get("sample_count"), 826)
    _expect_equal(errors, f"ablation {name}.dataset.split", dataset.get("split"), "test")
    _expect_equal(
        errors,
        f"ablation {name}.dataset.evaluated_samples",
        dataset.get("evaluated_samples"),
        826,
    )
    _expect_equal(
        errors,
        f"ablation {name}.bootstrap_repeats",
        evaluation.get("bootstrap_repeats"),
        2000,
    )
    _expect_equal(errors, f"ablation {name}.class_count", len(classes), 19)
    if expected_classes is not None:
        _expect_equal(errors, f"ablation {name}.class_order", classes, list(expected_classes))
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece"):
        _expect_probability(errors, f"ablation {name}.{metric}", metrics.get(metric))
    predictions_path = path.parent / "predictions.jsonl"
    if not predictions_path.is_file():
        errors.append(f"ablation {name}: predictions are missing: {predictions_path}")
    else:
        _expect_equal(
            errors,
            f"ablation {name}.prediction_count",
            _count_jsonl(predictions_path),
            826,
        )
    return classes


def _audit_formal_ablations(
    reports_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    ablation_root = reports_root / "ablations"
    manifest_path = ablation_root / "FORMAL_RUN_MANIFEST.json"
    manifest_hash_count = 0
    if not manifest_path.is_file():
        errors.append(f"formal ablation run manifest is missing: {manifest_path}")
    else:
        manifest = _read_json(manifest_path)
        _expect_equal(
            errors,
            "formal_manifest.role",
            manifest.get("manifest_role"),
            "mid_queue_identity_and_recovery_snapshot",
        )
        _expect_equal(
            errors,
            "formal_manifest.experiments",
            set(manifest.get("scope", {}).get("experiments", [])),
            set(FORMAL_EXPERIMENTS),
        )
        for key, expected_count in (
            ("training_sources", 7),
            ("configs", 8),
            ("data", 4),
            ("orchestration_sources_after_recovery", 2),
        ):
            entries = manifest.get(key, [])
            _expect_equal(
                errors,
                f"formal_manifest.{key}.count",
                len(entries),
                expected_count,
            )
            manifest_hash_count += _audit_file_entries(
                entries,
                prefix=f"formal_manifest.{key}",
                errors=errors,
            )
        manifest_hash_count += _audit_file_entries(
            [manifest.get("initial_weights", {})],
            prefix="formal_manifest.initial_weights",
            errors=errors,
        )
    expected_classes: list[str] | None = None
    metrics: dict[str, dict[str, float]] = {}
    for name in ALL_ABLATIONS:
        evaluation_path = ablation_root / name / "evaluation.json"
        expected_classes = _audit_ablation_evaluation(
            name,
            evaluation_path,
            expected_classes,
            errors,
        )
        if evaluation_path.is_file():
            report = _read_json(evaluation_path)
            values = report.get("metrics", {})
            metrics[name] = {
                "accuracy": float(values.get("accuracy", 0.0)),
                "macro_f1": float(values.get("macro_f1", 0.0)),
                "ece": float(values.get("ece", 0.0)),
            }
        if name != "full":
            audit_path = ablation_root / name / "CHECKPOINT_AUDIT.json"
            if not audit_path.is_file():
                errors.append(f"ablation {name}: checkpoint audit is missing")
            else:
                audit = _read_json(audit_path)
                _expect_equal(errors, f"ablation {name}.checkpoint_valid", audit.get("valid"), True)
                _expect_equal(
                    errors,
                    f"ablation {name}.completed_epochs",
                    audit.get("training_summary_completed_epochs"),
                    100,
                )

    summary_path = ablation_root / "ablation_summary.json"
    statistics_path = ablation_root / "ablation_pairwise_statistics.json"
    if not summary_path.is_file():
        errors.append(f"formal ablation summary is missing: {summary_path}")
    else:
        summary = _read_json(summary_path)
        _expect_equal(errors, "ablation_summary.protocol", summary.get("protocol"), "formal_retrained")
        _expect_equal(
            errors,
            "ablation_summary.experiments",
            set(summary.get("experiment_order", [])),
            set(ALL_ABLATIONS),
        )
        _expect_equal(errors, "ablation_summary.row_count", len(summary.get("summary", [])), 9)
    if not statistics_path.is_file():
        errors.append(f"ablation pairwise statistics are missing: {statistics_path}")
    else:
        statistics = _read_json(statistics_path)
        rows = statistics.get("summary", [])
        _expect_equal(errors, "ablation_statistics.reference", statistics.get("reference"), "full")
        _expect_equal(errors, "ablation_statistics.comparison_count", statistics.get("comparison_count"), 8)
        _expect_equal(
            errors,
            "ablation_statistics.experiments",
            {row.get("experiment") for row in rows},
            set(FORMAL_EXPERIMENTS),
        )
        for row in rows:
            _expect_probability(
                errors,
                f"ablation_statistics.{row.get('experiment')}.holm_p",
                row.get("mcnemar_holm_p_value"),
            )
    return {
        "experiment_count": len(metrics),
        "comparison_count": len(FORMAL_EXPERIMENTS),
        "manifest_hash_count": manifest_hash_count,
        "metrics": metrics,
    }


def _audit_yolo_seg(
    reports_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    report_path = reports_root / "yolo-seg" / "evaluation.json"
    if not report_path.is_file():
        errors.append(f"YOLO-seg evaluation is missing: {report_path}")
        return {}
    report = _read_json(report_path)
    dataset = report.get("dataset", {})
    metrics = report.get("metrics", {})
    _expect_equal(errors, "yolo.protocol", report.get("protocol"), "held_out_source_grouped_test")
    _expect_equal(errors, "yolo.test_source_images", dataset.get("test_source_images"), 570)
    _expect_equal(errors, "yolo.test_objects", dataset.get("test_objects"), 825)
    _expect_equal(errors, "yolo.class_count", dataset.get("class_count"), 19)
    _expect_equal(errors, "yolo.source_grouped_split", dataset.get("source_grouped_split"), True)
    _expect_equal(errors, "yolo.copied_image_count", dataset.get("copied_image_count"), 0)
    required_metrics = (
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
    )
    for key in required_metrics:
        _expect_probability(errors, f"yolo.{key}", metrics.get(key))
    checkpoint = report.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if not checkpoint_path.is_file():
        errors.append(f"YOLO-seg best checkpoint is missing: {checkpoint_path}")
    elif checkpoint.get("sha256") != _sha256(checkpoint_path):
        errors.append("YOLO-seg best checkpoint SHA-256 does not match evaluation")
    return {
        "test_source_image_count": dataset.get("test_source_images"),
        "test_object_count": dataset.get("test_objects"),
        "metrics": metrics,
        "checkpoint_sha256": checkpoint.get("sha256"),
    }


def _audit_scene_pipeline(
    reports_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    protocol_path = reports_root / "yolo-seg" / "INTEGRATED_EVALUATION_PROTOCOL.json"
    report_path = reports_root / "yolo-seg" / "integrated" / "evaluation.json"
    if not report_path.is_file():
        errors.append(f"integrated scene evaluation is missing: {report_path}")
        return {}
    if not protocol_path.is_file():
        errors.append(f"integrated evaluation protocol is missing: {protocol_path}")
        return {}
    report = _read_json(report_path)
    protocol = _read_json(protocol_path)
    addendum_path = (
        reports_root
        / "yolo-seg"
        / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
    )
    if not addendum_path.is_file():
        errors.append(
            f"integrated protocol addendum is missing: {addendum_path}"
        )
        addendum = {}
        revised_source_hash = ""
    else:
        addendum = _read_json(addendum_path)
        revised_source = addendum.get("implementation_fix", {}).get(
            "revised_evaluation_source", {}
        )
        revised_source_hash = str(revised_source.get("sha256", ""))
        _expect_equal(
            errors,
            "scene.addendum_status",
            addendum.get("status"),
            "frozen_after_path_identity_failure_before_recovery_rerun",
        )
        _expect_equal(
            errors,
            "scene.addendum_valid",
            addendum.get("valid"),
            True,
        )
        _expect_equal(
            errors,
            "scene.addendum_protocol_sha256",
            addendum.get("original_protocol", {}).get("sha256"),
            _sha256(protocol_path),
        )
        _expect_equal(
            errors,
            "scene.addendum_original_source_sha256",
            addendum.get("implementation_fix", {})
            .get("original_evaluation_source", {})
            .get("sha256"),
            protocol.get("inputs", {})
            .get("evaluation_source", {})
            .get("sha256"),
        )
        revised_path = Path(str(revised_source.get("path", "")))
        revised_path = (
            revised_path.resolve()
            if revised_path.is_absolute()
            else (PROJECT_ROOT / revised_path).resolve()
        )
        if not revised_path.is_file():
            errors.append(
                f"scene.addendum revised source is missing: {revised_path}"
            )
        else:
            _expect_equal(
                errors,
                "scene.addendum_revised_source_sha256",
                revised_source_hash,
                _sha256(revised_path),
            )
    protocol_input_count = _audit_protocol_input_hashes(
        protocol,
        prefix="scene.inputs",
        errors=errors,
        expected_hash_overrides={
            "evaluation_source": revised_source_hash
        },
    )
    frozen = report.get("frozen_protocol", {})
    dataset = report.get("dataset", {})
    _expect_equal(errors, "scene.protocol", report.get("protocol"), "fixed_source_grouped_scene_test")
    _expect_equal(errors, "scene.protocol_status", protocol.get("status"), "frozen_before_detector_training")
    _expect_equal(errors, "scene.protocol_sha256", frozen.get("sha256"), _sha256(protocol_path))
    _expect_equal(
        errors,
        "scene.protocol_addendum_sha256",
        frozen.get("addendum", {}).get("sha256"),
        _sha256(addendum_path) if addendum_path.is_file() else None,
    )
    _expect_equal(errors, "scene.source_image_count", dataset.get("source_image_count"), 570)
    _expect_equal(errors, "scene.ground_truth_object_count", dataset.get("ground_truth_object_count"), 825)

    scenarios = report.get("scenarios", {})
    _expect_equal(errors, "scene.thresholds", set(scenarios), {"0.25", "0.80"})
    for threshold, sources in scenarios.items():
        _expect_equal(errors, f"scene.{threshold}.sources", set(sources), set(SCENE_SOURCES))
        for source, metrics in sources.items():
            _expect_equal(
                errors,
                f"scene.{threshold}.{source}.source_images",
                metrics.get("source_image_count"),
                570,
            )
            _expect_equal(
                errors,
                f"scene.{threshold}.{source}.ground_truth",
                metrics.get("ground_truth_count"),
                825,
            )
            for metric in (
                "localization_precision",
                "localization_recall",
                "classification_accuracy_on_localized",
                "end_to_end_precision",
                "end_to_end_recall",
                "macro_precision",
                "macro_recall",
                "macro_f1",
            ):
                _expect_probability(
                    errors,
                    f"scene.{threshold}.{source}.{metric}",
                    metrics.get(metric),
                )
    comparisons = report.get("paired_comparisons", {})
    _expect_equal(errors, "scene.comparison_thresholds", set(comparisons), {"0.25", "0.80"})
    for threshold, values in comparisons.items():
        _expect_equal(
            errors,
            f"scene.{threshold}.comparisons",
            set(values),
            set(SCENE_COMPARISONS),
        )
    return {
        "test_source_image_count": dataset.get("source_image_count"),
        "test_object_count": dataset.get("ground_truth_object_count"),
        "protocol_input_hash_count": protocol_input_count,
        "protocol_addendum_sha256": (
            _sha256(addendum_path) if addendum_path.is_file() else None
        ),
        "scenarios": scenarios,
        "paired_comparisons": comparisons,
    }


def _audit_sensor_source_transfer(
    reports_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    root = reports_root / "sensor-source-transfer"
    protocol_path = root / "PROTOCOL.json"
    report_path = root / "evaluation.json"
    if not protocol_path.is_file():
        errors.append(f"sensor-source protocol is missing: {protocol_path}")
        return {}
    if not report_path.is_file():
        errors.append(f"sensor-source evaluation is missing: {report_path}")
        return {}
    protocol = _read_json(protocol_path)
    report = _read_json(report_path)
    protocol_input_count = _audit_protocol_input_hashes(
        protocol,
        prefix="sensor_source.inputs",
        errors=errors,
    )
    frozen = report.get("frozen_protocol", {})
    _expect_equal(
        errors,
        "sensor_source.protocol_status",
        protocol.get("status"),
        "frozen_before_source_transfer_test",
    )
    _expect_equal(
        errors,
        "sensor_source.protocol",
        report.get("protocol"),
        "same_checkpoint_sensor_source_transfer",
    )
    _expect_equal(
        errors,
        "sensor_source.protocol_sha256",
        frozen.get("sha256"),
        _sha256(protocol_path),
    )
    _expect_equal(
        errors,
        "sensor_source.sample_count",
        report.get("sample_count"),
        826,
    )
    _expect_equal(
        errors,
        "sensor_source.source_image_count",
        report.get("source_image_count"),
        570,
    )
    sources = report.get("sources", {})
    _expect_equal(
        errors,
        "sensor_source.conditions",
        set(sources),
        {"qwen", "rule-v1", "missing"},
    )
    for source, values in sources.items():
        metrics = values.get("metrics", {})
        for metric in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "ece",
        ):
            _expect_probability(
                errors,
                f"sensor_source.{source}.{metric}",
                metrics.get(metric),
            )
        predictions = root / source / "predictions.jsonl"
        if not predictions.is_file():
            errors.append(
                f"sensor-source {source}: predictions are missing: {predictions}"
            )
        else:
            _expect_equal(
                errors,
                f"sensor_source.{source}.prediction_count",
                _count_jsonl(predictions),
                826,
            )
    comparisons = report.get("paired_comparisons", {})
    _expect_equal(
        errors,
        "sensor_source.comparisons",
        set(comparisons),
        {
            "rule_v1_vs_qwen",
            "missing_vs_qwen",
            "rule_v1_vs_missing",
        },
    )
    return {
        "sample_count": report.get("sample_count"),
        "source_image_count": report.get("source_image_count"),
        "protocol_input_hash_count": protocol_input_count,
        "sources": sources,
        "paired_comparisons": comparisons,
    }


def _audit_sensor_source_retraining(
    reports_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    root = reports_root / "sensor-source-retraining"
    protocol_path = root / "PROTOCOL.json"
    config_path = root / "resolved_rule_v1_config.json"
    candidate_root = root / "rule-v1"
    evaluation_path = candidate_root / "test" / "evaluation.json"
    audit_path = candidate_root / "CHECKPOINT_AUDIT.json"
    comparison_path = (
        root / "RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json"
    )
    if not protocol_path.is_file():
        errors.append(
            f"sensor-source retraining protocol is missing: {protocol_path}"
        )
        return {}
    protocol = _read_json(protocol_path)
    input_count = _audit_file_entries(
        protocol.get("input_hashes", []),
        prefix="sensor_retraining.inputs",
        errors=errors,
    )
    _expect_equal(
        errors,
        "sensor_retraining.protocol_status",
        protocol.get("status"),
        "frozen_before_rule_v1_retraining",
    )
    if not config_path.is_file():
        errors.append(
            f"sensor-source resolved config is missing: {config_path}"
        )
    else:
        _expect_equal(
            errors,
            "sensor_retraining.resolved_config_sha256",
            _sha256(config_path),
            protocol.get("resolved_config_sha256"),
        )

    reference_evaluation_path = (
        reports_root
        / "full_cross_modal_quality"
        / "test"
        / "evaluation.json"
    )
    expected_classes = None
    if reference_evaluation_path.is_file():
        expected_classes = [
            str(value)
            for value in _read_json(reference_evaluation_path).get(
                "class_names",
                [],
            )
        ]
    _audit_ablation_evaluation(
        "sensor_source_rule_v1_retrained",
        evaluation_path,
        expected_classes,
        errors,
    )

    checkpoint_summary: dict[str, Any] = {}
    if not audit_path.is_file():
        errors.append(
            f"sensor-source checkpoint audit is missing: {audit_path}"
        )
    else:
        checkpoint_audit = _read_json(audit_path)
        _expect_equal(
            errors,
            "sensor_retraining.checkpoint_valid",
            checkpoint_audit.get("valid"),
            True,
        )
        _expect_equal(
            errors,
            "sensor_retraining.completed_epochs",
            checkpoint_audit.get("training_summary_completed_epochs"),
            100,
        )
        checkpoint_summary = checkpoint_audit.get("best", {})
        checkpoint_path = Path(
            str(checkpoint_summary.get("path", ""))
        )
        if not checkpoint_path.is_file():
            errors.append(
                f"sensor-source best checkpoint is missing: {checkpoint_path}"
            )
        else:
            _expect_equal(
                errors,
                "sensor_retraining.best_sha256",
                _sha256(checkpoint_path),
                checkpoint_summary.get("sha256"),
            )
            if evaluation_path.is_file():
                evaluation_checkpoint = _read_json(
                    evaluation_path
                ).get("checkpoint", {})
                _expect_equal(
                    errors,
                    "sensor_retraining.evaluation_checkpoint_path",
                    Path(
                        str(evaluation_checkpoint.get("path", ""))
                    ).resolve(),
                    checkpoint_path.resolve(),
                )
                _expect_equal(
                    errors,
                    "sensor_retraining.evaluation_checkpoint_size",
                    evaluation_checkpoint.get("size_bytes"),
                    checkpoint_summary.get("size_bytes"),
                )
                _expect_equal(
                    errors,
                    "sensor_retraining.evaluation_checkpoint_epoch",
                    evaluation_checkpoint.get("epoch"),
                    checkpoint_summary.get("epoch"),
                )

    comparison: dict[str, Any] = {}
    if not comparison_path.is_file():
        errors.append(
            f"sensor-source retraining comparison is missing: "
            f"{comparison_path}"
        )
    else:
        comparison = _read_json(comparison_path)
        _expect_equal(
            errors,
            "sensor_retraining.comparison_reference",
            comparison.get("reference", {}).get("name"),
            "qwen_retrained",
        )
        _expect_equal(
            errors,
            "sensor_retraining.comparison_candidate",
            comparison.get("candidate", {}).get("name"),
            "rule_v1_retrained",
        )
        paired = comparison.get("paired_bootstrap", {})
        _expect_equal(
            errors,
            "sensor_retraining.paired_repeats",
            paired.get("repeats"),
            5000,
        )
        _expect_equal(
            errors,
            "sensor_retraining.paired_sample_count",
            paired.get("sample_count"),
            826,
        )
        _expect_equal(
            errors,
            "sensor_retraining.paired_cluster_count",
            paired.get("cluster_count"),
            570,
        )
        _expect_probability(
            errors,
            "sensor_retraining.mcnemar_p",
            comparison.get("mcnemar_exact", {}).get("p_value"),
        )
        for metric in (
            "accuracy",
            "macro_f1",
            "ece",
            "negative_log_likelihood",
            "multiclass_brier_score",
        ):
            interval = paired.get("differences", {}).get(metric, {})
            try:
                lower = float(interval.get("lower"))
                point = float(interval.get("point"))
                upper = float(interval.get("upper"))
                if not lower <= point <= upper:
                    errors.append(
                        f"sensor_retraining.{metric}: point estimate is "
                        "outside its paired interval"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"sensor_retraining.{metric}: paired interval is invalid"
                )
    metrics = (
        _read_json(evaluation_path).get("metrics", {})
        if evaluation_path.is_file()
        else {}
    )
    return {
        "protocol_input_hash_count": input_count,
        "resolved_config_sha256": (
            _sha256(config_path) if config_path.is_file() else None
        ),
        "checkpoint": checkpoint_summary,
        "metrics": metrics,
        "comparison": comparison,
    }


def _audit_stratified_rows(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    errors: list[str],
) -> None:
    thresholds = {
        float(value)
        for value in protocol.get("evaluation", {}).get(
            "confidence_thresholds",
            [],
        )
    }
    sources = set(
        protocol.get("evaluation", {}).get("prediction_sources", [])
    )
    support_by_stratum: dict[tuple[str, str], int] = {}
    for dimension, definition in protocol.get("strata", {}).items():
        for stratum, support in definition.get(
            "expected_support",
            {},
        ).items():
            support_by_stratum[(dimension, stratum)] = int(support)
    expected = {
        (threshold, source, dimension, stratum)
        for threshold in thresholds
        for source in sources
        for dimension, stratum in support_by_stratum
    }
    actual: set[tuple[float, str, str, str]] = set()
    for index, row in enumerate(rows):
        key = (
            float(row.get("confidence_threshold", -1.0)),
            str(row.get("prediction_source", "")),
            str(row.get("dimension", "")),
            str(row.get("stratum", "")),
        )
        if key in actual:
            errors.append(f"scene_strata.rows[{index}]: duplicate {key}")
        actual.add(key)
        expected_support = support_by_stratum.get((key[2], key[3]))
        _expect_equal(
            errors,
            f"scene_strata.rows[{index}].support",
            row.get("support"),
            expected_support,
        )
        try:
            support = int(row.get("support"))
            localized = int(row.get("localized_count"))
            correct = int(row.get("correct_count"))
        except (TypeError, ValueError):
            errors.append(
                f"scene_strata.rows[{index}]: invalid integer counts"
            )
            continue
        if not 0 <= correct <= localized <= support:
            errors.append(
                f"scene_strata.rows[{index}]: expected "
                "0 <= correct <= localized <= support"
            )
        for metric in (
            "localization_recall",
            "classification_accuracy_on_localized",
            "end_to_end_recall",
        ):
            _expect_probability(
                errors,
                f"scene_strata.rows[{index}].{metric}",
                row.get(metric),
            )
            interval_name = (
                "classification_accuracy"
                if metric == "classification_accuracy_on_localized"
                else metric
            )
            lower = row.get(f"{interval_name}_ci_lower")
            upper = row.get(f"{interval_name}_ci_upper")
            _expect_probability(
                errors,
                f"scene_strata.rows[{index}].{interval_name}_ci_lower",
                lower,
            )
            _expect_probability(
                errors,
                f"scene_strata.rows[{index}].{interval_name}_ci_upper",
                upper,
            )
            try:
                point = float(row.get(metric))
                if not float(lower) <= point <= float(upper):
                    errors.append(
                        f"scene_strata.rows[{index}].{metric}: "
                        "point estimate is outside its interval"
                    )
            except (TypeError, ValueError):
                pass
    _expect_equal(errors, "scene_strata.combinations", actual, expected)


def _audit_scene_strata_reporting_recovery(
    reports_root: Path,
    *,
    report_path: Path,
    errors: list[str],
) -> dict[str, Any]:
    initial_error_count = len(errors)
    yolo_root = reports_root / "yolo-seg"
    incident_path = yolo_root / "SCENE_STRATA_REPORTING_INCIDENT.json"
    addendum_path = (
        yolo_root / "SCENE_STRATA_REPORTING_PROTOCOL_ADDENDUM.json"
    )
    outcome_path = (
        yolo_root / "SCENE_STRATA_REPORTING_RECOVERY_OUTCOME.json"
    )
    metric_dir = report_path.parent
    evaluation_path = metric_dir / "evaluation.json"
    metrics_path = metric_dir / "metrics.csv"
    markdown_path = metric_dir / "REPORT.md"
    required = (
        incident_path,
        incident_path.with_suffix(".md"),
        addendum_path,
        addendum_path.with_suffix(".md"),
        outcome_path,
        outcome_path.with_suffix(".md"),
        markdown_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        errors.extend(
            f"scene_strata.reporting_recovery missing artifact: {path}"
            for path in missing
        )
        return {
            "valid": False,
            "incident": incident_path.as_posix(),
            "addendum": addendum_path.as_posix(),
            "outcome": outcome_path.as_posix(),
            "missing_count": len(missing),
        }

    incident = _read_json(incident_path)
    addendum = _read_json(addendum_path)
    outcome = _read_json(outcome_path)
    _expect_equal(
        errors,
        "scene_strata.reporting_incident.state",
        incident.get("state"),
        "complete",
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_incident.valid",
        incident.get("valid"),
        True,
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_incident.exception_type",
        incident.get("failure", {}).get("exception_type"),
        "KeyError",
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_incident.row_count",
        incident.get("partial_outputs", {}).get("row_count"),
        42,
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_addendum.status",
        addendum.get("status"),
        "frozen_after_metrics_complete_before_report_recovery",
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_addendum.valid",
        addendum.get("valid"),
        True,
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_addendum.incident_sha256",
        addendum.get("incident", {}).get("sha256"),
        _sha256(incident_path),
    )
    invariants = addendum.get("invariants", {})
    for name in (
        "metric_recomputation",
        "training_rerun",
        "detector_rerun",
        "matching_rerun",
        "bootstrap_rerun",
    ):
        _expect_equal(
            errors,
            f"scene_strata.reporting_addendum.{name}",
            invariants.get(name),
            False,
        )
    _expect_equal(
        errors,
        "scene_strata.reporting_addendum.expected_row_count",
        invariants.get("expected_row_count"),
        42,
    )

    def audit_bindings(
        value: Mapping[str, Any],
        *,
        prefix: str,
        expected_count: int,
    ) -> int:
        bindings = value.get("source_bindings", {})
        if not isinstance(bindings, Mapping):
            errors.append(f"{prefix}: source bindings are missing")
            return 0
        for name, binding in bindings.items():
            if not isinstance(binding, Mapping):
                errors.append(f"{prefix}.{name}: binding is malformed")
                continue
            path = _resolve(Path(str(binding.get("path", ""))))
            if not path.is_file():
                errors.append(f"{prefix}.{name}: source is missing: {path}")
                continue
            _expect_equal(
                errors,
                f"{prefix}.{name}.size_bytes",
                binding.get("size_bytes"),
                path.stat().st_size,
            )
            _expect_equal(
                errors,
                f"{prefix}.{name}.sha256",
                binding.get("sha256"),
                _sha256(path),
            )
        if len(bindings) != expected_count:
            errors.append(
                f"{prefix}: expected {expected_count} bindings, "
                f"got {len(bindings)}"
            )
        return len(bindings)

    addendum_binding_count = audit_bindings(
        addendum,
        prefix="scene_strata.reporting_addendum",
        expected_count=17,
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.state",
        outcome.get("state"),
        "complete",
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.valid",
        outcome.get("valid"),
        True,
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.incident_sha256",
        outcome.get("incident", {}).get("sha256"),
        _sha256(incident_path),
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.addendum_sha256",
        outcome.get("addendum", {}).get("sha256"),
        _sha256(addendum_path),
    )
    recovery = outcome.get("recovery", {})
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.metric_recomputation",
        recovery.get("metric_recomputation"),
        False,
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.metric_hashes_unchanged",
        recovery.get("metric_hashes_unchanged"),
        True,
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.row_count",
        recovery.get("row_count"),
        42,
    )
    expected_hashes = {
        "evaluation_json": _sha256(evaluation_path),
        "metrics_csv": _sha256(metrics_path),
    }
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.before_hashes",
        recovery.get("before_hashes"),
        expected_hashes,
    )
    _expect_equal(
        errors,
        "scene_strata.reporting_outcome.after_hashes",
        recovery.get("after_hashes"),
        expected_hashes,
    )
    outcome_binding_count = audit_bindings(
        outcome,
        prefix="scene_strata.reporting_outcome",
        expected_count=6,
    )
    return {
        "valid": len(errors) == initial_error_count,
        "incident": incident_path.as_posix(),
        "incident_sha256": _sha256(incident_path),
        "addendum": addendum_path.as_posix(),
        "addendum_sha256": _sha256(addendum_path),
        "outcome": outcome_path.as_posix(),
        "outcome_sha256": _sha256(outcome_path),
        "metric_hashes_unchanged": recovery.get(
            "metric_hashes_unchanged"
        ),
        "source_binding_count": (
            addendum_binding_count + outcome_binding_count
        ),
    }


def _audit_scene_strata(
    reports_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    protocol_path = (
        reports_root
        / "yolo-seg"
        / "STRATIFIED_EVALUATION_PROTOCOL.json"
    )
    report_path = (
        reports_root
        / "yolo-seg"
        / "integrated"
        / "stratified"
        / "evaluation.json"
    )
    integrated_path = (
        reports_root / "yolo-seg" / "integrated" / "evaluation.json"
    )
    if not protocol_path.is_file():
        errors.append(
            f"scene-strata protocol is missing: {protocol_path}"
        )
        return {}
    if not report_path.is_file():
        errors.append(
            f"scene-strata evaluation is missing: {report_path}"
        )
        return {}
    protocol = _read_json(protocol_path)
    report = _read_json(report_path)
    integrated_addendum_path = (
        reports_root
        / "yolo-seg"
        / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
    )
    stratified_addendum_path = (
        reports_root
        / "yolo-seg"
        / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
    )
    integrated_addendum: Mapping[str, Any] = {}
    stratified_addendum: Mapping[str, Any] = {}
    revised_integrated_hash = ""
    revised_strata_hash = ""
    if not integrated_addendum_path.is_file():
        errors.append(
            "scene_strata integrated protocol addendum is missing: "
            f"{integrated_addendum_path}"
        )
    else:
        integrated_addendum = _read_json(integrated_addendum_path)
    if not stratified_addendum_path.is_file():
        errors.append(
            "scene_strata protocol addendum is missing: "
            f"{stratified_addendum_path}"
        )
    else:
        stratified_addendum = _read_json(stratified_addendum_path)

    protocol_hashes = {
        Path(str(entry.get("path", ""))).as_posix(): str(
            entry.get("sha256", "")
        ).lower()
        for entry in protocol.get("input_hashes", [])
    }
    if integrated_addendum:
        _expect_equal(
            errors,
            "scene_strata.integrated_addendum_status",
            integrated_addendum.get("status"),
            "frozen_after_path_identity_failure_before_recovery_rerun",
        )
        _expect_equal(
            errors,
            "scene_strata.integrated_addendum_valid",
            integrated_addendum.get("valid"),
            True,
        )
        revised_integrated = integrated_addendum.get(
            "implementation_fix", {}
        ).get("revised_evaluation_source", {})
        revised_integrated_hash = str(
            revised_integrated.get("sha256", "")
        ).lower()
        _expect_equal(
            errors,
            "scene_strata.integrated_original_source_sha256",
            integrated_addendum.get("implementation_fix", {})
            .get("original_evaluation_source", {})
            .get("sha256"),
            protocol_hashes.get("evaluate_scene_pipeline.py"),
        )
    if stratified_addendum:
        _expect_equal(
            errors,
            "scene_strata.addendum_status",
            stratified_addendum.get("status"),
            "frozen_after_integrated_path_addendum_before_stratified_run",
        )
        _expect_equal(
            errors,
            "scene_strata.addendum_valid",
            stratified_addendum.get("valid"),
            True,
        )
        _expect_equal(
            errors,
            "scene_strata.addendum_protocol_sha256",
            stratified_addendum.get("original_protocol", {}).get("sha256"),
            _sha256(protocol_path),
        )
        _expect_equal(
            errors,
            "scene_strata.addendum_dependency_sha256",
            stratified_addendum.get(
                "dependency_protocol_addendum", {}
            ).get("sha256"),
            (
                _sha256(integrated_addendum_path)
                if integrated_addendum_path.is_file()
                else None
            ),
        )
        fix = stratified_addendum.get("implementation_fix", {})
        revised_strata = fix.get("revised_strata_source", {})
        revised_strata_hash = str(
            revised_strata.get("sha256", "")
        ).lower()
        _expect_equal(
            errors,
            "scene_strata.addendum_original_strata_sha256",
            fix.get("original_strata_source", {}).get("sha256"),
            protocol_hashes.get("evaluate_scene_strata.py"),
        )
        _expect_equal(
            errors,
            "scene_strata.addendum_original_integrated_sha256",
            fix.get("original_integrated_source", {}).get("sha256"),
            protocol_hashes.get("evaluate_scene_pipeline.py"),
        )
        _expect_equal(
            errors,
            "scene_strata.addendum_revised_integrated_sha256",
            fix.get("revised_integrated_source", {}).get("sha256"),
            revised_integrated_hash,
        )
        for name, binding in (
            ("revised_strata", revised_strata),
            ("revised_integrated", fix.get("revised_integrated_source", {})),
        ):
            revised_path = _resolve(Path(str(binding.get("path", ""))))
            if not revised_path.is_file():
                errors.append(
                    f"scene_strata.{name} source is missing: {revised_path}"
                )
            else:
                _expect_equal(
                    errors,
                    f"scene_strata.{name}_source_sha256",
                    binding.get("sha256"),
                    _sha256(revised_path),
                )
    input_count = _audit_file_entries(
        protocol.get("input_hashes", []),
        prefix="scene_strata.inputs",
        errors=errors,
        expected_hash_overrides={
            "evaluate_scene_pipeline.py": revised_integrated_hash,
            "evaluate_scene_strata.py": revised_strata_hash,
        },
    )
    expected_tree = protocol.get("label_tree", {})
    label_root = _resolve(Path(str(expected_tree.get("path", ""))))
    if not label_root.is_dir():
        errors.append(
            f"scene_strata.label_tree: directory is missing: {label_root}"
        )
    else:
        actual_tree = _sha256_tree(label_root)
        for key in ("file_count", "size_bytes", "sha256"):
            _expect_equal(
                errors,
                f"scene_strata.label_tree.{key}",
                actual_tree[key],
                expected_tree.get(key),
            )
    frozen = report.get("frozen_protocol", {})
    _expect_equal(
        errors,
        "scene_strata.protocol_status",
        protocol.get("status"),
        "frozen_before_detector_training",
    )
    _expect_equal(
        errors,
        "scene_strata.protocol",
        report.get("protocol"),
        "preregistered_scene_stratified_analysis",
    )
    _expect_equal(
        errors,
        "scene_strata.protocol_sha256",
        frozen.get("sha256"),
        _sha256(protocol_path),
    )
    _expect_equal(
        errors,
        "scene_strata.integrated_protocol_addendum_sha256",
        frozen.get("integrated_protocol_addendum", {}).get("sha256"),
        (
            _sha256(integrated_addendum_path)
            if integrated_addendum_path.is_file()
            else None
        ),
    )
    _expect_equal(
        errors,
        "scene_strata.stratified_protocol_addendum_sha256",
        frozen.get("stratified_protocol_addendum", {}).get("sha256"),
        (
            _sha256(stratified_addendum_path)
            if stratified_addendum_path.is_file()
            else None
        ),
    )
    _expect_equal(
        errors,
        "scene_strata.stratified_source_sha256",
        frozen.get("stratified_protocol_addendum", {}).get(
            "strata_source_sha256"
        ),
        revised_strata_hash,
    )
    _expect_equal(
        errors,
        "scene_strata.input_hash_check_count",
        len(frozen.get("input_hash_checks", [])),
        input_count + 1,
    )
    source = report.get("source_integrated_report", {})
    if integrated_path.is_file():
        _expect_equal(
            errors,
            "scene_strata.integrated_report_sha256",
            source.get("sha256"),
            _sha256(integrated_path),
        )
        integrated = _read_json(integrated_path)
        _expect_equal(
            errors,
            "scene_strata.detector_checkpoint_sha256",
            source.get("detector_checkpoint_sha256"),
            integrated.get("detector", {}).get("sha256"),
        )
    else:
        errors.append(
            f"scene_strata integrated source is missing: {integrated_path}"
        )
    dataset = report.get("dataset", {})
    _expect_equal(
        errors,
        "scene_strata.source_image_count",
        dataset.get("source_image_count"),
        570,
    )
    _expect_equal(
        errors,
        "scene_strata.ground_truth_object_count",
        dataset.get("ground_truth_object_count"),
        825,
    )
    _expect_equal(
        errors,
        "scene_strata.bootstrap_repeats",
        report.get("bootstrap", {}).get("repeats"),
        2000,
    )
    rows = list(report.get("rows", []))
    _audit_stratified_rows(rows, protocol, errors)
    reporting_recovery = _audit_scene_strata_reporting_recovery(
        reports_root,
        report_path=report_path,
        errors=errors,
    )
    return {
        "source_image_count": dataset.get("source_image_count"),
        "ground_truth_object_count": dataset.get(
            "ground_truth_object_count"
        ),
        "protocol_input_hash_count": input_count + 1,
        "row_count": len(rows),
        "rows": rows,
        "reporting_recovery": reporting_recovery,
    }


def audit_final_evidence(
    *,
    reports_root: Path,
    allow_pending: bool = False,
) -> dict[str, Any]:
    reports_root = _resolve(reports_root)
    errors: list[str] = []
    pending: list[str] = []
    sections: dict[str, Any] = {
        "dataset": _audit_dataset(reports_root, errors),
    }
    training_root = reports_root / "training"
    formal_status_path = training_root / "formal_ablations.status.json"
    statistics_status_path = (
        training_root / "formal_ablations.reporting_status.json"
    )
    yolo_status_path = training_root / "yolo_seg.status.json"
    sensor_source_status_path = (
        training_root / "sensor_source_transfer.status.json"
    )
    formal_state = (
        _read_json(formal_status_path).get("state")
        if formal_status_path.is_file()
        else None
    )
    statistics_state = (
        _read_json(statistics_status_path).get("state")
        if statistics_status_path.is_file()
        else None
    )
    yolo_state = (
        _read_json(yolo_status_path).get("state")
        if yolo_status_path.is_file()
        else None
    )
    sensor_source_state = (
        _read_json(sensor_source_status_path).get("state")
        if sensor_source_status_path.is_file()
        else None
    )
    formal_active = _status_is_active(formal_state)
    statistics_active = formal_active or _status_is_active(
        statistics_state
    )
    yolo_active = statistics_active or _status_is_active(yolo_state)
    sensor_source_active = (
        yolo_active or _status_is_active(sensor_source_state)
    )
    formal_complete = _dependency_state(
        formal_status_path,
        "formal ablation queue",
        allow_pending=allow_pending,
        errors=errors,
        pending=pending,
    )
    statistics_complete = _dependency_state(
        statistics_status_path,
        "formal ablation statistics",
        allow_pending=allow_pending,
        errors=errors,
        pending=pending,
        defer_failed=formal_active,
        defer_reason="formal ablation queue",
    )
    yolo_complete = _dependency_state(
        yolo_status_path,
        "YOLO-seg and integrated evaluation",
        allow_pending=allow_pending,
        errors=errors,
        pending=pending,
        defer_failed=statistics_active,
        defer_reason="formal ablation reporting chain",
    )
    sensor_source_complete = _dependency_state(
        sensor_source_status_path,
        "sensor-source transfer evaluation",
        allow_pending=allow_pending,
        errors=errors,
        pending=pending,
        defer_failed=yolo_active,
        defer_reason="YOLO-seg and integrated evaluation chain",
    )
    sensor_retraining_complete = _dependency_state(
        training_root / "sensor_source_retraining.status.json",
        "sensor-source independent retraining",
        allow_pending=allow_pending,
        errors=errors,
        pending=pending,
        defer_failed=sensor_source_active,
        defer_reason="sensor-source transfer evaluation chain",
    )
    if formal_complete and statistics_complete:
        sections["formal_ablations"] = _audit_formal_ablations(
            reports_root,
            errors,
        )
    if yolo_complete:
        sections["yolo_seg"] = _audit_yolo_seg(reports_root, errors)
        sections["scene_pipeline"] = _audit_scene_pipeline(reports_root, errors)
        scene_strata_path = (
            reports_root
            / "yolo-seg"
            / "integrated"
            / "stratified"
            / "evaluation.json"
        )
        if scene_strata_path.is_file():
            sections["scene_strata"] = _audit_scene_strata(
                reports_root,
                errors,
            )
        elif allow_pending:
            pending.append(
                "scene stratified analysis: evaluation report is missing"
            )
        else:
            errors.append(
                f"scene-strata evaluation is missing: {scene_strata_path}"
            )
    if sensor_source_complete:
        sections["sensor_source_transfer"] = _audit_sensor_source_transfer(
            reports_root,
            errors,
        )
    if sensor_retraining_complete:
        sections[
            "sensor_source_retraining"
        ] = _audit_sensor_source_retraining(
            reports_root,
            errors,
        )

    warnings = [
        "The 826-object test split is a repeatedly inspected internal benchmark, not an external blind set.",
        "Qwen/rule sidecars are image-conditioned soft sensors, not independent hardware measurements.",
        "Eight exact duplicate object records and 24 high-confidence cross-split near-duplicate pairs must remain disclosed.",
        "Formal ablations currently quantify one training seed; paired test intervals do not include training-seed variance.",
        "YOLO-seg and integrated scene metrics do not mean that MMFC features were injected into the detector head.",
        "Relative-size quartiles and bbox overlap are diagnostic strata, not standard small-target or human occlusion labels.",
        "The sensor-source comparison is a same-checkpoint runtime intervention, not an independently retrained data-source ablation.",
        "The independent Qwen/rule-v1 retraining comparison uses one seed and does not include training-seed variance.",
    ]
    state = "failed" if errors else ("pending" if pending else "complete")
    return {
        "schema_version": 1,
        "state": state,
        "valid": state == "complete",
        "error_count": len(errors),
        "pending_count": len(pending),
        "warning_count": len(warnings),
        "errors": errors,
        "pending": pending,
        "warnings": warnings,
        "sections": sections,
    }


def main() -> None:
    args = parse_args()
    report = audit_final_evidence(
        reports_root=args.reports,
        allow_pending=args.allow_pending,
    )
    if args.output is None:
        filename = (
            "FINAL_EVIDENCE_AUDIT.pending.json"
            if args.allow_pending
            else "FINAL_EVIDENCE_AUDIT.json"
        )
        output = _resolve(args.reports) / filename
    else:
        output = _resolve(args.output)
    _atomic_write_json(output, report)
    print(json.dumps({"output": output.as_posix(), **report}, ensure_ascii=False, indent=2))
    if report["state"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
