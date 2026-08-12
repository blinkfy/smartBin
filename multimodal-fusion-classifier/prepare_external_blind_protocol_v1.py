"""Preregister the external blind-confirmation intake and analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evaluate_uncertainty import _source_binding
from mmfc.training import PROJECT_ROOT, load_config, resolve_project_path


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "external-blind-v1"
)
DEFAULT_OUTPUT = REPORT_ROOT / "PROTOCOL.json"
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "ablations"
    / "full_cross_modal_quality.json"
)
CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "ablations"
    / "full_cross_modal_quality"
    / "best.pt"
)
INTEGRITY = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SOURCE_IMAGE_INTEGRITY_CURRENT.json"
)
DHASH_INDEX = (
    PROJECT_ROOT
    / "data"
    / "external-blind-governance"
    / "development_dhash_index.jsonl"
)
DHASH_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "EXTERNAL_BLIND_DEVELOPMENT_DHASH_INDEX.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the external blind confirmation protocol"
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
    class_names = [
        str(item["name"]) for item in config.get("classes", [])
    ]
    paths = {
        "config": CONFIG,
        "checkpoint": CHECKPOINT,
        "development_integrity": INTEGRITY,
        "development_dhash_index": DHASH_INDEX,
        "development_dhash_report": DHASH_REPORT,
        "contract_source": PROJECT_ROOT / "mmfc" / "external_blind.py",
        "validator_source": (
            PROJECT_ROOT / "validate_external_blind_intake.py"
        ),
        "evaluator_source": PROJECT_ROOT / "evaluate_external_blind_v1.py",
        "evaluation_core": PROJECT_ROOT / "mmfc" / "evaluation.py",
        "dataset_core": PROJECT_ROOT / "mmfc" / "training.py",
        "model_core": PROJECT_ROOT / "mmfc" / "model.py",
        "checkpoint_core": PROJECT_ROOT / "mmfc" / "checkpoints.py",
        "protocol_builder_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "external blind protocol inputs missing: " + ", ".join(missing)
        )
    if len(class_names) != 19:
        raise ValueError("external blind protocol requires 19 classes")
    dhash_report = json.loads(
        DHASH_REPORT.read_text(encoding="utf-8-sig")
    )
    if (
        dhash_report.get("valid") is not True
        or dhash_report.get("source_image_count") != 5990
        or dhash_report.get("bindings", {})
        .get("output", {})
        .get("sha256")
        != _source_binding(DHASH_INDEX)["sha256"]
    ):
        raise ValueError("development dHash evidence is invalid")

    return {
        "schema_version": 1,
        "protocol": "external_blind_confirmation_v1",
        "status": "preregistered_awaiting_external_data",
        "formal_paper_metric": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Perform one external confirmation of the selected v3 reliable "
            "multimodal object classifier after model, code, intake gates, "
            "metrics, and statistics are frozen."
        ),
        "checkpoint": _source_binding(CHECKPOINT),
        "class_names": class_names,
        "scope": {
            "task": "19-class general-waste object classification",
            "primary_model": "full_cross_modal_quality_v3",
            "primary_condition": "clean external sensor sidecar",
            "paired_reference": "same-checkpoint RGB-predicted fallback",
            "joint_detection_segmentation": False,
            "medical_waste_taxonomy": False,
        },
        "intake_criteria": {
            "minimum_total_samples": 380,
            "minimum_source_images": 190,
            "minimum_samples_per_class": 20,
            "minimum_source_images_per_class": 10,
            "minimum_acquisition_batches": 3,
            "minimum_capture_sessions": 3,
            "minimum_label_reviewers": 2,
            "exact_development_overlap_allowed": False,
            "perceptual_hash_error_distance": 2,
            "perceptual_hash_review_distance": 6,
            "allowed_sensor_source_modes": [
                "qwen-soft-sensor",
                "synchronized-hardware",
            ],
            "sensor_generation_reads_annotations": False,
            "sensor_generation_reads_class_labels": False,
            "corrupt_image_exclusion_allowed": (
                "only when decoding failure is logged before model forward"
            ),
            "post_label_sample_exclusion_allowed": False,
        },
        "release_contract": {
            "required_files": [
                "RELEASE.json",
                "manifest.jsonl",
                "sensors.jsonl",
            ],
            "release_binds_protocol_sha256": True,
            "release_binds_checkpoint_sha256": True,
            "labels_unsealed_after_protocol_freeze": True,
            "prior_model_development_access": False,
            "one_time_evaluation_authorized": True,
            "label_consensus_required": True,
            "acquisition_batch_and_session_ids_required": True,
            "collection_site_and_device_ids_required": True,
            "sensor_source_mode_required": True,
            "collection_completed_before_label_unseal": True,
        },
        "leakage_gates": {
            "development_source_image_count": 5990,
            "sample_id_overlap": "must_equal_zero",
            "source_image_id_overlap": "must_equal_zero",
            "exact_image_sha256_overlap": "must_equal_zero",
            "difference_hash_distance_le_2": "reject",
            "difference_hash_distance_3_to_6": "manual_review_required",
        },
        "evaluation": {
            "one_time_primary_analysis": True,
            "batch_size": 16,
            "num_workers": 0,
            "device": "cuda",
            "ece_bins": 15,
            "warmup_batches": 3,
            "seed": 42,
            "bootstrap_unit": "source_image_id",
            "bootstrap_repeats": 5000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 20260729,
            "primary_metrics": [
                "Accuracy",
                "Macro F1",
                "negative log likelihood",
                "multiclass Brier score",
                "ECE",
            ],
            "secondary_metrics": [
                "Macro Precision",
                "Macro Recall",
                "Weighted F1",
                "per-class Precision/Recall/F1",
                "inference milliseconds per sample",
            ],
            "paired_analysis": (
                "clean candidate minus same-checkpoint RGB fallback"
            ),
            "performance_acceptance_threshold": None,
            "model_selection_or_retraining_after_results": False,
        },
        "reporting": {
            "report_all_registered_metrics": True,
            "report_all_19_classes": True,
            "report_sensor_source_mode_and_site_count": True,
            "report_intake_warnings": True,
            "negative_or_null_result_retained": True,
        },
        "claim_boundaries": [
            (
                "External confirmation applies only to the released sites, "
                "devices, acquisition period, and sensor-source mode."
            ),
            (
                "Qwen soft sensors remain image-conditioned estimates and "
                "must not be described as physical measurements."
            ),
            (
                "This protocol does not add RGB-D, instance segmentation, "
                "or the medical-waste taxonomy."
            ),
            (
                "No external performance threshold is used to decide whether "
                "the result is retained or reported."
            ),
        ],
        "current_gate": {
            "external_bundle_available": False,
            "labels_unsealed": False,
            "intake_validated": False,
            "one_time_evaluation_started": False,
        },
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
                "refusing to replace a drifted external blind protocol"
            )
        return existing
    _atomic_text(
        output,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 外部盲测确认协议 v1",
        "",
        "- 状态：已预注册，等待新批次外部数据。",
        "- 主模型：固定 v3 full_cross_modal_quality best.pt。",
        "- 任务：19类一般垃圾对象分类。",
        "- 最低380个对象、190张源图，每类至少20个对象和10张源图。",
        "- 至少3个采集批次和3个拍摄会话，双人标签一致。",
        "- 与开发集执行ID、SHA-256和64位dHash隔离门。",
        "- 主分析只执行一次；所有注册指标与负结果均保留。",
        "- 主条件为外部sidecar，配对参考为同检查点RGB fallback。",
        "",
        (
            "Qwen sidecar仍是图像条件软传感器；本协议不包含RGB-D、"
            "实体阵列或医疗废物类别。"
        ),
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
                "minimum_total_samples": protocol["intake_criteria"][
                    "minimum_total_samples"
                ],
                "minimum_source_images": protocol["intake_criteria"][
                    "minimum_source_images"
                ],
                "binding_count": len(protocol["bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
