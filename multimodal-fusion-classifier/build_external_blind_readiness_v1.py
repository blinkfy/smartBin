"""Summarize readiness for the frozen external blind confirmation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evaluate_uncertainty import _source_binding
from mmfc.external_blind import sha256_file, validate_external_blind_bundle
from mmfc.training import PROJECT_ROOT
from validate_external_blind_intake import (
    DEFAULT_BUNDLE,
    DEFAULT_DHASH,
    DEFAULT_INTEGRITY,
    DEFAULT_PROTOCOL,
)


REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORT_ROOT / "EXTERNAL_BLIND_READINESS_V1.json"
DEFAULT_EVALUATION = REPORT_ROOT / "external-blind-v1" / "evaluation"
DHASH_REPORT = REPORT_ROOT / "EXTERNAL_BLIND_DEVELOPMENT_DHASH_INDEX.json"
GOVERNANCE = REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V11.json"
PHYSICAL_AUDIT = REPORT_ROOT / "PHYSICAL_SENSOR_INTERFACE_AUDIT.json"
HARDWARE_RELEASE_SOURCES = {
    "physical_release_module": PROJECT_ROOT / "mmfc" / "physical_release.py",
    "physical_release_cli": (
        PROJECT_ROOT / "prepare_synchronized_hardware_release.py"
    ),
    "physical_release_attestation_schema": (
        PROJECT_ROOT
        / "schemas"
        / "synchronized_hardware_attestation_v1.schema.json"
    ),
    "physical_release_document": (
        REPORT_ROOT / "SYNCHRONIZED_HARDWARE_RELEASE.md"
    ),
}


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


def _bound_sources_are_current(protocol: dict[str, Any]) -> tuple[bool, list[str]]:
    drifted: list[str] = []
    for name, binding in protocol.get("bindings", {}).items():
        path = PROJECT_ROOT / str(binding.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != binding.get("size_bytes")
            or sha256_file(path) != binding.get("sha256")
        ):
            drifted.append(str(name))
    return not drifted, drifted


def build_readiness(
    output: Path = DEFAULT_OUTPUT,
    bundle: Path = DEFAULT_BUNDLE,
    evaluation_dir: Path = DEFAULT_EVALUATION,
) -> dict[str, Any]:
    protocol = _read(DEFAULT_PROTOCOL)
    governance = _read(GOVERNANCE)
    physical_audit = _read(PHYSICAL_AUDIT)
    protocol_identity_valid = (
        protocol.get("protocol") == "external_blind_confirmation_v1"
        and protocol.get("status") == "preregistered_awaiting_external_data"
        and protocol.get("formal_paper_metric") is True
        and len(protocol.get("class_names", [])) == 19
    )
    bindings_current, drifted_bindings = _bound_sources_are_current(protocol)
    hardware_release_sources_available = all(
        path.is_file() for path in HARDWARE_RELEASE_SOURCES.values()
    )
    hardware_release_interface_ready = (
        hardware_release_sources_available
        and "synchronized-hardware"
        in protocol.get("intake_criteria", {}).get(
            "allowed_sensor_source_modes", []
        )
        and physical_audit.get("valid") is True
        and physical_audit.get("software_interface_ready") is True
        and physical_audit.get("physical_record_count") == 0
        and physical_audit.get("hardware_validation_ready") is False
    )
    required_names = protocol.get("release_contract", {}).get(
        "required_files", []
    )
    required_files = [bundle / str(name) for name in required_names]
    external_bundle_available = bool(required_files) and all(
        path.is_file() for path in required_files
    )
    intake: dict[str, Any] | None = None
    if external_bundle_available:
        intake = validate_external_blind_bundle(
            bundle,
            protocol_path=DEFAULT_PROTOCOL,
            development_integrity_path=DEFAULT_INTEGRITY,
            development_dhash_index_path=DEFAULT_DHASH,
        )
    evaluation_report = evaluation_dir / "EVALUATION.json"
    evaluation_started = evaluation_dir.is_dir() and any(
        evaluation_dir.iterdir()
    )
    formal_external_result_available = evaluation_report.is_file()
    tooling_ready = (
        protocol_identity_valid
        and bindings_current
        and hardware_release_interface_ready
    )
    valid = tooling_ready and not (
        formal_external_result_available
        and not _read(evaluation_report).get("valid", False)
    )
    if formal_external_result_available:
        gate_state = "formal_external_result_available"
    elif intake is not None and intake.get("valid") is True:
        gate_state = "validated_bundle_awaiting_one_time_evaluation"
    elif external_bundle_available:
        gate_state = "external_bundle_failed_intake"
    else:
        gate_state = "awaiting_external_bundle"

    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": valid,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["protocol"],
        "protocol_status": protocol["status"],
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "tooling_ready": tooling_ready,
        "confirmed_manuscript_option": governance.get("decision", {}).get(
            "confirmed_option"
        ),
        "hardware_release_interface_ready": hardware_release_interface_ready,
        "physical_hardware_record_count": physical_audit.get(
            "physical_record_count"
        ),
        "physical_hardware_validation_ready": physical_audit.get(
            "hardware_validation_ready"
        ),
        "bound_sources_current": bindings_current,
        "drifted_bindings": drifted_bindings,
        "external_bundle_available": external_bundle_available,
        "intake_valid": None if intake is None else bool(intake["valid"]),
        "intake_error_count": (
            None if intake is None else int(intake["error_count"])
        ),
        "one_time_evaluation_started": evaluation_started,
        "formal_external_result_available": formal_external_result_available,
        "gate_state": gate_state,
        "submission_ready": (
            governance.get("decision", {}).get("confirmed_option") == "A"
            and formal_external_result_available
        ),
        "registered_minimums": {
            "total_samples": protocol["intake_criteria"][
                "minimum_total_samples"
            ],
            "source_images": protocol["intake_criteria"][
                "minimum_source_images"
            ],
            "samples_per_class": protocol["intake_criteria"][
                "minimum_samples_per_class"
            ],
            "source_images_per_class": protocol["intake_criteria"][
                "minimum_source_images_per_class"
            ],
            "acquisition_batches": protocol["intake_criteria"][
                "minimum_acquisition_batches"
            ],
            "capture_sessions": protocol["intake_criteria"][
                "minimum_capture_sessions"
            ],
        },
        "next_actions": [
            {
                "owner": "external_data_custodian",
                "action": (
                    "Provide RELEASE.json, manifest.jsonl, sensors.jsonl, "
                    "and referenced images under data/external-blind-v1. "
                    "For synchronized hardware, first bind calibrated sidecars "
                    "with the custodian attestation release tool."
                ),
                "blocking": True,
            },
            {
                "owner": "pipeline",
                "action": (
                    "Run validate_external_blind_intake.py, resolve any "
                    "manual dHash review warnings, then run "
                    "evaluate_external_blind_v1.py exactly once."
                ),
                "blocking": external_bundle_available,
            },
        ],
        "claim_boundary": (
            "The tooling and protocol are ready, but no external blind "
            "performance result exists until a new independent bundle passes "
            "intake and the registered one-time evaluation completes."
        ),
        "bindings": {
            "protocol": _source_binding(DEFAULT_PROTOCOL),
            "development_integrity": _source_binding(DEFAULT_INTEGRITY),
            "development_dhash_index": _source_binding(DEFAULT_DHASH),
            "development_dhash_report": _source_binding(DHASH_REPORT),
            "manuscript_governance_v11": _source_binding(GOVERNANCE),
            "physical_sensor_interface_audit": _source_binding(
                PHYSICAL_AUDIT
            ),
            **{
                name: _source_binding(path)
                for name, path in HARDWARE_RELEASE_SOURCES.items()
            },
            "builder_source": _source_binding(Path(__file__)),
        },
    }
    _atomic_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 外部盲测就绪状态 v1",
        "",
        f"- 协议：`{report['protocol']}`，已冻结。",
        f"- 工具链就绪：`{str(tooling_ready).lower()}`。",
        (
            "- 方案A已确认："
            f"`{str(report['confirmed_manuscript_option'] == 'A').lower()}`。"
        ),
        (
            "- 同步硬件保管人证明发布接口就绪："
            f"`{str(hardware_release_interface_ready).lower()}`；"
            "当前实体记录数仍为0。"
        ),
        (
            "- 外部数据包可用："
            f"`{str(external_bundle_available).lower()}`。"
        ),
        (
            "- 正式外部结果可用："
            f"`{str(formal_external_result_available).lower()}`。"
        ),
        f"- 当前门状态：`{gate_state}`。",
        "",
        "## 固定下限",
        "",
        "- 380 个对象、190 张源图。",
        "- 19 类每类至少 20 个对象、10 张源图。",
        "- 至少 3 个采集批次、3 个拍摄会话、2 名标签复核者。",
        "",
        "## 下一步",
        "",
        (
            "外部数据保管方提供三份清单及图像后，先运行 "
            "`validate_external_blind_intake.py`，通过后再运行 "
            "`evaluate_external_blind_v1.py`。正式评估只执行一次。"
        ),
        "",
        (
            "当前只能表述为协议和工具链已就绪，不能表述为已经完成"
            "外部泛化验证。"
        ),
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return report


def main() -> None:
    report = build_readiness()
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "tooling_ready": report["tooling_ready"],
                "external_bundle_available": report[
                    "external_bundle_available"
                ],
                "formal_external_result_available": report[
                    "formal_external_result_available"
                ],
                "gate_state": report["gate_state"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
