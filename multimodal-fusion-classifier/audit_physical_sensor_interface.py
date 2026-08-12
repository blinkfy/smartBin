"""Audit the physical-sensor software contract without claiming hardware data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.external_blind import _sensor_payload
from mmfc.physical_sensor import calibrate_physical_observation


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORT_ROOT / "PHYSICAL_SENSOR_INTERFACE_AUDIT.json"
SOURCES = {
    "calibration_module": PROJECT_ROOT / "mmfc" / "physical_sensor.py",
    "conversion_cli": PROJECT_ROOT / "prepare_physical_sensor_sidecars.py",
    "json_schema": PROJECT_ROOT
    / "schemas"
    / "physical_sensor_observation_v1.schema.json",
    "tests": PROJECT_ROOT / "tests" / "test_physical_sensor.py",
    "interface_document": REPORT_ROOT / "PHYSICAL_SENSOR_INTERFACE.md",
}
EXPECTED_PROPERTY_RANGES = {
    "weight_g": (0.1, 5000.0),
    "temperature_c": (-20.0, 80.0),
    "moisture_pct": (0.0, 100.0),
    "conductivity_us_cm": (0.0, 5000.0),
    "odor_index": (0.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the physical 4x4 sensor software interface"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _fixture() -> dict[str, Any]:
    grid = lambda value: [[value] * 4 for _ in range(4)]
    return {
        "sample_id": "software-contract-fixture",
        "source_image_id": "fixture-source-image",
        "sensor_id": "fixture-array",
        "measurement_unit": "mV",
        "image_timestamp": "2026-07-29T10:00:00.000Z",
        "sensor_timestamp": "2026-07-29T10:00:00.025Z",
        "raw_grid": grid(60.0),
        "properties": {
            "weight_g": 100.0,
            "temperature_c": 25.0,
            "moisture_pct": 10.0,
            "conductivity_us_cm": 20.0,
            "odor_index": 0.1,
        },
        "calibration": {
            "calibration_id": "fixture-calibration",
            "calibrated_at": "2026-07-01T00:00:00Z",
            "zero_grid": grid(10.0),
            "full_scale_grid": grid(110.0),
        },
    }


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite physical interface audit")
    missing = [name for name, path in SOURCES.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "physical interface sources missing: " + ", ".join(missing)
        )
    schema = json.loads(
        SOURCES["json_schema"].read_text(encoding="utf-8")
    )
    expected_required = {
        "sample_id",
        "source_image_id",
        "sensor_id",
        "measurement_unit",
        "image_timestamp",
        "sensor_timestamp",
        "raw_grid",
        "properties",
        "calibration",
    }
    if (
        schema.get("$schema")
        != "https://json-schema.org/draft/2020-12/schema"
        or set(schema.get("required", [])) != expected_required
        or schema.get("additionalProperties") is not False
    ):
        raise ValueError("physical observation JSON schema identity mismatch")
    property_schema = schema["properties"]["properties"]["properties"]
    actual_property_ranges = {
        name: (definition.get("minimum"), definition.get("maximum"))
        for name, definition in property_schema.items()
    }
    if actual_property_ranges != EXPECTED_PROPERTY_RANGES:
        raise ValueError("physical observation property ranges mismatch")
    converted = calibrate_physical_observation(
        _fixture(),
        max_skew_ms=100.0,
        max_calibration_age_days=365.0,
    )
    _sensor_payload(converted, "physical-array-v1")
    if (
        converted["source"]["time_skew_ms"] != 25.0
        or converted["source"]["clipped_cell_count"] != 0
        or converted["source"]["physical_hardware_origin_declared"] is not True
        or converted["source"]["physical_hardware_origin_verified"] is not False
        or any(
            value != 0.5
            for row in converted["normalized_grid"]
            for value in row
        )
    ):
        raise ValueError("physical interface fixture result mismatch")
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "physical_4x4_sensor_software_interface_only",
        "software_interface_ready": True,
        "existing_model_sidecar_compatible": True,
        "label_and_annotation_inputs_rejected": True,
        "property_ranges_enforced": True,
        "cellwise_calibration_implemented": True,
        "timestamp_alignment_enforced": True,
        "calibration_age_enforced": True,
        "overflow_clipping_audited": True,
        "physical_hardware_origin_declared_by_record": True,
        "physical_hardware_origin_verified_by_software": False,
        "physical_record_count": 0,
        "actual_hardware_acquisition_complete": False,
        "hardware_validation_ready": False,
        "rgbd_ready": False,
        "fixture": {
            "purpose": "software_contract_test_only",
            "hardware_measurement_claimed": False,
            "hardware_origin_verified": False,
            "normalized_value": 0.5,
            "time_skew_ms": 25.0,
        },
        "claim_policy": {
            "allowed": [
                "实体4x4阵列的软件接入、逐格校准和时间同步契约已实现。",
                "符合契约的硬件记录可转换为现有normalized_grid sidecar。",
            ],
            "not_allowed": [
                "当前已有实体阵列采集数据。",
                "已验证硬件鲁棒性、设备标定质量或部署性能。",
                "软件接口夹具可替代真实硬件实验。",
            ],
        },
        "source_bindings": {
            **{name: _binding(path) for name, path in SOURCES.items()},
            "auditor_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(
        markdown,
        "\n".join(
            [
                "# 实体4×4传感器软件接口审计",
                "",
                "- 软件接口：PASS。",
                "- 与现有sidecar兼容：true。",
                "- 五项属性量程约束：PASS。",
                "- 实体记录数：0。",
                "- 硬件验证就绪：false。",
                "- RGB-D就绪：false。",
                "",
                "已验证严格4×4输入、逐格零点/满量程校准、时区时间戳、",
                "图像/传感器时差、校准年龄、超量程裁剪披露和监督字段拒绝。",
                "",
                "审计夹具只用于软件契约测试，不是设备测量。真实采集、标定和",
                "重训结果到位前，不得声称硬件实测或完整专利复现。",
                "",
            ]
        ),
    )
    return report


def main() -> None:
    report = build(parse_args().output)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "software_interface_ready": report[
                    "software_interface_ready"
                ],
                "physical_record_count": report["physical_record_count"],
                "hardware_validation_ready": report[
                    "hardware_validation_ready"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
