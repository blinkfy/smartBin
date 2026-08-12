from __future__ import annotations

import hashlib
import json

import pytest

from mmfc.external_blind import _sensor_payload
from mmfc.physical_release import (
    ATTESTATION_KEYS,
    ATTESTATION_TYPE,
    PROJECT_ROOT,
    release_sidecars,
)
from prepare_physical_sensor_sidecars import build_sidecars


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid(value):
    return [[value] * 4 for _ in range(4)]


def _raw_record():
    return {
        "sample_id": "sample-1",
        "source_image_id": "image-1",
        "sensor_id": "array-1",
        "measurement_unit": "mV",
        "image_timestamp": "2026-07-29T10:00:00.000Z",
        "sensor_timestamp": "2026-07-29T10:00:00.025Z",
        "raw_grid": _grid(60.0),
        "properties": {
            "weight_g": 100.0,
            "temperature_c": 25.0,
            "moisture_pct": 10.0,
            "conductivity_us_cm": 20.0,
            "odor_index": 0.1,
        },
        "calibration": {
            "calibration_id": "cal-1",
            "calibrated_at": "2026-07-01T00:00:00Z",
            "zero_grid": _grid(10.0),
            "full_scale_grid": _grid(110.0),
        },
    }


def _physical_files(tmp_path):
    raw = tmp_path / "raw.jsonl"
    sidecars = tmp_path / "physical.jsonl"
    build_report = tmp_path / "BUILD_REPORT.json"
    manifest = tmp_path / "manifest.jsonl"
    raw.write_text(
        json.dumps(_raw_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_sidecars(raw, sidecars, build_report)
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "source_image_id": "image-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecars, build_report, manifest


def _attestation(sidecars, build_report, manifest):
    return {
        "schema_version": 1,
        "attestation_type": "synchronized_hardware_origin_v1",
        "dataset_id": "external-batch-1",
        "custodian_id": "custodian-1",
        "signed_at": "2026-07-29T11:00:00+08:00",
        "physical_sidecar_sha256": _sha256(sidecars),
        "physical_build_report_sha256": _sha256(build_report),
        "external_manifest_sha256": _sha256(manifest),
        "sample_count": 1,
        "collection_site_ids": ["site-1"],
        "capture_device_ids": ["camera-1", "array-1"],
        "hardware_origin_attested_by_custodian": True,
        "independent_of_model_development": True,
        "annotation_and_class_inputs_excluded": True,
    }


def test_attestation_schema_matches_runtime_contract():
    schema = json.loads(
        (
            PROJECT_ROOT
            / "schemas/synchronized_hardware_attestation_v1.schema.json"
        ).read_text(encoding="utf-8")
    )

    assert set(schema["required"]) == ATTESTATION_KEYS
    assert set(schema["properties"]) == ATTESTATION_KEYS
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == 1
    assert (
        schema["properties"]["attestation_type"]["const"]
        == ATTESTATION_TYPE
    )


def test_releases_attested_sidecars_without_claiming_software_verification(
    tmp_path,
):
    sidecars, build_report, manifest = _physical_files(tmp_path)
    attestation = tmp_path / "ATTESTATION.json"
    output = tmp_path / "sensors.jsonl"
    release_report = tmp_path / "HARDWARE_RELEASE_REPORT.json"
    attestation.write_text(
        json.dumps(_attestation(sidecars, build_report, manifest)),
        encoding="utf-8",
    )

    report = release_sidecars(
        sidecars,
        build_report,
        manifest,
        attestation,
        output,
        release_report,
    )

    released = json.loads(output.read_text(encoding="utf-8"))
    _sensor_payload(released, "synchronized-hardware")
    assert released["source"]["physical_hardware_origin_verified"] is False
    assert (
        released["source"]["hardware_origin_attested_by_custodian"] is True
    )
    assert report["physical_hardware_origin_verified_by_software"] is False
    assert report["physical_hardware_origin_attested_by_custodian"] is True
    assert report["hardware_performance_validated"] is False
    assert report["external_blind_release_json_still_required"] is True
    assert json.loads(sidecars.read_text(encoding="utf-8"))["source"][
        "mode"
    ] == "physical-array-v1"


def test_rejects_hash_mismatch_and_false_attestation(tmp_path):
    sidecars, build_report, manifest = _physical_files(tmp_path)
    attestation_path = tmp_path / "ATTESTATION.json"
    output = tmp_path / "sensors.jsonl"
    value = _attestation(sidecars, build_report, manifest)
    value["physical_sidecar_sha256"] = "0" * 64
    attestation_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar hash"):
        release_sidecars(
            sidecars,
            build_report,
            manifest,
            attestation_path,
            output,
        )

    value = _attestation(sidecars, build_report, manifest)
    value["external_manifest_sha256"] = "0" * 64
    attestation_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="external-manifest hash"):
        release_sidecars(
            sidecars,
            build_report,
            manifest,
            attestation_path,
            output,
        )

    value = _attestation(sidecars, build_report, manifest)
    value["hardware_origin_attested_by_custodian"] = False
    attestation_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="must be true"):
        release_sidecars(
            sidecars,
            build_report,
            manifest,
            attestation_path,
            output,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("sample_count", True, "sample_count"),
    ],
)
def test_rejects_boolean_integer_fields(tmp_path, field, value, message):
    sidecars, build_report, manifest = _physical_files(tmp_path)
    attestation_path = tmp_path / "ATTESTATION.json"
    attestation = _attestation(sidecars, build_report, manifest)
    attestation[field] = value
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        release_sidecars(
            sidecars,
            build_report,
            manifest,
            attestation_path,
            tmp_path / "sensors.jsonl",
        )


def test_rejects_manifest_source_image_mismatch(tmp_path):
    sidecars, build_report, manifest = _physical_files(tmp_path)
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "source_image_id": "wrong-image",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    attestation = tmp_path / "ATTESTATION.json"
    attestation.write_text(
        json.dumps(_attestation(sidecars, build_report, manifest)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_image_id"):
        release_sidecars(
            sidecars,
            build_report,
            manifest,
            attestation,
            tmp_path / "sensors.jsonl",
        )


def test_refuses_output_overwrite(tmp_path):
    sidecars, build_report, manifest = _physical_files(tmp_path)
    attestation = tmp_path / "ATTESTATION.json"
    output = tmp_path / "sensors.jsonl"
    output.write_text("occupied\n", encoding="utf-8")
    attestation.write_text(
        json.dumps(_attestation(sidecars, build_report, manifest)),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        release_sidecars(
            sidecars,
            build_report,
            manifest,
            attestation,
            output,
        )


def test_requires_distinct_output_and_report_paths(tmp_path):
    sidecars, build_report, manifest = _physical_files(tmp_path)
    attestation = tmp_path / "ATTESTATION.json"
    output = tmp_path / "combined.json"
    attestation.write_text(
        json.dumps(_attestation(sidecars, build_report, manifest)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be distinct"):
        release_sidecars(
            sidecars,
            build_report,
            manifest,
            attestation,
            output,
            output,
        )
