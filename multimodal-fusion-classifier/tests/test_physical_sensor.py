from __future__ import annotations

import json

import pytest

from mmfc.external_blind import _sensor_payload
from mmfc.physical_sensor import calibrate_physical_observation
from prepare_physical_sensor_sidecars import build_sidecars


def _grid(value: float):
    return [[value for _ in range(4)] for _ in range(4)]


def _record(sample_id: str = "sample-1"):
    raw = [
        [10.0, 20.0, 30.0, 40.0],
        [50.0, 60.0, 70.0, 80.0],
        [90.0, 100.0, 110.0, 10.0],
        [20.0, 30.0, 40.0, 50.0],
    ]
    return {
        "sample_id": sample_id,
        "source_image_id": "source-1",
        "sensor_id": "array-A",
        "measurement_unit": "mV",
        "image_timestamp": "2026-07-29T10:00:00.000Z",
        "sensor_timestamp": "2026-07-29T10:00:00.030Z",
        "raw_grid": raw,
        "properties": {
            "weight_g": 123.4,
            "temperature_c": 24.0,
            "moisture_pct": 17.0,
            "conductivity_us_cm": 44.0,
            "odor_index": 0.2,
        },
        "calibration": {
            "calibration_id": "cal-2026-07",
            "calibrated_at": "2026-07-01T00:00:00Z",
            "zero_grid": _grid(10.0),
            "full_scale_grid": _grid(110.0),
        },
    }


def test_calibrates_to_existing_sidecar_contract():
    result = calibrate_physical_observation(_record())

    assert result["sample_id"] == "sample-1"
    assert result["grid_shape"] == [4, 4]
    assert result["normalized_grid"][0] == [0.0, 0.1, 0.2, 0.3]
    assert result["normalized_grid"][2][2] == 1.0
    assert result["source"]["mode"] == "physical-array-v1"
    assert result["source"]["time_skew_ms"] == pytest.approx(30.0)
    assert result["source"]["physical_hardware_origin_declared"] is True
    assert result["source"]["physical_hardware_origin_verified"] is False
    assert result["source"]["annotation_inputs"] is False
    assert result["source"]["class_inputs"] is False
    assert len(result["source"]["raw_grid_sha256"]) == 64
    assert len(result["source"]["calibration_sha256"]) == 64
    _sensor_payload(result, "physical-array-v1")


def test_reports_clipping_without_silently_hiding_it():
    record = _record()
    record["raw_grid"][0][0] = 0.0
    record["raw_grid"][0][1] = 120.0

    result = calibrate_physical_observation(record)

    assert result["normalized_grid"][0][:2] == [0.0, 1.0]
    assert result["source"]["clipped_cell_count"] == 2
    assert result["source"]["clipping_fraction"] == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda record: record.update(
                {"sensor_timestamp": "2026-07-29T10:00:01Z"}
            ),
            "time skew",
        ),
        (
            lambda record: record.update({"label": "battery"}),
            "supervision fields",
        ),
        (
            lambda record: record.update({"freeform_metadata": {}}),
            "unexpected fields",
        ),
        (
            lambda record: record["calibration"].update(
                {"full_scale_grid": _grid(10.0)}
            ),
            "must exceed",
        ),
        (
            lambda record: record.update(
                {"image_timestamp": "2026-07-29T10:00:00"}
            ),
            "timezone",
        ),
    ],
)
def test_rejects_unsynchronized_or_leaking_observations(mutation, match):
    record = _record()
    mutation(record)

    with pytest.raises(ValueError, match=match):
        calibrate_physical_observation(record)


def test_builds_atomic_jsonl_and_auditable_report(tmp_path):
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "sensors.jsonl"
    report_path = tmp_path / "BUILD_REPORT.json"
    records = [_record("sample-1"), _record("sample-2")]
    records[1]["source_image_id"] = "source-2"
    input_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    report = build_sidecars(
        input_path,
        output_path,
        report_path,
        max_skew_ms=50.0,
    )

    assert report["valid"] is True
    assert report["sample_count"] == 2
    assert report["observed_max_skew_ms"] == pytest.approx(30.0)
    assert report["physical_hardware_origin_declared"] is True
    assert report["physical_hardware_origin_verified"] is False
    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in output_rows] == [
        "sample-1",
        "sample-2",
    ]
    assert report_path.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_sidecars(input_path, output_path, report_path)


def test_build_rejects_duplicate_sample_ids(tmp_path):
    input_path = tmp_path / "raw.jsonl"
    records = [_record("duplicate"), _record("duplicate")]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_sidecars(
            input_path,
            tmp_path / "sensors.jsonl",
            tmp_path / "report.json",
        )


def test_rejects_unregistered_nested_fields():
    record = _record()
    record["properties"]["label_hint"] = 1.0
    with pytest.raises(ValueError, match="properties contains unexpected"):
        calibrate_physical_observation(record)

    record = _record()
    record["calibration"]["operator_label"] = "battery"
    with pytest.raises(ValueError, match="calibration contains unexpected"):
        calibrate_physical_observation(record)


def test_rejects_property_values_outside_existing_project_ranges():
    record = _record()
    record["properties"]["moisture_pct"] = 101.0

    with pytest.raises(ValueError, match=r"moisture_pct must be in"):
        calibrate_physical_observation(record)
