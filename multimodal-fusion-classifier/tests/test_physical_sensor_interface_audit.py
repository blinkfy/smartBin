from __future__ import annotations

import pytest

from audit_physical_sensor_interface import build


def test_builds_software_ready_but_hardware_false_audit(tmp_path):
    output = tmp_path / "audit.json"

    report = build(output)

    assert report["valid"] is True
    assert report["software_interface_ready"] is True
    assert report["existing_model_sidecar_compatible"] is True
    assert report["property_ranges_enforced"] is True
    assert report["physical_record_count"] == 0
    assert report["physical_hardware_origin_declared_by_record"] is True
    assert report["physical_hardware_origin_verified_by_software"] is False
    assert report["actual_hardware_acquisition_complete"] is False
    assert report["hardware_validation_ready"] is False
    assert report["fixture"]["hardware_measurement_claimed"] is False
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build(output)
