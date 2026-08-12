from __future__ import annotations

from build_external_sensor_joint_archive import _zip_info


def test_external_sensor_archive_members_are_deterministic() -> None:
    info = _zip_info("reports/example.json")

    assert info.filename == "reports/example.json"
    assert info.date_time == (1980, 1, 1, 0, 0, 0)
    assert info.external_attr == 0o100644 << 16
