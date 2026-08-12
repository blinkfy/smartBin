from __future__ import annotations

from pathlib import Path

from build_joint_multimodal_seg_archive import _zip_info


def test_joint_archive_members_are_deterministic() -> None:
    info = _zip_info("reports/example.json")

    assert info.filename == "reports/example.json"
    assert info.date_time == (1980, 1, 1, 0, 0, 0)
    assert info.external_attr == 0o100644 << 16
