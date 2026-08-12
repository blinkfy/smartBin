from __future__ import annotations

from build_reliable_external_extension_archive import collect_files


def test_reliable_archive_inventory_excludes_checkpoints_and_images() -> None:
    files = collect_files()
    suffixes = {path.suffix.lower() for path in files}

    assert ".pt" not in suffixes
    assert ".png" not in suffixes
    assert ".jpg" not in suffixes
    assert any(path.name == "SENSOR_SEPARATION.json" for path in files)
    assert any(
        path.name == "RELIABLE_EXTERNAL_EXTENSION_INDEX.json"
        for path in files
    )
