from __future__ import annotations

import hashlib

from create_run_manifest import inventory, sha256_file


def test_sha256_inventory_is_stable_and_deduplicated(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"multimodal\n")

    record = sha256_file(path)
    records = inventory([path, path])

    assert record["sha256"] == hashlib.sha256(
        b"multimodal\n"
    ).hexdigest()
    assert record["size_bytes"] == len(b"multimodal\n")
    assert len(records) == 1


def test_inventory_rejects_missing_files(tmp_path):
    missing = tmp_path / "missing.txt"

    try:
        inventory([missing])
    except FileNotFoundError as error:
        assert str(missing) in str(error)
    else:
        raise AssertionError("missing file was accepted")
