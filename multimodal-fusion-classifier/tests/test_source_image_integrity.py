import json
from pathlib import Path

from PIL import Image

from audit_image_recovery import audit_image_recovery
from audit_source_image_integrity import audit_source_image_integrity


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_source_image_inventory_is_deterministic_and_filename_free(
    tmp_path,
):
    project = tmp_path / "project"
    source = tmp_path / "data" / "images" / "damaged.jpg"
    fallback = tmp_path / "data" / "combination" / "damaged.jpg"
    healthy = tmp_path / "data" / "images" / "healthy.jpg"
    source.parent.mkdir(parents=True)
    fallback.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(source)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(fallback)
    Image.new("RGB", (8, 8), (40, 50, 60)).save(healthy)
    manifests = []
    rows_by_split = {
        "train": [
            {
                "sample_id": "sample-a",
                "source_image_id": "source-a",
                "split": "train",
                "image_path": str(source),
            }
        ],
        "val": [
            {
                "sample_id": "sample-b",
                "source_image_id": "source-b",
                "split": "val",
                "image_path": str(healthy),
            }
        ],
        "test": [],
    }
    for split, rows in rows_by_split.items():
        path = project / "data" / f"{split}.jsonl"
        _write_manifest(path, rows)
        manifests.append(path)
    shim = project / "runtime_recovery" / "sitecustomize.py"
    shim.parent.mkdir(parents=True)
    shim.write_text("shim", encoding="utf-8")
    recovery = audit_image_recovery(
        source=source,
        fallback=fallback,
        manifests=manifests,
        shim=shim,
    )
    # The generic recovery audit is intentionally invalid because source is
    # healthy; create the minimal valid binding required by this inventory.
    recovery["valid"] = True
    recovery_path = project / "recovery.json"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")

    first = audit_source_image_integrity(
        manifests=manifests,
        recovery_audit_path=recovery_path,
    )
    second = audit_source_image_integrity(
        manifests=manifests,
        recovery_audit_path=recovery_path,
    )

    assert first == second
    assert first["valid"] is False
    assert first["summary"]["source_image_count"] == 2
    assert first["summary"]["decode_failure_count"] == 0
    assert first["summary"]["documented_recovery_count"] == 0
    assert all("path" not in row for row in first["images"])
    assert (
        first["summary"]["raw_inventory_sha256"]
        == first["summary"]["effective_inventory_sha256"]
    )

    repaired = audit_source_image_integrity(
        manifests=manifests,
        recovery_audit_path=recovery_path,
        expected_recovery_state="repaired",
    )
    assert repaired["valid"] is True
    assert repaired["expected_recovery_state"] == "repaired"
