from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from audit_dataset_leakage import (
    aligned_image_similarity,
    audit_manifests,
    perceptual_hash,
    prediction_duplicate_exclusion_sensitivity,
)


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(
    sample_id,
    source_id,
    image_path,
    label="a",
    *,
    annotation_file=None,
    shape_index=None,
):
    record = {
        "sample_id": sample_id,
        "source_image_id": source_id,
        "image_path": str(image_path),
        "label": label,
        "label_id": 0,
    }
    if annotation_file is not None:
        record["annotation_file"] = annotation_file
    if shape_index is not None:
        record["shape_index"] = shape_index
    return record


def test_perceptual_hash_and_aligned_similarity_identify_same_image():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[8:32, 16:48] = 190

    assert perceptual_hash(image) == perceptual_hash(image.copy())
    mae, correlation = aligned_image_similarity(image, image.copy())

    assert mae == pytest.approx(0.0)
    assert correlation == pytest.approx(1.0)


def test_audit_manifests_reports_cross_split_duplicates(tmp_path):
    image_path = tmp_path / "shared.png"
    other_path = tmp_path / "other.png"
    image = np.zeros((64, 64), dtype=np.uint8)
    image[8:32, 16:48] = 190
    Image.fromarray(image).save(image_path)
    Image.fromarray(np.rot90(image).copy()).save(other_path)
    manifests = {
        "train": tmp_path / "train.jsonl",
        "val": tmp_path / "val.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    _write_jsonl(
        manifests["train"],
        [_record("train-1", "source-train", image_path)],
    )
    _write_jsonl(
        manifests["val"],
        [_record("val-1", "source-val", image_path)],
    )
    _write_jsonl(
        manifests["test"],
        [_record("test-1", "source-test", other_path)],
    )

    report, sample_sources, flagged = audit_manifests(manifests)

    assert report["source_image_count"] == 3
    assert report["cross_split_exact_duplicate_group_count"] == 1
    assert report["high_confidence_pair_count"] == 1
    assert flagged == {"source-train", "source-val"}
    assert sample_sources["test-1"] == "source-test"


def test_audit_manifests_rejects_source_crossing_splits(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("L", (16, 16), color=128).save(image_path)
    manifests = {
        "train": tmp_path / "train.jsonl",
        "val": tmp_path / "val.jsonl",
    }
    _write_jsonl(
        manifests["train"],
        [_record("train-1", "shared-source", image_path)],
    )
    _write_jsonl(
        manifests["val"],
        [_record("val-1", "shared-source", image_path)],
    )

    with pytest.raises(ValueError, match="crosses splits"):
        audit_manifests(manifests)


def test_audit_manifests_reports_exact_annotated_object_duplicates(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("L", (16, 16), color=128).save(image_path)
    manifests = {
        "train": tmp_path / "train.jsonl",
        "val": tmp_path / "val.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    duplicate_records = [
        {
            **_record(
                sample_id,
                "source",
                image_path,
                annotation_file=f"{sample_id}.json",
                shape_index=0,
            ),
            "bbox": [1.0, 2.0, 8.0, 9.0],
        }
        for sample_id in ("sample-b", "sample-a")
    ]
    _write_jsonl(manifests["train"], duplicate_records)
    _write_jsonl(manifests["val"], [])
    _write_jsonl(manifests["test"], [])

    report, _, _ = audit_manifests(manifests)

    assert report["sample_count"] == 2
    assert report["unique_annotated_object_count"] == 1
    assert report["exact_object_duplicate_group_count"] == 1
    assert report["exact_object_duplicate_record_count"] == 1
    group = report["exact_object_duplicate_groups"][0]
    assert group["canonical_sample_id"] == "sample-a"
    assert group["duplicate_sample_ids"] == ["sample-b"]


def test_duplicate_prediction_sensitivity_drops_only_noncanonical_record():
    records = [
        {
            "sample_id": "sample-a",
            "target_id": 0,
            "probabilities": [0.9, 0.1],
        },
        {
            "sample_id": "sample-b",
            "target_id": 0,
            "probabilities": [0.4, 0.6],
        },
        {
            "sample_id": "sample-c",
            "target_id": 1,
            "probabilities": [0.1, 0.9],
        },
    ]

    result = prediction_duplicate_exclusion_sensitivity(
        records,
        excluded_sample_ids={"sample-b"},
        class_names=["a", "b"],
    )

    assert result["all"]["sample_count"] == 3
    assert result["after_exact_object_duplicate_exclusion"]["sample_count"] == 2
    assert result["after_exact_object_duplicate_exclusion"]["accuracy"] == 1.0
    assert result["excluded_sample_count"] == 1
