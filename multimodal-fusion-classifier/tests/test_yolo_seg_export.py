from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from export_yolo_seg_dataset import export_dataset
from plot_segmentation_export import render_export_samples
from validate_yolo_seg_export import validate_export


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_yolo_seg_export_preserves_locked_split_without_copying_images(
    tmp_path: Path,
) -> None:
    image = tmp_path / "source" / "sample.jpg"
    image.parent.mkdir()
    Image.new("RGB", (20, 10), "white").save(image)
    annotation_dir = tmp_path / "annotations"
    _write_json(
        annotation_dir / "sample.json",
        {
            "shapes": [
                {
                    "label": "paper",
                    "shape_type": "polygon",
                    "points": [[2, 1], [18, 1], [18, 9], [2, 9]],
                }
            ]
        },
    )
    config = tmp_path / "config.json"
    _write_json(config, {"classes": [{"name": "paper"}]})
    manifest_root = tmp_path / "manifests"
    record = {
        "sample_id": "object-1",
        "source_image_id": "source-1",
        "image_path": image.as_posix(),
        "image_width": 20,
        "image_height": 10,
        "label": "paper",
        "shape_index": 0,
        "annotation_file": "sample.json",
    }
    _write_jsonl(manifest_root / "train.jsonl", [record])
    _write_jsonl(manifest_root / "val.jsonl", [])
    _write_jsonl(manifest_root / "test.jsonl", [])

    output = tmp_path / "export"
    report = export_dataset(
        config_path=config,
        manifest_root=manifest_root,
        annotations_dir=annotation_dir,
        output_dir=output,
    )

    assert report["valid"] is True
    assert report["copied_image_count"] == 0
    assert report["linked_image_count"] == 0
    assert report["source_image_count"] == 1
    assert report["object_count"] == 1
    label = (output / "labels" / "train" / "source-1.txt").read_text(
        encoding="utf-8"
    )
    assert label.startswith("0 0.10000000 0.10000000")
    assert not (output / "images").exists()
    assert "images/train" in (output / "dataset.yaml").read_text(
        encoding="utf-8"
    )
    validation = validate_export(output)
    assert validation["valid"] is True
    assert validation["object_count"] == 1
    plot = render_export_samples(
        dataset_dir=output,
        split="train",
        output=tmp_path / "samples.png",
        max_images=1,
        columns=1,
    )
    assert plot["selected_image_count"] == 1
    assert (tmp_path / "samples.png").stat().st_size > 1000

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_dataset(
            config_path=config,
            manifest_root=manifest_root,
            annotations_dir=annotation_dir,
            output_dir=output,
        )

    (output / "labels" / "train" / "source-1.txt").write_text(
        "0 1.5 0.0 0.0 0.0 1.0 1.0\n",
        encoding="utf-8",
    )
    invalid = validate_export(output)
    assert invalid["valid"] is False
    assert any("outside [0,1]" in error for error in invalid["errors"])


def test_yolo_seg_export_rejects_source_crossing_splits(
    tmp_path: Path,
) -> None:
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    annotation_dir = tmp_path / "annotations"
    _write_json(
        annotation_dir / "sample.json",
        {
            "shapes": [
                {
                    "label": "paper",
                    "points": [[1, 1], [9, 1], [9, 9]],
                }
            ]
        },
    )
    config = tmp_path / "config.json"
    _write_json(config, {"classes": [{"name": "paper"}]})
    record = {
        "source_image_id": "same-source",
        "image_path": image.as_posix(),
        "image_width": 10,
        "image_height": 10,
        "label": "paper",
        "shape_index": 0,
        "annotation_file": "sample.json",
    }
    manifest_root = tmp_path / "manifests"
    _write_jsonl(manifest_root / "train.jsonl", [record])
    _write_jsonl(manifest_root / "val.jsonl", [record])
    _write_jsonl(manifest_root / "test.jsonl", [])

    with pytest.raises(ValueError, match="crosses splits"):
        export_dataset(
            config_path=config,
            manifest_root=manifest_root,
            annotations_dir=annotation_dir,
            output_dir=tmp_path / "export",
        )


def test_yolo_seg_export_deduplicates_identical_annotated_object_records(
    tmp_path: Path,
) -> None:
    image = tmp_path / "source.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    annotation_dir = tmp_path / "annotations"
    _write_json(
        annotation_dir / "sample.json",
        {
            "shapes": [
                {
                    "label": "paper",
                    "points": [[1, 1], [9, 1], [9, 9]],
                }
            ]
        },
    )
    config = tmp_path / "config.json"
    _write_json(config, {"classes": [{"name": "paper"}]})
    manifest_root = tmp_path / "manifests"
    records = [
        {
            "sample_id": sample_id,
            "source_image_id": "source",
            "image_path": image.as_posix(),
            "image_width": 10,
            "image_height": 10,
            "label": "paper",
            "shape_index": 0,
            "annotation_file": "sample.json",
        }
        for sample_id in ("sample-b", "sample-a")
    ]
    _write_jsonl(manifest_root / "train.jsonl", records)
    _write_jsonl(manifest_root / "val.jsonl", [])
    _write_jsonl(manifest_root / "test.jsonl", [])

    output = tmp_path / "export"
    report = export_dataset(
        config_path=config,
        manifest_root=manifest_root,
        annotations_dir=annotation_dir,
        output_dir=output,
    )

    assert report["schema_version"] == 2
    assert report["input_object_count"] == 2
    assert report["object_count"] == 1
    assert report["duplicate_object_group_count"] == 1
    assert report["duplicate_object_record_count"] == 1
    assert (
        report["duplicate_object_groups"][0]["canonical_sample_id"]
        == "sample-a"
    )
    label_lines = (
        output / "labels" / "train" / "source.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert len(label_lines) == 1
    assert validate_export(output)["valid"] is True
