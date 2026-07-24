from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from build_dataset import _parser as build_parser
from build_dataset import build
from mmfc.data import (
    parse_labelme_annotations,
    polygon_to_bbox,
    read_jsonl,
    resolve_manifest_path,
    split_samples,
)
from validate_dataset import validate_generated


def _write_fixture(
    annotations: Path,
    images: Path,
    name: str,
    *,
    shapes: list[dict] | None = None,
    with_image: bool = True,
) -> None:
    if with_image:
        Image.new("RGB", (40, 30), (100, 120, 140)).save(images / f"{name}.jpg")
    payload = {
        "version": "5.5.0",
        "imagePath": f"..\\old_folder\\{name}.jpg",
        "imageWidth": 40,
        "imageHeight": 30,
        "shapes": shapes
        if shapes is not None
        else [
            {
                "label": "object",
                "shape_type": "polygon",
                "points": [[2, 3], [20, 3], [20, 25], [2, 25]],
            }
        ],
    }
    (annotations / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _write_config(path: Path, labels: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "classes": [
                    {"name": label, "description": f"{label} description"}
                    for label in labels
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _build_fixture_dataset(
    tmp_path: Path, *, annotation_label: str = "potato"
) -> tuple[Path, Path]:
    annotations = tmp_path / "Annotations"
    images = tmp_path / "images"
    output = tmp_path / "generated"
    annotations.mkdir()
    images.mkdir()
    _write_fixture(
        annotations,
        images,
        "sample",
        shapes=[
            {
                "label": annotation_label,
                "shape_type": "polygon",
                "points": [[2, 3], [20, 3], [20, 25], [2, 25]],
            }
        ],
    )
    config = tmp_path / "default.json"
    _write_config(config, ["potato", "rabbitcut", "battery1"])
    args = build_parser().parse_args(
        [
            "--annotations",
            str(annotations),
            "--images",
            str(images),
            "--output",
            str(output),
            "--config",
            str(config),
        ]
    )
    build(args)
    return output, config


def test_polygon_bbox_is_clamped_and_rejects_invalid() -> None:
    assert polygon_to_bbox([[-2, 3], [8, 2], [12, 20]], 10, 12) == [
        0.0,
        2.0,
        10.0,
        12.0,
    ]
    assert polygon_to_bbox([[1, 1], [2, 2]], 10, 10) is None
    assert polygon_to_bbox([[1, 1], [1, 2], [1, 3]], 10, 10) is None


def test_parse_labelme_reports_missing_and_invalid_shapes(tmp_path: Path) -> None:
    annotations = tmp_path / "Annotations"
    images = tmp_path / "images"
    project = tmp_path / "project"
    annotations.mkdir()
    images.mkdir()
    project.mkdir()
    _write_fixture(
        annotations,
        images,
        "valid",
        shapes=[
            {
                "label": "纸杯",
                "shape_type": "polygon",
                "points": [[2, 3], [20, 3], [20, 25], [2, 25]],
            },
            {
                "label": "bad",
                "shape_type": "polygon",
                "points": [[1, 1], [2, 2]],
            },
            {
                "label": "point",
                "shape_type": "point",
                "points": [[1, 1]],
            },
        ],
    )
    _write_fixture(annotations, images, "missing", with_image=False)

    samples, report = parse_labelme_annotations(
        annotations, images, project_root=project
    )

    assert len(samples) == 1
    assert samples[0]["label"] == "纸杯"
    assert samples[0]["bbox"] == [2.0, 3.0, 20.0, 25.0]
    assert resolve_manifest_path(samples[0]["image_path"], project).is_file()
    assert report["issue_counts"]["invalid_polygon"] == 1
    assert report["issue_counts"]["unsupported_shape_type"] == 1
    assert report["issue_counts"]["missing_image"] == 1


def test_grouped_split_is_deterministic_and_has_no_leakage() -> None:
    samples = []
    for source_number in range(30):
        for shape_number in range(2):
            samples.append(
                {
                    "sample_id": f"{source_number}-{shape_number}",
                    "source_image_id": f"source-{source_number}",
                }
            )
    first = split_samples(samples, seed=7)
    second = split_samples(reversed(samples), seed=7)
    first_pairs = {
        sample["sample_id"]: split
        for split, records in first.items()
        for sample in records
    }
    second_pairs = {
        sample["sample_id"]: split
        for split, records in second.items()
        for sample in records
    }
    assert first_pairs == second_pairs
    source_splits: dict[str, set[str]] = {}
    for split, records in first.items():
        for sample in records:
            source_splits.setdefault(sample["source_image_id"], set()).add(split)
    assert all(len(values) == 1 for values in source_splits.values())


def test_build_uses_config_order_for_manifest_and_index(tmp_path: Path) -> None:
    output, config = _build_fixture_dataset(tmp_path)
    manifests = [
        sample
        for split in ("train", "val", "test")
        for sample in read_jsonl(output / f"{split}.jsonl")
    ]
    assert len(manifests) == 1
    assert manifests[0]["label"] == "potato"
    assert manifests[0]["label_id"] == 0

    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["labels"] == ["potato", "rabbitcut", "battery1"]
    assert list(index["label_to_index"]) == [
        "potato",
        "rabbitcut",
        "battery1",
    ]
    assert index["label_to_index"]["potato"] == 0
    assert validate_generated(output, config_path=config)["valid"]


def test_build_rejects_annotation_label_missing_from_config(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "Annotations"
    images = tmp_path / "images"
    output = tmp_path / "generated"
    annotations.mkdir()
    images.mkdir()
    _write_fixture(
        annotations,
        images,
        "unknown",
        shapes=[
            {
                "label": "unknown-class",
                "shape_type": "polygon",
                "points": [[2, 3], [20, 3], [20, 25], [2, 25]],
            }
        ],
    )
    config = tmp_path / "default.json"
    _write_config(config, ["potato"])
    args = build_parser().parse_args(
        [
            "--annotations",
            str(annotations),
            "--images",
            str(images),
            "--output",
            str(output),
            "--config",
            str(config),
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="unknown-class"):
        build(args)
    assert not output.exists()


def test_validator_rejects_index_mapping_order_mismatch(tmp_path: Path) -> None:
    output, config = _build_fixture_dataset(tmp_path)
    index_path = output / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["labels"] = ["rabbitcut", "potato", "battery1"]
    index["label_to_index"] = {
        "rabbitcut": 0,
        "potato": 1,
        "battery1": 2,
    }
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = validate_generated(output, config_path=config)
    assert not report["valid"]
    assert any("配置类别顺序不一致" in error for error in report["errors"])
    assert any("配置类别映射不一致" in error for error in report["errors"])
