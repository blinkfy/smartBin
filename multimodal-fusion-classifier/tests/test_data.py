from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import pytest
from PIL import Image

import build_dataset
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


def _build_many_args(
    tmp_path: Path,
    *,
    count: int = 4,
    sensor_mode: str = "qwen",
    output: Path | None = None,
    extra: list[str] | None = None,
) -> argparse.Namespace:
    annotations = tmp_path / "Annotations"
    images = tmp_path / "images"
    annotations.mkdir(exist_ok=True)
    images.mkdir(exist_ok=True)
    for index in range(count):
        _write_fixture(
            annotations,
            images,
            f"sample_{index:02d}",
            shapes=[
                {
                    "label": "potato",
                    "shape_type": "polygon",
                    "points": [[2, 3], [20, 3], [20, 25], [2, 25]],
                }
            ],
        )
    config = tmp_path / "default.json"
    _write_config(config, ["potato"])
    output = output or (tmp_path / "generated")
    argv = [
        "--annotations",
        str(annotations),
        "--images",
        str(images),
        "--output",
        str(output),
        "--config",
        str(config),
        "--sensor-mode",
        sensor_mode,
        "--workers",
        "2",
        "--progress-every",
        "1",
        "--checkpoint-every",
        "1",
    ]
    if extra:
        argv.extend(extra)
    return build_parser().parse_args(argv)


def test_build_parser_rejects_invalid_workers_and_uses_safe_env_name() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.api_key_env == "DASHSCOPE_API_KEY"
    assert args.workers == 4
    assert not args.api_key_env.startswith("sk-")

    for value in ("0", "33"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--workers", value])


def test_qwen_build_runs_workers_concurrently_and_keeps_sorted_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5.0)

    class FakeEnhancer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def enhance(self, image, bbox, rule_record):
            with lock:
                state["active"] += 1
                state["max_active"] = max(
                    state["max_active"], state["active"]
                )
            try:
                barrier.wait()
            finally:
                with lock:
                    state["active"] -= 1
            return {
                **rule_record,
                "source": {
                    **rule_record["source"],
                    "generator": self.model,
                    "model": self.model,
                    "enhancement_status": "success",
                    "attempts": 1,
                },
            }

    monkeypatch.setattr(build_dataset, "qwen_api_key", lambda env_name: "fake")
    monkeypatch.setattr(build_dataset, "QwenSensorEnhancer", FakeEnhancer)

    args = _build_many_args(
        tmp_path,
        count=4,
        sensor_mode="qwen",
        extra=["--checkpoint-every", "2"],
    )
    stats = build(args)

    assert state["max_active"] >= 2
    assert stats["qwen_success_records"] == 4
    sensors = read_jsonl(args.output / "sensors.jsonl")
    sample_ids = [record["sample_id"] for record in sensors]
    assert sample_ids == sorted(sample_ids)
    assert all(
        record["source"]["enhancement_status"] == "success"
        for record in sensors
    )
    progress = capsys.readouterr().err
    assert "100.0%" in progress
    assert "qwen_success=4" in progress
    assert "fallback=0" in progress
    assert "errors=0" in progress


def test_qwen_resume_reuses_success_records_without_new_enhance_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    class FakeEnhancer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def enhance(self, image, bbox, rule_record):
            calls["count"] += 1
            return {
                **rule_record,
                "source": {
                    **rule_record["source"],
                    "generator": self.model,
                    "model": self.model,
                    "enhancement_status": "success",
                    "attempts": 1,
                },
            }

    monkeypatch.setattr(build_dataset, "qwen_api_key", lambda env_name: "fake")
    monkeypatch.setattr(build_dataset, "QwenSensorEnhancer", FakeEnhancer)
    output = tmp_path / "generated"
    first = _build_many_args(tmp_path, count=3, sensor_mode="qwen", output=output)
    build(first)
    assert calls["count"] == 3

    second = _build_many_args(
        tmp_path,
        count=3,
        sensor_mode="qwen",
        output=output,
        extra=["--resume"],
    )
    stats = build(second)

    assert calls["count"] == 3
    assert stats["resumed_sensor_records"] == 3
    assert stats["qwen_success_records"] == 3


def test_periodic_sensor_checkpoint_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write_jsonl = build_dataset.write_jsonl
    checkpoint_sizes: list[int] = []

    def spy_write_jsonl(path, records):
        original_write_jsonl(path, records)
        if Path(path).name == "sensors.jsonl":
            checkpoint_sizes.append(len(read_jsonl(path)))

    monkeypatch.setattr(build_dataset, "write_jsonl", spy_write_jsonl)
    args = _build_many_args(tmp_path, count=3, sensor_mode="rule")
    stats = build(args)

    assert any(0 < size < stats["samples"] for size in checkpoint_sizes)
    assert checkpoint_sizes[-1] == stats["samples"]


def test_qwen_build_without_key_uses_fallback_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    args = _build_many_args(tmp_path, count=2, sensor_mode="qwen")

    stats = build(args)

    assert stats["qwen_api_key_available"] is False
    assert stats["qwen_success_records"] == 0
    assert stats["fallback_sensor_records"] == 2
    sensors = read_jsonl(args.output / "sensors.jsonl")
    assert {
        record["source"]["enhancement_status"] for record in sensors
    } == {"fallback_no_api_key"}


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
