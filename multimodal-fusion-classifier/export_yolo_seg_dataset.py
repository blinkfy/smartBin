"""Export locked object splits as independent YOLO segmentation metadata."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmfc.training import PROJECT_ROOT, load_config, read_jsonl, resolve_project_path


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从锁定清单和 LabelMe polygon 导出独立 YOLO-seg 元数据"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.json",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "qwen-material",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT.parent / "data" / "Annotations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "yolo-seg-unique-metadata",
    )
    parser.add_argument(
        "--link-mode",
        choices=("none", "symlink"),
        default="none",
        help="默认不创建图像入口；symlink 只创建指向原图的符号链接。",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)


def _class_names(config: Mapping[str, Any]) -> list[str]:
    names = [
        str(item["name"] if isinstance(item, Mapping) else item).strip()
        for item in config.get("classes", [])
    ]
    if not names or any(not name for name in names):
        raise ValueError("config classes must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("config class names must be unique")
    return names


def _normalized_polygon(
    points: Any,
    width: int,
    height: int,
) -> list[float]:
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("polygon must contain at least three points")
    coordinates: list[float] = []
    for point in points:
        if not isinstance(point, Sequence) or len(point) < 2:
            raise ValueError("polygon point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("polygon coordinates must be finite")
        coordinates.extend(
            (
                min(1.0, max(0.0, x / width)),
                min(1.0, max(0.0, y / height)),
            )
        )
    return coordinates


def _resolve_image_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_annotation(
    cache: dict[str, Mapping[str, Any]],
    annotations_dir: Path,
    name: str,
) -> Mapping[str, Any]:
    if name not in cache:
        path = annotations_dir / name
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, Mapping):
            raise ValueError(f"annotation root must be an object: {path}")
        cache[name] = document
    return cache[name]


def _yaml_document(output_dir: Path, class_names: Sequence[str]) -> str:
    names = json.dumps(list(class_names), ensure_ascii=False)
    return "\n".join(
        [
            f"path: {output_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"names: {names}",
            "",
        ]
    )


def export_dataset(
    *,
    config_path: Path,
    manifest_root: Path,
    annotations_dir: Path,
    output_dir: Path,
    link_mode: str = "none",
) -> dict[str, Any]:
    if link_mode not in {"none", "symlink"}:
        raise ValueError("link_mode must be none or symlink")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    config = load_config(config_path)
    class_names = _class_names(config)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    annotations_dir = annotations_dir.resolve()
    manifest_root = manifest_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_source: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    source_split: dict[str, str] = {}
    manifest_paths: dict[str, Path] = {}
    split_input_object_counts: Counter[str] = Counter()
    for split in SPLITS:
        manifest_path = manifest_root / f"{split}.jsonl"
        manifest_paths[split] = manifest_path
        for record in read_jsonl(manifest_path):
            source_id = str(record["source_image_id"])
            previous = source_split.setdefault(source_id, split)
            if previous != split:
                raise ValueError(f"source image crosses splits: {source_id}")
            records_by_source[(split, source_id)].append(record)
            split_input_object_counts[split] += 1

    annotation_cache: dict[str, Mapping[str, Any]] = {}
    class_counts: Counter[str] = Counter()
    split_object_counts: Counter[str] = Counter()
    split_source_counts: Counter[str] = Counter()
    split_duplicate_object_counts: Counter[str] = Counter()
    duplicate_object_groups: list[dict[str, Any]] = []
    link_count = 0
    image_manifest_records: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    for (split, source_id), records in sorted(records_by_source.items()):
        records.sort(
            key=lambda record: (
                int(record["shape_index"]),
                str(record.get("sample_id", "")),
            )
        )
        image_paths = {
            _resolve_image_path(str(record["image_path"])) for record in records
        }
        if len(image_paths) != 1:
            raise ValueError(f"source image maps to multiple paths: {source_id}")
        image_path = next(iter(image_paths))
        if not image_path.is_file():
            raise FileNotFoundError(f"missing source image: {image_path}")
        width = int(records[0]["image_width"])
        height = int(records[0]["image_height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image dimensions for {source_id}")

        lines: list[str] = []
        canonical_by_line: dict[str, dict[str, Any]] = {}
        duplicates_by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            label = str(record["label"])
            if label not in class_to_id:
                raise ValueError(f"unknown class in manifest: {label}")
            annotation_name = str(record["annotation_file"])
            annotation = _load_annotation(
                annotation_cache,
                annotations_dir,
                annotation_name,
            )
            shapes = annotation.get("shapes")
            shape_index = int(record["shape_index"])
            if not isinstance(shapes, list) or not 0 <= shape_index < len(shapes):
                raise ValueError(
                    f"invalid shape index {shape_index} in {annotation_name}"
                )
            shape = shapes[shape_index]
            if not isinstance(shape, Mapping):
                raise ValueError(f"invalid shape in {annotation_name}")
            shape_label = str(shape.get("label", "")).strip()
            if shape_label != label:
                raise ValueError(
                    f"manifest/annotation label mismatch: {label} != {shape_label}"
                )
            polygon = _normalized_polygon(shape.get("points"), width, height)
            line = " ".join(
                [
                    str(class_to_id[label]),
                    *(f"{value:.8f}" for value in polygon),
                ]
            )
            if line in canonical_by_line:
                duplicates_by_line[line].append(record)
                split_duplicate_object_counts[split] += 1
                continue
            canonical_by_line[line] = record
            lines.append(line)
            class_counts[label] += 1
            split_object_counts[split] += 1
        for line, duplicate_records in duplicates_by_line.items():
            canonical = canonical_by_line[line]
            duplicate_object_groups.append(
                {
                    "split": split,
                    "source_image_id": source_id,
                    "annotation_file": str(canonical["annotation_file"]),
                    "shape_index": int(canonical["shape_index"]),
                    "label": str(canonical["label"]),
                    "canonical_sample_id": str(
                        canonical.get("sample_id", "")
                    ),
                    "duplicate_sample_ids": [
                        str(record.get("sample_id", ""))
                        for record in duplicate_records
                    ],
                    "duplicate_annotation_files": [
                        str(record["annotation_file"])
                        for record in duplicate_records
                    ],
                    "duplicate_shape_indices": [
                        int(record["shape_index"])
                        for record in duplicate_records
                    ],
                    "input_record_count": 1 + len(duplicate_records),
                    "removed_record_count": len(duplicate_records),
                }
            )

        label_path = output_dir / "labels" / split / f"{source_id}.txt"
        _atomic_write_text(label_path, "\n".join(lines) + "\n")
        linked_path = (
            output_dir / "images" / split / f"{source_id}{image_path.suffix.lower()}"
        )
        if link_mode == "symlink":
            linked_path.parent.mkdir(parents=True, exist_ok=True)
            linked_path.symlink_to(image_path)
            link_count += 1
        image_manifest_records[split].append(
            {
                "source_image_id": source_id,
                "original_image_path": image_path.as_posix(),
                "linked_image_path": (
                    linked_path.as_posix() if link_mode == "symlink" else None
                ),
                "label_path": label_path.as_posix(),
                "input_object_count": len(records),
                "object_count": len(lines),
                "duplicate_object_count": len(records) - len(lines),
            }
        )
        split_source_counts[split] += 1

    for split, records in image_manifest_records.items():
        text = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        _atomic_write_text(
            output_dir / "manifests" / f"{split}.jsonl",
            text,
        )
    _atomic_write_text(
        output_dir / "dataset.yaml",
        _yaml_document(output_dir, class_names),
    )
    report = {
        "schema_version": 2,
        "valid": True,
        "output_dir": output_dir.resolve().as_posix(),
        "link_mode": link_mode,
        "copied_image_count": 0,
        "linked_image_count": link_count,
        "source_image_count": sum(split_source_counts.values()),
        "input_object_count": sum(split_input_object_counts.values()),
        "object_count": sum(split_object_counts.values()),
        "split_source_counts": dict(split_source_counts),
        "split_input_object_counts": dict(split_input_object_counts),
        "split_object_counts": dict(split_object_counts),
        "duplicate_object_group_count": len(duplicate_object_groups),
        "duplicate_object_record_count": sum(
            split_duplicate_object_counts.values()
        ),
        "split_duplicate_object_counts": dict(
            split_duplicate_object_counts
        ),
        "duplicate_object_groups": duplicate_object_groups,
        "class_counts": {
            name: int(class_counts[name]) for name in class_names
        },
        "class_names": class_names,
        "inputs": {
            "config": config_path.resolve().as_posix(),
            "config_sha256": _sha256(config_path),
            "annotations": annotations_dir.as_posix(),
            "manifests": {
                split: {
                    "path": path.resolve().as_posix(),
                    "sha256": _sha256(path),
                }
                for split, path in manifest_paths.items()
            },
        },
        "training_ready": link_mode == "symlink",
        "limitations": [
            "RGB only; no depth channel or hardware sensor synchronization.",
            "Exact duplicate normalized polygon rows are retained in the input "
            "count but removed deterministically from YOLO labels.",
            "This export creates labels and optional links but does not train a detector.",
        ],
    }
    _atomic_write_text(
        output_dir / "export_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> None:
    args = parse_args()
    report = export_dataset(
        config_path=resolve_project_path(args.config),
        manifest_root=resolve_project_path(args.manifest_root),
        annotations_dir=args.annotations.expanduser().resolve(),
        output_dir=resolve_project_path(args.output),
        link_mode=args.link_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
