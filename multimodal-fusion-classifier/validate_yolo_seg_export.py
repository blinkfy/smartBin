"""Validate independent YOLO segmentation metadata and image references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mmfc.training import PROJECT_ROOT, read_jsonl, resolve_project_path


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验 YOLO-seg 标签、划分、坐标和原图引用"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "yolo-seg-unique-metadata",
    )
    return parser.parse_args()


def _validate_label(
    path: Path,
    class_count: int,
    errors: list[str],
) -> int:
    object_count = 0
    seen_lines: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read label {path}: {exc}")
        return 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        normalized_line = " ".join(line.split())
        if normalized_line in seen_lines:
            errors.append(f"{path}:{line_number}: duplicate polygon label")
        seen_lines.add(normalized_line)
        tokens = line.split()
        if len(tokens) < 7 or len(tokens) % 2 == 0:
            errors.append(
                f"{path}:{line_number}: expected class plus at least 3 xy points"
            )
            continue
        try:
            class_id = int(tokens[0])
        except ValueError:
            errors.append(f"{path}:{line_number}: class id is not an integer")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{path}:{line_number}: class id out of range")
        try:
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: coordinate is not numeric")
            continue
        if any(not math.isfinite(value) for value in coordinates):
            errors.append(f"{path}:{line_number}: coordinate is not finite")
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            errors.append(f"{path}:{line_number}: coordinate is outside [0,1]")
        object_count += 1
    return object_count


def validate_export(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    errors: list[str] = []
    report_path = dataset_dir / "export_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "error_count": 1,
            "errors": [f"cannot read export report: {exc}"],
        }
    class_names = report.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        errors.append("export report class_names must be a non-empty list")
        class_names = []
    expected_sources: set[tuple[str, str]] = set()
    seen_source_ids: set[str] = set()
    total_sources = 0
    total_objects = 0
    for split in SPLITS:
        manifest_path = dataset_dir / "manifests" / f"{split}.jsonl"
        try:
            records = read_jsonl(manifest_path)
        except (OSError, ValueError) as exc:
            errors.append(f"cannot read {split} manifest: {exc}")
            continue
        for record in records:
            source_id = str(record.get("source_image_id", ""))
            key = (split, source_id)
            if not source_id or key in expected_sources:
                errors.append(f"{split} manifest has missing/duplicate source id")
                continue
            if source_id in seen_source_ids:
                errors.append(f"source image crosses splits: {source_id}")
            expected_sources.add(key)
            seen_source_ids.add(source_id)
            original = Path(str(record.get("original_image_path", "")))
            if not original.is_file():
                errors.append(f"missing original image: {original}")
            label = Path(str(record.get("label_path", "")))
            expected_label = (
                dataset_dir / "labels" / split / f"{source_id}.txt"
            )
            if label.resolve() != expected_label.resolve():
                errors.append(f"unexpected label path for {source_id}: {label}")
            if not label.is_file():
                errors.append(f"missing label: {label}")
                continue
            counted = _validate_label(label, len(class_names), errors)
            declared = int(record.get("object_count", -1))
            if counted != declared:
                errors.append(
                    f"{label}: object count {counted} != declared {declared}"
                )
            total_objects += counted
            linked_value = record.get("linked_image_path")
            if report.get("link_mode") == "none":
                if linked_value is not None:
                    errors.append(
                        f"link_mode=none but linked path is set for {source_id}"
                    )
            else:
                linked = Path(str(linked_value or ""))
                if not linked.is_symlink():
                    errors.append(f"missing image symlink: {linked}")
                elif linked.resolve() != original.resolve():
                    errors.append(f"image symlink target mismatch: {linked}")
            total_sources += 1

    expected_label_paths = {
        (
            dataset_dir / "labels" / split / f"{source_id}.txt"
        ).resolve()
        for split, source_id in expected_sources
    }
    actual_label_paths = {
        path.resolve()
        for path in (dataset_dir / "labels").glob("*/*.txt")
    }
    for unexpected in sorted(actual_label_paths - expected_label_paths):
        errors.append(f"unexpected label file: {unexpected}")
    if total_sources != int(report.get("source_image_count", -1)):
        errors.append(
            "source image count mismatch: "
            f"{total_sources} != {report.get('source_image_count')}"
        )
    if total_objects != int(report.get("object_count", -1)):
        errors.append(
            f"object count mismatch: {total_objects} != {report.get('object_count')}"
        )
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "source_image_count": total_sources,
        "input_object_count": int(
            report.get("input_object_count", total_objects)
        ),
        "object_count": total_objects,
        "duplicate_object_group_count": int(
            report.get("duplicate_object_group_count", 0)
        ),
        "duplicate_object_record_count": int(
            report.get("duplicate_object_record_count", 0)
        ),
        "class_count": len(class_names),
        "copied_image_count": int(report.get("copied_image_count", -1)),
        "link_mode": report.get("link_mode"),
        "dataset": dataset_dir.as_posix(),
    }


def main() -> None:
    result = validate_export(resolve_project_path(parse_args().dataset))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
