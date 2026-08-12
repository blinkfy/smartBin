"""Render deterministic visual QA for exported YOLO segmentation polygons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageOps

from mmfc.training import PROJECT_ROOT, read_jsonl, resolve_project_path


PALETTE = (
    (220, 55, 45),
    (0, 145, 130),
    (45, 105, 190),
    (235, 155, 30),
    (145, 75, 170),
    (80, 150, 65),
    (210, 80, 130),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制 YOLO-seg polygon 固定抽样核验图"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "yolo-seg-unique-metadata",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "segmentation_export_samples.png"
        ),
    )
    return parser.parse_args()


def _read_label(path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    polygons: list[tuple[int, list[tuple[float, float]]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        class_id = int(tokens[0])
        coordinates = [float(value) for value in tokens[1:]]
        points = list(zip(coordinates[0::2], coordinates[1::2]))
        polygons.append((class_id, points))
    return polygons


def _select_records(
    records: Sequence[dict[str, Any]],
    max_images: int,
) -> list[tuple[dict[str, Any], list[tuple[int, list[tuple[float, float]]]]]]:
    prepared = [
        (record, _read_label(Path(str(record["label_path"]))))
        for record in records
    ]
    selected: list[
        tuple[dict[str, Any], list[tuple[int, list[tuple[float, float]]]]]
    ] = []
    selected_ids: set[str] = set()
    seen_classes: set[int] = set()
    for record, polygons in prepared:
        classes = {class_id for class_id, _ in polygons}
        if classes - seen_classes:
            selected.append((record, polygons))
            selected_ids.add(str(record["source_image_id"]))
            seen_classes.update(classes)
            if len(selected) >= max_images:
                return selected
    for record, polygons in prepared:
        source_id = str(record["source_image_id"])
        if source_id not in selected_ids:
            selected.append((record, polygons))
            selected_ids.add(source_id)
            if len(selected) >= max_images:
                break
    return selected


def render_export_samples(
    *,
    dataset_dir: Path,
    split: str,
    output: Path,
    max_images: int = 12,
    columns: int = 4,
) -> dict[str, Any]:
    if max_images <= 0:
        raise ValueError("max_images must be greater than zero")
    if columns <= 0:
        raise ValueError("columns must be greater than zero")
    dataset_dir = dataset_dir.resolve()
    report = json.loads(
        (dataset_dir / "export_report.json").read_text(encoding="utf-8")
    )
    class_names = [str(name) for name in report["class_names"]]
    records = read_jsonl(dataset_dir / "manifests" / f"{split}.jsonl")
    selected = _select_records(records, max_images)
    if not selected:
        raise ValueError(f"split has no records: {split}")

    cell_width, cell_height = 380, 310
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new(
        "RGB",
        (cell_width * columns, cell_height * rows),
        (248, 248, 246),
    )
    draw = ImageDraw.Draw(canvas)
    selected_summary: list[dict[str, Any]] = []
    for index, (record, polygons) in enumerate(selected):
        column, row = index % columns, index // columns
        cell_x, cell_y = column * cell_width, row * cell_height
        image_path = Path(str(record["original_image_path"]))
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        thumbnail = ImageOps.contain(
            image,
            (cell_width - 20, cell_height - 55),
            Image.Resampling.BILINEAR,
        )
        image_x = cell_x + (cell_width - thumbnail.width) // 2
        image_y = cell_y + 36
        canvas.paste(thumbnail, (image_x, image_y))
        labels: list[str] = []
        for class_id, points in polygons:
            color = PALETTE[class_id % len(PALETTE)]
            scaled = [
                (
                    image_x + x * thumbnail.width,
                    image_y + y * thumbnail.height,
                )
                for x, y in points
            ]
            if len(scaled) >= 3:
                draw.line(
                    [*scaled, scaled[0]],
                    fill=color,
                    width=3,
                    joint="curve",
                )
            labels.append(class_names[class_id])
        source_id = str(record["source_image_id"])
        draw.text(
            (cell_x + 10, cell_y + 7),
            f"{source_id[:10]} | {', '.join(sorted(set(labels)))}",
            fill=(30, 30, 30),
        )
        draw.rectangle(
            (
                image_x,
                image_y,
                image_x + thumbnail.width,
                image_y + thumbnail.height,
            ),
            outline=(80, 80, 80),
            width=1,
        )
        selected_summary.append(
            {
                "source_image_id": source_id,
                "original_image_path": image_path.as_posix(),
                "labels": sorted(set(labels)),
                "object_count": len(polygons),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": dataset_dir.as_posix(),
                "split": split,
                "selection": "deterministic_greedy_class_coverage",
                "selected": selected_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "output": output.resolve().as_posix(),
        "metadata": metadata_path.resolve().as_posix(),
        "selected_image_count": len(selected_summary),
        "selected_object_count": sum(
            int(item["object_count"]) for item in selected_summary
        ),
    }


def main() -> None:
    args = parse_args()
    result = render_export_samples(
        dataset_dir=resolve_project_path(args.dataset),
        split=args.split,
        output=resolve_project_path(args.output),
        max_images=args.max_images,
        columns=args.columns,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
