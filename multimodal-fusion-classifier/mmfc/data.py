"""Dataset parsing and deterministic source-image grouped splitting.

The generated manifests contain object-level references to the original images.
No image is copied or modified by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent / "data"
DEFAULT_ANNOTATIONS_DIR = DEFAULT_SOURCE_ROOT / "Annotations"
DEFAULT_IMAGES_DIR = DEFAULT_SOURCE_ROOT / "images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"
SPLITS = ("train", "val", "test")


def _stable_id(namespace: str, *parts: object) -> str:
    value = "\0".join([namespace, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _portable_path(path: Path, base_dir: Path) -> str:
    """Return a forward-slash path relative to ``base_dir`` when possible."""
    try:
        return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()
    except (OSError, ValueError):
        return path.resolve().as_posix()


def resolve_manifest_path(path_value: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a manifest image path without relying on the process CWD."""
    path = Path(path_value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _image_catalog(images_dir: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    if not images_dir.is_dir():
        return by_name, by_stem
    for path in sorted(images_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_file() and path.suffix.casefold() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }:
            by_name[path.name.casefold()].append(path)
            by_stem[path.stem.casefold()].append(path)
    return by_name, by_stem


def _find_image(
    annotation_path: Path,
    image_path_value: object,
    by_name: Mapping[str, list[Path]],
    by_stem: Mapping[str, list[Path]],
) -> tuple[Path | None, str | None]:
    raw_name = str(image_path_value or "").replace("\\", "/").rsplit("/", 1)[-1]
    candidates: list[Path] = []
    if raw_name:
        candidates = list(by_name.get(raw_name.casefold(), ()))
    if not candidates:
        candidates = list(by_stem.get(annotation_path.stem.casefold(), ()))
    if not candidates and raw_name:
        candidates = list(by_stem.get(Path(raw_name).stem.casefold(), ()))
    if not candidates:
        return None, "missing_image"
    unique = sorted(set(candidates), key=lambda item: item.as_posix().casefold())
    if len(unique) > 1:
        return None, "ambiguous_image"
    return unique[0], None


def polygon_to_bbox(
    points: object,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    """Convert a LabelMe polygon to a clamped ``[x1, y1, x2, y2]`` bbox."""
    if not isinstance(points, list) or len(points) < 3:
        return None
    parsed: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        parsed.append((x, y))
    x1 = max(0.0, min(x for x, _ in parsed))
    y1 = max(0.0, min(y for _, y in parsed))
    x2 = min(float(image_width), max(x for x, _ in parsed))
    y2 = min(float(image_height), max(y for _, y in parsed))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(value, 4) for value in (x1, y1, x2, y2)]


def parse_labelme_annotations(
    annotations_dir: Path | str = DEFAULT_ANNOTATIONS_DIR,
    images_dir: Path | str = DEFAULT_IMAGES_DIR,
    *,
    project_root: Path | str = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse LabelMe polygons into deterministic object-level samples.

    Invalid JSON files, missing images, unsupported shapes and invalid polygons
    are reported and skipped instead of aborting the whole build.
    """
    annotations_dir = Path(annotations_dir).resolve()
    images_dir = Path(images_dir).resolve()
    project_root = Path(project_root).resolve()
    by_name, by_stem = _image_catalog(images_dir)
    issues: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    annotation_files = sorted(
        annotations_dir.glob("*.json"), key=lambda item: item.name.casefold()
    )

    for annotation_path in annotation_files:
        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            issues.append(
                {
                    "code": "invalid_json",
                    "annotation": annotation_path.name,
                    "detail": str(exc),
                }
            )
            continue
        if not isinstance(payload, dict):
            issues.append(
                {"code": "invalid_json_root", "annotation": annotation_path.name}
            )
            continue

        image_path, image_error = _find_image(
            annotation_path, payload.get("imagePath"), by_name, by_stem
        )
        if image_path is None:
            issues.append(
                {
                    "code": image_error,
                    "annotation": annotation_path.name,
                    "image_path": payload.get("imagePath"),
                }
            )
            continue
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            issues.append(
                {
                    "code": "unreadable_image",
                    "annotation": annotation_path.name,
                    "image": image_path.name,
                    "detail": str(exc),
                }
            )
            continue
        if width <= 0 or height <= 0:
            issues.append(
                {
                    "code": "invalid_image_dimensions",
                    "annotation": annotation_path.name,
                    "image": image_path.name,
                }
            )
            continue

        relative_source = image_path.relative_to(images_dir).as_posix().casefold()
        source_image_id = _stable_id("source-image", relative_source)
        manifest_image_path = _portable_path(image_path, project_root)
        declared = (payload.get("imageWidth"), payload.get("imageHeight"))
        if declared != (width, height):
            issues.append(
                {
                    "code": "dimension_mismatch",
                    "annotation": annotation_path.name,
                    "declared": list(declared),
                    "actual": [width, height],
                }
            )

        shapes = payload.get("shapes")
        if not isinstance(shapes, list):
            issues.append(
                {"code": "invalid_shapes", "annotation": annotation_path.name}
            )
            continue
        for shape_index, shape in enumerate(shapes):
            issue_base = {
                "annotation": annotation_path.name,
                "shape_index": shape_index,
            }
            if not isinstance(shape, dict):
                issues.append({"code": "invalid_shape", **issue_base})
                continue
            shape_type = shape.get("shape_type", "polygon")
            if shape_type not in (None, "", "polygon"):
                issues.append(
                    {
                        "code": "unsupported_shape_type",
                        "shape_type": shape_type,
                        **issue_base,
                    }
                )
                continue
            label = str(shape.get("label") or "").strip()
            if not label:
                issues.append({"code": "missing_label", **issue_base})
                continue
            bbox = polygon_to_bbox(shape.get("points"), width, height)
            if bbox is None:
                issues.append({"code": "invalid_polygon", **issue_base})
                continue
            # The object ID intentionally excludes its class label. It can safely
            # seed label-independent sensor simulation.
            sample_id = _stable_id(
                "object",
                source_image_id,
                annotation_path.name.casefold(),
                shape_index,
                *[f"{v:.4f}" for v in bbox],
            )
            samples.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "source_image_id": source_image_id,
                    "image_path": manifest_image_path,
                    "image_width": width,
                    "image_height": height,
                    "bbox": bbox,
                    "label": label,
                    "shape_index": shape_index,
                    "annotation_file": annotation_path.name,
                }
            )

    samples.sort(key=lambda item: (item["source_image_id"], item["shape_index"]))
    report = {
        "schema_version": 1,
        "annotations_dir": annotations_dir.as_posix(),
        "images_dir": images_dir.as_posix(),
        "annotation_files": len(annotation_files),
        "samples": len(samples),
        "source_images": len({item["source_image_id"] for item in samples}),
        "class_counts": dict(sorted(Counter(item["label"] for item in samples).items())),
        "issue_counts": dict(sorted(Counter(item["code"] for item in issues).items())),
        "issues": issues,
    }
    return samples, report


def assign_source_split(
    source_image_id: str,
    *,
    seed: int = 20260724,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> str:
    """Assign a source image to a stable split with hash thresholds."""
    if len(ratios) != 3 or any(value < 0 for value in ratios):
        raise ValueError("ratios must be three non-negative values")
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("ratios must have a positive sum")
    normalized = [value / total for value in ratios]
    digest = hashlib.sha256(f"{seed}\0{source_image_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit < normalized[0]:
        return "train"
    if unit < normalized[0] + normalized[1]:
        return "val"
    return "test"


def split_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    seed: int = 20260724,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict[str, list[dict[str, Any]]]:
    """Split samples while keeping every source image in exactly one split."""
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    source_assignments: dict[str, str] = {}
    for original in samples:
        sample = dict(original)
        source_id = str(sample["source_image_id"])
        split = source_assignments.setdefault(
            source_id, assign_source_split(source_id, seed=seed, ratios=ratios)
        )
        sample["split"] = split
        result[split].append(sample)
    for split in SPLITS:
        result[split].sort(key=lambda item: item["sample_id"])
    return result


def write_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL and report malformed line locations."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_json(path: Path | str, value: object) -> None:
    """Atomically write indented UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        )
        handle.write("\n")
    temporary.replace(path)
