"""Audit the read-only fallback used for one damaged training image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFile, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT.parent / "data" / "images" / "s_cabblestone_6.jpg"
DEFAULT_FALLBACK = (
    PROJECT_ROOT.parent / "data" / "combination" / "s_cabblestone_6.jpg"
)
DEFAULT_MANIFESTS = (
    PROJECT_ROOT / "data" / "qwen-material" / "train.jsonl",
    PROJECT_ROOT / "data" / "qwen-material" / "val.jsonl",
    PROJECT_ROOT / "data" / "qwen-material" / "test.jsonl",
)
DEFAULT_SHIM = PROJECT_ROOT / "runtime_recovery" / "sitecustomize.py"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "IMAGE_READ_RECOVERY_AUDIT.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a hash-locked read-only image recovery mapping"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--fallback", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=list(DEFAULT_MANIFESTS),
    )
    parser.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        try:
            return "../" + path.relative_to(PROJECT_ROOT.parent).as_posix()
        except ValueError:
            return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _decode(path: Path, *, allow_truncated: bool) -> dict[str, Any]:
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
    image_open = getattr(Image, "_mmfc_original_open", Image.open)
    try:
        with image_open(path) as image:
            decoded = ImageOps.exif_transpose(image).convert("RGB")
            decoded.load()
        return {
            "valid": True,
            "error_type": None,
            "error_message": None,
            "width": decoded.width,
            "height": decoded.height,
        }
    except Exception as exc:
        return {
            "valid": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "width": None,
            "height": None,
        }
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _decoded_array(path: Path, *, allow_truncated: bool) -> np.ndarray:
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
    image_open = getattr(Image, "_mmfc_original_open", Image.open)
    try:
        with image_open(path) as image:
            return np.asarray(
                ImageOps.exif_transpose(image).convert("RGB"),
                dtype=np.float32,
            )
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def _byte_difference(source: Path, fallback: Path) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    fallback_bytes = fallback.read_bytes()
    shared = min(len(source_bytes), len(fallback_bytes))
    offsets = [
        index
        for index in range(shared)
        if source_bytes[index] != fallback_bytes[index]
    ]
    return {
        "source_size_bytes": len(source_bytes),
        "fallback_size_bytes": len(fallback_bytes),
        "same_size": len(source_bytes) == len(fallback_bytes),
        "differing_byte_count": (
            len(offsets) + abs(len(source_bytes) - len(fallback_bytes))
        ),
        "first_differing_offset": offsets[0] if offsets else None,
        "last_differing_offset": offsets[-1] if offsets else None,
    }


def _manifest_occurrences(
    manifests: Sequence[Path],
    *,
    source: Path,
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for manifest in manifests:
        sample_ids: list[str] = []
        source_image_ids: list[str] = []
        with manifest.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                image_path = _resolve(Path(str(row["image_path"])))
                if image_path != source:
                    continue
                sample_ids.append(str(row["sample_id"]))
                source_image_ids.append(str(row["source_image_id"]))
        occurrences.append(
            {
                "manifest": _display_path(manifest),
                "manifest_sha256": _sha256(manifest),
                "split": manifest.stem,
                "sample_count": len(sample_ids),
                "sample_ids": sample_ids,
                "source_image_ids": source_image_ids,
            }
        )
    return occurrences


def audit_image_recovery(
    *,
    source: Path,
    fallback: Path,
    manifests: Sequence[Path],
    shim: Path,
) -> dict[str, Any]:
    source = _resolve(source)
    fallback = _resolve(fallback)
    manifests = [_resolve(path) for path in manifests]
    shim = _resolve(shim)
    source_decode = _decode(source, allow_truncated=False)
    fallback_decode = _decode(fallback, allow_truncated=False)
    source_tolerant = _decode(source, allow_truncated=True)
    byte_difference = _byte_difference(source, fallback)
    pixel_difference: dict[str, Any] | None = None
    if source_tolerant["valid"] and fallback_decode["valid"]:
        source_array = _decoded_array(source, allow_truncated=True)
        fallback_array = _decoded_array(fallback, allow_truncated=False)
        if source_array.shape == fallback_array.shape:
            absolute = np.abs(source_array - fallback_array)
            changed_pixels = np.any(absolute > 0, axis=2)
            pixel_difference = {
                "mean_absolute_error": float(absolute.mean()),
                "maximum_absolute_error": float(absolute.max()),
                "changed_pixel_count": int(changed_pixels.sum()),
                "changed_pixel_fraction": float(changed_pixels.mean()),
            }
    occurrences = _manifest_occurrences(manifests, source=source)
    occurrence_by_split = {
        row["split"]: row["sample_count"] for row in occurrences
    }
    errors: list[str] = []
    if source_decode["valid"]:
        errors.append("source unexpectedly decodes without recovery")
    if not fallback_decode["valid"]:
        errors.append("fallback does not decode normally")
    if not source_tolerant["valid"]:
        errors.append("source dimensions cannot be recovered for comparison")
    if (
        source_tolerant["width"],
        source_tolerant["height"],
    ) != (
        fallback_decode["width"],
        fallback_decode["height"],
    ):
        errors.append("source/fallback dimensions differ")
    if not byte_difference["same_size"]:
        errors.append("source/fallback byte lengths differ")
    if byte_difference["differing_byte_count"] > 32:
        errors.append("source/fallback differ by more than 32 bytes")
    if occurrence_by_split != {"train": 1, "val": 0, "test": 0}:
        errors.append(
            "expected exactly one train occurrence and no val/test occurrence"
        )
    if not shim.is_file():
        errors.append("runtime recovery shim is missing")

    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "read_only_training_image_integrity_recovery",
        "source": {
            "path": _display_path(source),
            "sha256": _sha256(source),
            "decode_without_recovery": source_decode,
            "decode_with_truncated_tolerance": source_tolerant,
        },
        "fallback": {
            "path": _display_path(fallback),
            "sha256": _sha256(fallback),
            "decode_without_recovery": fallback_decode,
        },
        "comparison": {
            "bytes": byte_difference,
            "decoded_pixels_after_tolerant_source_load": pixel_difference,
        },
        "manifest_occurrences": occurrences,
        "runtime_shim": {
            "path": _display_path(shim),
            "sha256": _sha256(shim) if shim.is_file() else None,
            "activation": (
                "opt-in PYTHONPATH plus four hash-locked "
                "MMFC_IMAGE_RECOVERY_* environment variables"
            ),
        },
        "original_data_write_performed": False,
        "errors": errors,
        "boundaries": [
            (
                "The source image, manifests, annotations, and fallback image "
                "are opened read-only and are not rewritten."
            ),
            (
                "The fallback affects one train object only; validation and "
                "test inputs are unchanged."
            ),
            (
                "This recovery preserves execution after a storage-integrity "
                "incident; it is not a data augmentation or model change."
            ),
            (
                "The source and fallback hashes must be reported with formal "
                "results because one training input was redirected."
            ),
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    source = report["source"]
    fallback = report["fallback"]
    comparison = report["comparison"]["bytes"]
    train_occurrence = next(
        row
        for row in report["manifest_occurrences"]
        if row["split"] == "train"
    )
    return "\n".join(
        [
            "# 训练图像读取恢复审计",
            "",
            f"状态：`{report['state']}`；有效：`{str(report['valid']).lower()}`。",
            "",
            "## 事件",
            "",
            (
                f"`{source['path']}` 无法被 Pillow 正常完整解码，错误类型为 "
                f"`{source['decode_without_recovery']['error_type']}`。"
            ),
            (
                f"同名镜像 `{fallback['path']}` 可正常解码；两文件长度相同，"
                f"共有 {comparison['differing_byte_count']} 个差异字节，"
                f"首个差异偏移为 {comparison['first_differing_offset']}。"
            ),
            "",
            "## 影响范围",
            "",
            (
                f"受影响路径只出现在 train 清单 {train_occurrence['sample_count']} 次，"
                f"sample ID 为 `{train_occurrence['sample_ids'][0]}`；val/test 为 0。"
            ),
            "原图、清单、标注和镜像文件均未改写。",
            "",
            "## 恢复方式",
            "",
            (
                "仅在显式设置 `PYTHONPATH=runtime_recovery` 和四个 "
                "`MMFC_IMAGE_RECOVERY_*` 环境变量时启用只读路径重定向。"
            ),
            (
                f"损坏源 SHA-256：`{source['sha256']}`；"
                f"镜像 SHA-256：`{fallback['sha256']}`。"
            ),
            (
                "shim 在进程启动时复核两份哈希，任一文件变化都会拒绝运行；"
                "其他路径不受影响。"
            ),
            "",
            "## 表述边界",
            "",
            "- 这是存储完整性故障恢复，不是数据增强或模型改动。",
            "- 恢复只涉及一个训练对象，验证和测试输入不变。",
            "- 正式结果必须同时披露源文件、镜像文件和 shim 的 SHA-256。",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    if output == DEFAULT_OUTPUT.resolve() and output.exists():
        raise FileExistsError(
            "historical image recovery audit is immutable; use the source "
            "repair transition and current integrity audit for the repaired state"
        )
    report = audit_image_recovery(
        source=args.source,
        fallback=args.fallback,
        manifests=args.manifests,
        shim=args.shim,
    )
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = output.with_suffix(".md")
    _atomic_write(markdown, render_markdown(report))
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "markdown": markdown.as_posix(),
                "valid": report["valid"],
                "source_sha256": report["source"]["sha256"],
                "fallback_sha256": report["fallback"]["sha256"],
                "differing_byte_count": report["comparison"]["bytes"][
                    "differing_byte_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
