"""Build a filename-free content and decode inventory for source images."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFESTS = (
    PROJECT_ROOT / "data" / "qwen-material" / "train.jsonl",
    PROJECT_ROOT / "data" / "qwen-material" / "val.jsonl",
    PROJECT_ROOT / "data" / "qwen-material" / "test.jsonl",
)
DEFAULT_RECOVERY_AUDIT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "IMAGE_READ_RECOVERY_AUDIT.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SOURCE_IMAGE_INTEGRITY_CURRENT.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash and fully decode all unique source images"
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=list(DEFAULT_MANIFESTS),
    )
    parser.add_argument(
        "--recovery-audit",
        type=Path,
        default=DEFAULT_RECOVERY_AUDIT,
    )
    parser.add_argument(
        "--expected-recovery-state",
        choices=("damaged", "repaired"),
        default="repaired",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(values: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            values,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _decode(path: Path) -> dict[str, Any]:
    image_open = getattr(Image, "_mmfc_original_open", Image.open)
    try:
        with image_open(path) as image:
            decoded = ImageOps.exif_transpose(image).convert("RGB")
            decoded.load()
        return {
            "valid": True,
            "width": decoded.width,
            "height": decoded.height,
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def _load_source_index(
    manifests: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    source_index: dict[str, dict[str, Any]] = {}
    manifest_bindings: list[dict[str, Any]] = []
    for manifest in manifests:
        manifest = _resolve(manifest)
        manifest_bindings.append(
            {
                "path": _display_path(manifest),
                "size_bytes": manifest.stat().st_size,
                "sha256": _sha256(manifest),
            }
        )
        with manifest.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source_id = str(row["source_image_id"])
                split = str(row["split"])
                image_path = _resolve(Path(str(row["image_path"])))
                existing = source_index.get(source_id)
                value = {
                    "source_image_id": source_id,
                    "split": split,
                    "path": image_path,
                }
                if existing is None:
                    source_index[source_id] = value
                elif (
                    existing["split"] != split
                    or existing["path"] != image_path
                ):
                    raise ValueError(
                        "source image mapping drift at "
                        f"{manifest}:{line_number}: {source_id}"
                    )
    return source_index, manifest_bindings


def audit_source_image_integrity(
    *,
    manifests: Sequence[Path],
    recovery_audit_path: Path,
    expected_recovery_state: str = "damaged",
) -> dict[str, Any]:
    if expected_recovery_state not in {"damaged", "repaired"}:
        raise ValueError(
            "expected_recovery_state must be 'damaged' or 'repaired'"
        )
    recovery_audit_path = _resolve(recovery_audit_path)
    recovery = _read_json(recovery_audit_path)
    if recovery.get("valid") is not True:
        raise ValueError("recovery audit must be valid")
    recovery_source = _resolve(Path(str(recovery["source"]["path"])))
    recovery_fallback = _resolve(Path(str(recovery["fallback"]["path"])))
    recovery_source_hash = str(recovery["source"]["sha256"])
    recovery_fallback_hash = str(recovery["fallback"]["sha256"])
    source_index, manifest_bindings = _load_source_index(manifests)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for source_id in sorted(source_index):
        entry = source_index[source_id]
        path = entry["path"]
        if not path.is_file():
            errors.append(f"missing source image: {source_id}")
            rows.append(
                {
                    "source_image_id": source_id,
                    "split": entry["split"],
                    "size_bytes": None,
                    "raw_sha256": None,
                    "decode_valid": False,
                    "decode_error_type": "FileNotFoundError",
                    "width": None,
                    "height": None,
                    "recovery_applied": False,
                    "effective_sha256": None,
                }
            )
            continue
        raw_hash = _sha256(path)
        decoded = _decode(path)
        recovery_applied = False
        effective_hash = raw_hash
        if not decoded["valid"]:
            if path != recovery_source:
                errors.append(
                    f"undocumented decode failure: {source_id} "
                    f"({decoded['error_type']})"
                )
            elif raw_hash != recovery_source_hash:
                errors.append(
                    f"recovery source hash mismatch: {source_id}"
                )
            elif _sha256(recovery_fallback) != recovery_fallback_hash:
                errors.append(
                    f"recovery fallback hash mismatch: {source_id}"
                )
            else:
                fallback_decode = _decode(recovery_fallback)
                if not fallback_decode["valid"]:
                    errors.append(
                        f"recovery fallback decode failed: {source_id}"
                    )
                else:
                    recovery_applied = True
                    effective_hash = recovery_fallback_hash
        rows.append(
            {
                "source_image_id": source_id,
                "split": entry["split"],
                "size_bytes": path.stat().st_size,
                "raw_sha256": raw_hash,
                "decode_valid": bool(decoded["valid"]),
                "decode_error_type": decoded["error_type"],
                "width": decoded["width"],
                "height": decoded["height"],
                "recovery_applied": recovery_applied,
                "effective_sha256": effective_hash,
            }
        )

    split_counts = Counter(row["split"] for row in rows)
    decode_failure_count = sum(not row["decode_valid"] for row in rows)
    recovered_count = sum(row["recovery_applied"] for row in rows)
    raw_fingerprint_rows = [
        {
            "source_image_id": row["source_image_id"],
            "split": row["split"],
            "size_bytes": row["size_bytes"],
            "sha256": row["raw_sha256"],
        }
        for row in rows
    ]
    effective_fingerprint_rows = [
        {
            "source_image_id": row["source_image_id"],
            "split": row["split"],
            "size_bytes": row["size_bytes"],
            "sha256": row["effective_sha256"],
        }
        for row in rows
    ]
    if expected_recovery_state == "damaged":
        if decode_failure_count != 1 or recovered_count != 1:
            errors.append(
                "expected exactly one documented decode failure and one recovery"
            )
    elif decode_failure_count != 0 or recovered_count != 0:
        errors.append(
            "expected the documented source repair to decode without recovery"
        )
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "filename_free_source_image_content_and_decode_inventory",
        "expected_recovery_state": expected_recovery_state,
        "manifest_bindings": manifest_bindings,
        "recovery_audit_binding": {
            "path": _display_path(recovery_audit_path),
            "size_bytes": recovery_audit_path.stat().st_size,
            "sha256": _sha256(recovery_audit_path),
        },
        "summary": {
            "source_image_count": len(rows),
            "split_source_counts": dict(sorted(split_counts.items())),
            "decode_failure_count": decode_failure_count,
            "documented_recovery_count": recovered_count,
            "raw_inventory_sha256": _sequence_sha256(
                raw_fingerprint_rows
            ),
            "effective_inventory_sha256": _sequence_sha256(
                effective_fingerprint_rows
            ),
            "original_data_write_performed": False,
        },
        "images": rows,
        "errors": errors,
        "boundaries": [
            (
                "Rows expose source_image_id and content hashes, not original "
                "filenames or image bytes."
            ),
            (
                "The raw fingerprint identifies current files; the effective "
                "fingerprint substitutes the documented fallback hash for the "
                "single unreadable train image."
            ),
            (
                "This inventory was created after the incident and cannot "
                "reconstruct an unrecorded pre-incident hash."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    recovery_state = report.get("expected_recovery_state", "damaged")
    recovered = [
        row for row in report["images"] if row["recovery_applied"]
    ]
    return "\n".join(
        [
            "# 源图内容完整性审计",
            "",
            (
                f"共核验 {summary['source_image_count']} 张唯一源图：train/val/test = "
                f"{summary['split_source_counts'].get('train', 0)}/"
                f"{summary['split_source_counts'].get('val', 0)}/"
                f"{summary['split_source_counts'].get('test', 0)}。"
            ),
            (
                f"完整解码失败 {summary['decode_failure_count']} 张，"
                f"已披露只读恢复 {summary['documented_recovery_count']} 张。"
            ),
            "",
            (
                f"当前原始文件清单指纹：`{summary['raw_inventory_sha256']}`。"
            ),
            (
                (
                    "应用已披露单图恢复后的有效输入指纹："
                    if recovery_state == "damaged"
                    else "源图已修复后的有效输入指纹："
                )
                + f"`{summary['effective_inventory_sha256']}`。"
            ),
            "",
            "## 恢复条目",
            "",
            *[
                (
                    f"- source_image_id `{row['source_image_id']}`，"
                    f"split `{row['split']}`，原始解码错误 "
                    f"`{row['decode_error_type']}`，有效 SHA-256 "
                    f"`{row['effective_sha256']}`。"
                )
                for row in recovered
            ],
            "",
            "## 边界",
            "",
            "- 报告不保存原文件名或图像字节，只保存哈希化源图 ID 与内容哈希。",
            "- 该清单在事件后生成，不能恢复未事先记录的故障前单图哈希。",
            (
                "- 原始指纹描述当前磁盘文件；有效指纹包含已披露的单图镜像替换。"
                if recovery_state == "damaged"
                else "- 当前损坏路径已与锁定镜像逐字节相同，未再应用运行时替换。"
            ),
            "",
        ]
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    report = audit_source_image_integrity(
        manifests=args.manifests,
        recovery_audit_path=args.recovery_audit,
        expected_recovery_state=args.expected_recovery_state,
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
                **report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
