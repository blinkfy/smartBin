"""Audit the observed transition from a damaged source to its locked mirror."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "recovery_audit": (
        "reports/qwen-material/IMAGE_READ_RECOVERY_AUDIT.json"
    ),
    "incident": (
        "reports/qwen-material/IMAGE_READ_RECOVERY_INCIDENT.json"
    ),
    "addendum": (
        "reports/qwen-material/IMAGE_READ_RECOVERY_INCIDENT_ADDENDUM.json"
    ),
    "outcome": (
        "reports/qwen-material/IMAGE_READ_RECOVERY_OUTCOME.json"
    ),
    "previous_integrity": (
        "reports/qwen-material/SOURCE_IMAGE_INTEGRITY_AUDIT.json"
    ),
    "current_integrity": (
        "reports/qwen-material/SOURCE_IMAGE_INTEGRITY_CURRENT.json"
    ),
    "source": "../data/images/s_cabblestone_6.jpg",
    "fallback": "../data/combination/s_cabblestone_6.jpg",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "SOURCE_IMAGE_REPAIR_TRANSITION.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the externally observed source-image repair"
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _decode(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            width, height = image.size
        return {"valid": True, "width": width, "height": height}
    except (OSError, SyntaxError, ValueError, ModuleNotFoundError) as exc:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "error_type": type(exc).__name__,
        }


def _binding(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _display_path(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
        "last_write_time": datetime.fromtimestamp(
            stat.st_mtime
        ).astimezone().isoformat(timespec="seconds"),
    }


def build_source_repair_transition(
    *,
    files: Mapping[str, Path],
) -> dict[str, Any]:
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "source repair transition inputs are missing: "
            + ", ".join(sorted(missing))
        )
    recovery = _read_json(resolved["recovery_audit"])
    incident = _read_json(resolved["incident"])
    addendum = _read_json(resolved["addendum"])
    outcome = _read_json(resolved["outcome"])
    previous = _read_json(resolved["previous_integrity"])
    current = _read_json(resolved["current_integrity"])
    errors: list[str] = []

    if recovery.get("valid") is not True:
        errors.append("historical recovery audit is not valid")
    if incident.get("valid") is not True:
        errors.append("incident snapshot is not valid")
    if addendum.get("valid") is not True:
        errors.append("incident addendum is not valid")
    if outcome.get("valid") is not True:
        errors.append("recovery outcome is not valid")
    if previous.get("valid") is not True:
        errors.append("previous source integrity audit is not valid")
    if current.get("valid") is not True:
        errors.append("current source integrity audit is not valid")

    recovery_hash = _sha256(resolved["recovery_audit"])
    incident_binding = (
        incident.get("source_bindings", {}).get("recovery_audit", {})
    )
    outcome_bindings = outcome.get("source_bindings", {})
    incident_mapping = incident.get("snapshot", {}).get(
        "recovery_mapping", {}
    )
    if not incident_binding.get("sha256"):
        errors.append("incident recovery-audit binding is missing")
    if (
        outcome_bindings.get("recovery_audit", {}).get("sha256")
        != recovery_hash
    ):
        errors.append("outcome does not bind the historical recovery audit")
    if (
        outcome_bindings.get("source_integrity", {}).get("sha256")
        != _sha256(resolved["previous_integrity"])
    ):
        errors.append("outcome does not bind the previous image inventory")

    old_source_hash = recovery.get("source", {}).get("sha256")
    locked_fallback_hash = recovery.get("fallback", {}).get("sha256")
    if (
        incident_mapping.get("source", {}).get("sha256")
        != old_source_hash
        or incident_mapping.get("fallback", {}).get("sha256")
        != locked_fallback_hash
    ):
        errors.append("incident recovery mapping hash mismatch")
    current_source_hash = _sha256(resolved["source"])
    current_fallback_hash = _sha256(resolved["fallback"])
    if old_source_hash == locked_fallback_hash:
        errors.append("historical source and fallback hashes do not differ")
    if current_source_hash != locked_fallback_hash:
        errors.append("current source does not equal the locked fallback hash")
    if current_fallback_hash != locked_fallback_hash:
        errors.append("current fallback hash drifted")
    if resolved["source"].read_bytes() != resolved["fallback"].read_bytes():
        errors.append("current source and fallback are not byte-identical")
    source_decode = _decode(resolved["source"])
    fallback_decode = _decode(resolved["fallback"])
    if not source_decode["valid"] or not fallback_decode["valid"]:
        errors.append("current source or fallback does not fully decode")
    if source_decode.get("width") != fallback_decode.get("width") or (
        source_decode.get("height") != fallback_decode.get("height")
    ):
        errors.append("current source/fallback dimensions differ")

    previous_summary = previous.get("summary", {})
    current_summary = current.get("summary", {})
    if current.get("expected_recovery_state") != "repaired":
        errors.append("current inventory is not marked as repaired")
    if (
        current_summary.get("decode_failure_count") != 0
        or current_summary.get("documented_recovery_count") != 0
    ):
        errors.append("current inventory still requires recovery")
    if (
        previous_summary.get("effective_inventory_sha256")
        != current_summary.get("raw_inventory_sha256")
        or current_summary.get("raw_inventory_sha256")
        != current_summary.get("effective_inventory_sha256")
    ):
        errors.append("repaired inventory does not equal prior effective input")
    if (
        previous_summary.get("raw_inventory_sha256")
        == current_summary.get("raw_inventory_sha256")
    ):
        errors.append("source inventory transition was not detected")

    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "observed_source_image_repair_transition",
        "observed_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "transition": {
            "cause": "unknown_external_state_change",
            "incident_recovery_audit_sha256": incident_binding.get(
                "sha256"
            ),
            "latest_historical_recovery_audit_sha256": recovery_hash,
            "historical_source_sha256": old_source_hash,
            "locked_fallback_sha256": locked_fallback_hash,
            "current_source_sha256": current_source_hash,
            "current_fallback_sha256": current_fallback_hash,
            "source_now_equals_locked_fallback": (
                current_source_hash == locked_fallback_hash
            ),
            "source_and_fallback_byte_identical": (
                resolved["source"].read_bytes()
                == resolved["fallback"].read_bytes()
            ),
            "source_decode": source_decode,
            "fallback_decode": fallback_decode,
            "previous_raw_inventory_sha256": previous_summary.get(
                "raw_inventory_sha256"
            ),
            "previous_effective_inventory_sha256": previous_summary.get(
                "effective_inventory_sha256"
            ),
            "current_raw_inventory_sha256": current_summary.get(
                "raw_inventory_sha256"
            ),
            "current_effective_inventory_sha256": current_summary.get(
                "effective_inventory_sha256"
            ),
        },
        "source_bindings": {
            name: _binding(path) for name, path in sorted(resolved.items())
        },
        "errors": errors,
        "boundaries": [
            (
                "The project observed but did not attribute the write that "
                "changed the damaged source path; filesystem timestamps are "
                "recorded but are not treated as a reliable transition time."
            ),
            (
                "The historical damaged-state audits remain immutable and are "
                "not recomputed as if they described the repaired source tree."
            ),
            (
                "The repaired raw inventory equals the previously documented "
                "effective inventory, so subsequent runs no longer require the "
                "runtime redirection shim."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    transition = report["transition"]
    return "\n".join(
        [
            "# 源图修复状态转换审计",
            "",
            f"观测时间：`{report['observed_at']}`。",
            "",
            (
                "先前损坏路径当前已与哈希锁定镜像逐字节相同，SHA-256 为 "
                f"`{transition['current_source_sha256']}`，可完整解码。"
            ),
            (
                "修复后 5,990 张源图的原始与有效清单指纹均为 "
                f"`{transition['current_raw_inventory_sha256']}`，等于修复前"
                "已披露的有效输入指纹。"
            ),
            "",
            "此次文件状态变化的写入来源和精确时刻无法从现有证据归因；项目保留"
            "损坏态快照，并将后续运行明确切换到不启用 shim 的已修复态。",
            "",
            "## 边界",
            "",
            *[f"- {item}" for item in report["boundaries"]],
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError(
            f"source image repair transition is immutable: {output}"
        )
    report = build_source_repair_transition(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "valid": report["valid"],
                "cause": report["transition"]["cause"],
                "current_source_sha256": report["transition"][
                    "current_source_sha256"
                ],
                "current_inventory_sha256": report["transition"][
                    "current_raw_inventory_sha256"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
