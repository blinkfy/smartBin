"""Release calibrated physical sidecars under a custodian attestation."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from mmfc.external_blind import _sensor_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_MODE = "physical-array-v1"
RELEASE_MODE = "synchronized-hardware"
ATTESTATION_TYPE = "synchronized_hardware_origin_v1"
ATTESTATION_KEYS = {
    "schema_version",
    "attestation_type",
    "dataset_id",
    "custodian_id",
    "signed_at",
    "physical_sidecar_sha256",
    "physical_build_report_sha256",
    "external_manifest_sha256",
    "sample_count",
    "collection_site_ids",
    "capture_device_ids",
    "hardware_origin_attested_by_custodian",
    "independent_of_model_development",
    "annotation_and_class_inputs_excluded",
}
SUPERVISION_KEYS = {
    "label",
    "label_id",
    "class",
    "class_id",
    "target",
    "annotation",
    "annotations",
}


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no sidecar records")
    return rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"attestation {field} must be a non-empty string")
    return value.strip()


def _nonempty_unique_strings(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(
            f"attestation {field} must be a non-empty string list"
        )
    cleaned = [item.strip() for item in value]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"attestation {field} must not contain duplicates")
    return cleaned


def _timezone_timestamp(value: object) -> str:
    text = _nonempty_string(value, "signed_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "attestation signed_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("attestation signed_at must include a timezone")
    return parsed.isoformat()


def _validate_attestation(
    attestation: Mapping[str, Any],
    *,
    sidecar_sha256: str,
    build_report_sha256: str,
    external_manifest_sha256: str,
    sample_count: int,
) -> dict[str, Any]:
    if set(attestation) != ATTESTATION_KEYS:
        missing = sorted(ATTESTATION_KEYS - set(attestation))
        extra = sorted(set(attestation) - ATTESTATION_KEYS)
        raise ValueError(
            "attestation fields mismatch "
            f"(missing={missing}, extra={extra})"
        )
    if (
        isinstance(attestation["schema_version"], bool)
        or not isinstance(attestation["schema_version"], int)
        or attestation["schema_version"] != 1
    ):
        raise ValueError("attestation schema_version must be 1")
    if attestation["attestation_type"] != ATTESTATION_TYPE:
        raise ValueError("attestation_type is not supported")
    if attestation["physical_sidecar_sha256"] != sidecar_sha256:
        raise ValueError("attestation sidecar hash does not match")
    if attestation["physical_build_report_sha256"] != build_report_sha256:
        raise ValueError("attestation build-report hash does not match")
    if attestation["external_manifest_sha256"] != external_manifest_sha256:
        raise ValueError("attestation external-manifest hash does not match")
    if (
        isinstance(attestation["sample_count"], bool)
        or not isinstance(attestation["sample_count"], int)
        or attestation["sample_count"] != sample_count
    ):
        raise ValueError("attestation sample_count does not match")
    for field in (
        "hardware_origin_attested_by_custodian",
        "independent_of_model_development",
        "annotation_and_class_inputs_excluded",
    ):
        if attestation[field] is not True:
            raise ValueError(f"attestation {field} must be true")
    return {
        "dataset_id": _nonempty_string(attestation["dataset_id"], "dataset_id"),
        "custodian_id": _nonempty_string(
            attestation["custodian_id"], "custodian_id"
        ),
        "signed_at": _timezone_timestamp(attestation["signed_at"]),
        "collection_site_ids": _nonempty_unique_strings(
            attestation["collection_site_ids"], "collection_site_ids"
        ),
        "capture_device_ids": _nonempty_unique_strings(
            attestation["capture_device_ids"], "capture_device_ids"
        ),
    }


def release_sidecars(
    sidecar_path: Path,
    build_report_path: Path,
    manifest_path: Path,
    attestation_path: Path,
    output_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Create blind-protocol-compatible sidecars after strict attestation."""

    sidecar_path = _resolve(sidecar_path)
    build_report_path = _resolve(build_report_path)
    manifest_path = _resolve(manifest_path)
    attestation_path = _resolve(attestation_path)
    output_path = _resolve(output_path)
    report_path = (
        _resolve(report_path)
        if report_path is not None
        else output_path.with_name("HARDWARE_RELEASE_REPORT.json")
    )
    if output_path == report_path:
        raise ValueError(
            "hardware release output and report paths must be distinct"
        )
    inputs = {
        "physical_sidecars": sidecar_path,
        "physical_build_report": build_report_path,
        "external_manifest": manifest_path,
        "custodian_attestation": attestation_path,
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "physical release inputs missing: " + ", ".join(missing)
        )
    occupied = [path for path in (output_path, report_path) if path.exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite hardware release outputs: "
            + ", ".join(str(path) for path in occupied)
        )

    sidecar_sha256 = _sha256(sidecar_path)
    build_report_sha256 = _sha256(build_report_path)
    external_manifest_sha256 = _sha256(manifest_path)
    rows = _read_jsonl(sidecar_path)
    manifest_rows = _read_jsonl(manifest_path)
    build_report = _read_json(build_report_path)
    if (
        build_report.get("state") != "complete"
        or build_report.get("valid") is not True
        or build_report.get("source_mode") != PHYSICAL_MODE
        or build_report.get("sample_count") != len(rows)
        or build_report.get("unique_sample_count") != len(rows)
        or build_report.get("physical_hardware_origin_declared") is not True
        or build_report.get("physical_hardware_origin_verified") is not False
        or build_report.get("output", {}).get("sha256") != sidecar_sha256
    ):
        raise ValueError("physical build report does not bind the sidecars")

    manifest_by_sample: dict[str, str] = {}
    for index, row in enumerate(manifest_rows, start=1):
        sample_id = str(row.get("sample_id", "")).strip()
        source_image_id = str(row.get("source_image_id", "")).strip()
        if (
            not sample_id
            or not source_image_id
            or sample_id in manifest_by_sample
        ):
            raise ValueError(
                f"external manifest row {index} has a missing or duplicate identity"
            )
        manifest_by_sample[sample_id] = source_image_id

    sample_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(
                f"physical sidecars row {index} has a missing or duplicate sample_id"
            )
        sample_ids.add(sample_id)
        if SUPERVISION_KEYS & set(row):
            raise ValueError(
                f"physical sidecars row {index} contains supervision fields"
            )
        _sensor_payload(row, PHYSICAL_MODE)
        source = row["source"]
        if (
            source.get("generator") != "hardware-calibrated-v1"
            or source.get("physical_hardware_origin_declared") is not True
            or source.get("physical_hardware_origin_verified") is not False
        ):
            raise ValueError(
                f"physical sidecars row {index} lacks the calibrated origin contract"
            )
        if source.get("source_image_id") != manifest_by_sample.get(sample_id):
            raise ValueError(
                f"physical sidecars row {index} source_image_id does not match "
                "the external manifest"
            )
    if sample_ids != set(manifest_by_sample):
        raise ValueError(
            "physical sidecar sample IDs do not match the external manifest"
        )

    attestation_sha256 = _sha256(attestation_path)
    attested = _validate_attestation(
        _read_json(attestation_path),
        sidecar_sha256=sidecar_sha256,
        build_report_sha256=build_report_sha256,
        external_manifest_sha256=external_manifest_sha256,
        sample_count=len(rows),
    )
    released_rows = []
    for row in rows:
        released = copy.deepcopy(row)
        source = released["source"]
        source["mode"] = RELEASE_MODE
        source["release_generator"] = (
            "custodian-attested-hardware-release-v1"
        )
        source["hardware_origin_attested_by_custodian"] = True
        source["hardware_attestation_sha256"] = attestation_sha256
        source["custodian_id"] = attested["custodian_id"]
        source["dataset_id"] = attested["dataset_id"]
        _sensor_payload(released, RELEASE_MODE)
        released_rows.append(released)

    _atomic_text(
        output_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in released_rows
        ),
    )
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "custodian_attested_synchronized_hardware_sidecar_release",
        "sample_count": len(released_rows),
        "dataset_id": attested["dataset_id"],
        "custodian_id": attested["custodian_id"],
        "attestation_signed_at": attested["signed_at"],
        "collection_site_ids": attested["collection_site_ids"],
        "capture_device_ids": attested["capture_device_ids"],
        "source_mode": PHYSICAL_MODE,
        "released_mode": RELEASE_MODE,
        "physical_hardware_origin_verified_by_software": False,
        "physical_hardware_origin_attested_by_custodian": True,
        "hardware_performance_validated": False,
        "external_blind_release_json_still_required": True,
        "inputs": {
            name: {
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in inputs.items()
        },
        "output": {
            "path": output_path.as_posix(),
            "size_bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "claim_boundary": (
            "The software verifies hashes and release declarations, not physical "
            "hardware origin or performance. The custodian remains responsible "
            "for the truth of the attestation, and the frozen external-blind "
            "RELEASE.json plus full bundle validation are still required."
        ),
        "errors": [],
    }
    _atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report
