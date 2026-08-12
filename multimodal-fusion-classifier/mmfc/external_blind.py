"""External blind-confirmation intake validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageOps

from .training import PROJECT_ROOT


PROPERTY_NAMES = (
    "weight_g",
    "temperature_c",
    "moisture_pct",
    "conductivity_us_cm",
    "odor_index",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            records.append(value)
    return records


def difference_hash(path: Path, hash_size: int = 8) -> str:
    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        resized = image.resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        values = np.asarray(resized, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    number = 0
    for bit in bits.reshape(-1):
        number = (number << 1) | int(bit)
    width = math.ceil(hash_size * hash_size / 4)
    return f"{number:0{width}x}"


def hash_hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right) or not left:
        raise ValueError("difference hashes must have equal non-zero width")
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError as exc:
        raise ValueError("difference hashes must be hexadecimal") from exc


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO timestamp")
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _inside(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image_path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("external image_path must be relative to bundle root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("image_path escapes the external bundle") from exc
    return resolved


def _bbox(value: Any, width: int, height: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must be [x1,y1,x2,y2]")
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox values must be numeric") from exc
    if not all(math.isfinite(item) for item in box):
        raise ValueError("bbox values must be finite")
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox is empty or outside the positive image plane")
    if x2 > width or y2 > height:
        raise ValueError("bbox exceeds decoded image dimensions")
    return box


def _sensor_payload(
    record: Mapping[str, Any],
    expected_source_mode: str | None = None,
) -> None:
    grid = record.get("normalized_grid")
    if (
        not isinstance(grid, list)
        or len(grid) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in grid)
    ):
        raise ValueError("normalized_grid must be a strict 4x4 list")
    flat: list[float] = []
    for row in grid:
        for value in row:
            if isinstance(value, bool):
                raise ValueError("normalized_grid values must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "normalized_grid values must be numeric"
                ) from exc
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(
                    "normalized_grid values must be finite in [0,1]"
                )
            flat.append(number)
    properties = record.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("sensor properties must be an object")
    for name in PROPERTY_NAMES:
        value = properties.get(name)
        if isinstance(value, bool):
            raise ValueError(f"sensor property {name} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sensor property {name} must be numeric"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"sensor property {name} must be finite")
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("sensor source declaration is missing")
    if (
        source.get("annotation_inputs") is not False
        or source.get("class_inputs") is not False
    ):
        raise ValueError(
            "sensor generation must explicitly exclude annotations/classes"
        )
    if (
        expected_source_mode is not None
        and source.get("mode") != expected_source_mode
    ):
        raise ValueError("sensor source mode does not match RELEASE.json")


def _development_inventory(
    integrity_path: Path,
) -> tuple[set[str], set[str], set[str]]:
    integrity = read_json(integrity_path)
    if integrity.get("valid") is not True:
        raise ValueError("development source integrity is not valid")
    source_ids: set[str] = set()
    image_hashes: set[str] = set()
    for item in integrity.get("images", []):
        source_ids.add(str(item["source_image_id"]))
        image_hashes.add(str(item["effective_sha256"]))
    sample_ids: set[str] = set()
    for binding in integrity.get("manifest_bindings", []):
        manifest = PROJECT_ROOT / str(binding["path"])
        if sha256_file(manifest) != binding["sha256"]:
            raise ValueError("bound development manifest drifted")
        for record in read_jsonl(manifest):
            sample_id = str(record.get("sample_id", record.get("id", "")))
            if sample_id:
                sample_ids.add(sample_id)
    return source_ids, sample_ids, image_hashes


def validate_external_blind_bundle(
    bundle_root: Path,
    *,
    protocol_path: Path,
    development_integrity_path: Path,
    development_dhash_index_path: Path,
) -> dict[str, Any]:
    root = bundle_root.resolve()
    protocol_path = protocol_path.resolve()
    integrity_path = development_integrity_path.resolve()
    dhash_path = development_dhash_index_path.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "release": root / "RELEASE.json",
        "manifest": root / "manifest.jsonl",
        "sensors": root / "sensors.jsonl",
        "protocol": protocol_path,
        "development_integrity": integrity_path,
        "development_dhash_index": dhash_path,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "schema_version": 1,
            "state": "incomplete",
            "valid": False,
            "error_count": len(missing),
            "errors": [f"missing required input: {name}" for name in missing],
            "warning_count": 0,
            "warnings": [],
        }

    protocol = read_json(protocol_path)
    release = read_json(required["release"])
    manifest = read_jsonl(required["manifest"])
    sensors = read_jsonl(required["sensors"])
    class_names = list(protocol.get("class_names", []))
    criteria = protocol.get("intake_criteria", {})
    if (
        protocol.get("protocol") != "external_blind_confirmation_v1"
        or protocol.get("status") != "preregistered_awaiting_external_data"
        or len(class_names) != 19
    ):
        raise ValueError("external blind protocol identity is invalid")

    unsealed_at: datetime | None = None
    try:
        frozen_at = _timestamp(protocol["frozen_at"], "protocol.frozen_at")
        unsealed_at = _timestamp(
            release.get("labels_unsealed_at"),
            "release.labels_unsealed_at",
        )
        if unsealed_at < frozen_at:
            errors.append("labels were unsealed before protocol freeze")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    if release.get("schema_version") != 1:
        errors.append("release schema_version must be 1")
    if release.get("role") != "external_blind_confirmation":
        errors.append("release role is not external_blind_confirmation")
    if release.get("protocol") != protocol["protocol"]:
        errors.append("release protocol name does not match")
    if release.get("protocol_sha256") != sha256_file(protocol_path):
        errors.append("release protocol hash does not match")
    if (
        release.get("checkpoint_sha256")
        != protocol.get("checkpoint", {}).get("sha256")
    ):
        errors.append("release checkpoint hash does not match")
    if release.get("prior_model_development_access") is not False:
        errors.append("release must declare no prior model development access")
    if release.get("one_time_evaluation_authorized") is not True:
        errors.append("release does not authorize the one-time evaluation")
    if release.get("class_names") != class_names:
        errors.append("release class order does not match protocol")
    for name in ("dataset_id", "custodian_id"):
        if not isinstance(release.get(name), str) or not release[name].strip():
            errors.append(f"release {name} is missing")
    sensor_source_mode = str(
        release.get("sensor_source_mode", "")
    ).strip()
    if sensor_source_mode not in set(
        criteria.get("allowed_sensor_source_modes", [])
    ):
        errors.append("release sensor_source_mode is not preregistered")
    for name in ("collection_site_ids", "capture_device_ids"):
        values = release.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            errors.append(f"release {name} must be a non-empty string list")
    try:
        collection_started = _timestamp(
            release.get("collection_started_at"),
            "release.collection_started_at",
        )
        collection_completed = _timestamp(
            release.get("collection_completed_at"),
            "release.collection_completed_at",
        )
        if collection_completed < collection_started:
            errors.append("collection completed before it started")
        if unsealed_at is None:
            raise ValueError("release.labels_unsealed_at is invalid")
        if unsealed_at < collection_completed:
            errors.append("labels were unsealed before collection completed")
    except ValueError as exc:
        errors.append(str(exc))

    protocol_bindings = protocol.get("bindings")
    if isinstance(protocol_bindings, Mapping):
        for name, path in (
            ("development_integrity", integrity_path),
            ("development_dhash_index", dhash_path),
        ):
            expected = protocol_bindings.get(name, {})
            if expected.get("sha256") != sha256_file(path):
                errors.append(f"protocol-bound {name} drifted")

    development_source_ids, development_sample_ids, development_hashes = (
        _development_inventory(integrity_path)
    )
    development_dhashes = [
        str(item["dhash"]) for item in read_jsonl(dhash_path)
    ]
    sample_ids: set[str] = set()
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    source_dhashes: dict[str, str] = {}
    source_batches: dict[str, set[str]] = defaultdict(set)
    source_sessions: dict[str, set[str]] = defaultdict(set)
    class_counts: Counter[str] = Counter()
    class_sources: dict[str, set[str]] = defaultdict(set)
    observed_batches: set[str] = set()
    observed_sessions: set[str] = set()

    for index, record in enumerate(manifest, start=1):
        prefix = f"manifest line {index}"
        sample_id = str(record.get("sample_id", "")).strip()
        source_id = str(record.get("source_image_id", "")).strip()
        batch_id = str(record.get("acquisition_batch_id", "")).strip()
        session_id = str(record.get("capture_session_id", "")).strip()
        label = str(record.get("label", "")).strip()
        if not sample_id or sample_id in sample_ids:
            errors.append(f"{prefix}: missing or duplicate sample_id")
            continue
        sample_ids.add(sample_id)
        if sample_id in development_sample_ids:
            errors.append(f"{prefix}: sample_id overlaps development data")
        if not source_id:
            errors.append(f"{prefix}: source_image_id is missing")
        elif source_id in development_source_ids:
            errors.append(f"{prefix}: source_image_id overlaps development data")
        if not batch_id or not session_id:
            errors.append(f"{prefix}: batch/session id is missing")
        else:
            observed_batches.add(batch_id)
            observed_sessions.add(session_id)
            source_batches[source_id].add(batch_id)
            source_sessions[source_id].add(session_id)
        if label not in class_names:
            errors.append(f"{prefix}: label is outside the 19-class taxonomy")
        else:
            expected_label_id = class_names.index(label)
            try:
                label_id = int(record.get("label_id"))
            except (TypeError, ValueError):
                label_id = -1
            if label_id != expected_label_id:
                errors.append(f"{prefix}: label_id does not match class order")
            class_counts[label] += 1
            class_sources[label].add(source_id)
        review = record.get("label_review")
        try:
            reviewer_count = int(
                review.get("reviewer_count", 0)
                if isinstance(review, Mapping)
                else 0
            )
        except (TypeError, ValueError):
            reviewer_count = 0
        if (
            not isinstance(review, Mapping)
            or review.get("consensus") is not True
            or reviewer_count < int(
                criteria.get("minimum_label_reviewers", 2)
            )
        ):
            errors.append(f"{prefix}: label review does not pass consensus")
        forbidden_sensor_fields = {
            "grid",
            "normalized_grid",
            "properties",
            "sensor_path",
        }
        if forbidden_sensor_fields.intersection(record):
            errors.append(
                f"{prefix}: sensor payload must stay in sensors.jsonl"
            )

        try:
            image_path = _inside(root, record.get("image_path"))
            if not image_path.is_file():
                raise FileNotFoundError("image file is missing")
            image_hash = sha256_file(image_path)
            if record.get("image_sha256") != image_hash:
                raise ValueError("image_sha256 does not match file")
            if image_hash in development_hashes:
                errors.append(f"{prefix}: exact image overlaps development data")
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                image.verify()
            _bbox(record.get("bbox"), width, height)
            if source_id in source_paths and source_paths[source_id] != image_path:
                errors.append(
                    f"{prefix}: one source_image_id maps to multiple files"
                )
            if (
                source_id in source_hashes
                and source_hashes[source_id] != image_hash
            ):
                errors.append(
                    f"{prefix}: one source_image_id maps to multiple hashes"
                )
            source_paths[source_id] = image_path
            source_hashes[source_id] = image_hash
            if source_id not in source_dhashes:
                source_dhashes[source_id] = difference_hash(image_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")

    for source_id, values in source_batches.items():
        if len(values) != 1:
            errors.append(
                f"source {source_id} crosses acquisition batches"
            )
    for source_id, values in source_sessions.items():
        if len(values) != 1:
            errors.append(f"source {source_id} crosses capture sessions")

    severe_threshold = int(
        criteria.get("perceptual_hash_error_distance", 2)
    )
    review_threshold = int(
        criteria.get("perceptual_hash_review_distance", 6)
    )
    severe_near_duplicates = 0
    review_near_duplicates = 0
    for source_id, external_hash in source_dhashes.items():
        minimum = min(
            (
                hash_hamming_distance(external_hash, development_hash)
                for development_hash in development_dhashes
            ),
            default=64,
        )
        if minimum <= severe_threshold:
            severe_near_duplicates += 1
            errors.append(
                f"source {source_id} has development dHash distance {minimum}"
            )
        elif minimum <= review_threshold:
            review_near_duplicates += 1
            warnings.append(
                f"source {source_id} needs near-duplicate review "
                f"(dHash distance {minimum})"
            )

    sensor_by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(sensors, start=1):
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id or sample_id in sensor_by_id:
            errors.append(
                f"sensors line {index}: missing or duplicate sample_id"
            )
            continue
        try:
            _sensor_payload(record, sensor_source_mode)
        except ValueError as exc:
            errors.append(f"sensors line {index}: {exc}")
        sensor_by_id[sample_id] = record
    missing_sensors = sample_ids - set(sensor_by_id)
    extra_sensors = set(sensor_by_id) - sample_ids
    if missing_sensors:
        errors.append(
            f"sensor records missing for {len(missing_sensors)} samples"
        )
    if extra_sensors:
        errors.append(
            f"sensor records contain {len(extra_sensors)} unknown samples"
        )

    minimum_samples = int(criteria.get("minimum_total_samples", 380))
    minimum_sources = int(criteria.get("minimum_source_images", 190))
    minimum_per_class = int(criteria.get("minimum_samples_per_class", 20))
    minimum_sources_per_class = int(
        criteria.get("minimum_source_images_per_class", 10)
    )
    minimum_batches = int(criteria.get("minimum_acquisition_batches", 3))
    minimum_sessions = int(criteria.get("minimum_capture_sessions", 3))
    if len(sample_ids) < minimum_samples:
        errors.append("external bundle has too few total samples")
    if len(source_paths) < minimum_sources:
        errors.append("external bundle has too few source images")
    if len(observed_batches) < minimum_batches:
        errors.append("external bundle has too few acquisition batches")
    if len(observed_sessions) < minimum_sessions:
        errors.append("external bundle has too few capture sessions")
    for name in class_names:
        if class_counts[name] < minimum_per_class:
            errors.append(f"class {name} has too few samples")
        if len(class_sources[name]) < minimum_sources_per_class:
            errors.append(f"class {name} has too few source images")

    released_batches = release.get("acquisition_batch_ids", [])
    declared_batches = (
        set(released_batches) if isinstance(released_batches, list) else set()
    )
    if not isinstance(released_batches, list):
        errors.append("release acquisition_batch_ids must be a list")
    if declared_batches != observed_batches:
        errors.append("release acquisition_batch_ids do not match manifest")
    released_sessions = release.get("capture_session_ids", [])
    declared_sessions = (
        set(released_sessions) if isinstance(released_sessions, list) else set()
    )
    if not isinstance(released_sessions, list):
        errors.append("release capture_session_ids must be a list")
    if declared_sessions != observed_sessions:
        errors.append("release capture_session_ids do not match manifest")
    if release.get("sample_count") != len(sample_ids):
        errors.append("release sample_count does not match manifest")
    if release.get("source_image_count") != len(source_paths):
        errors.append("release source_image_count does not match manifest")

    return {
        "schema_version": 1,
        "state": "complete",
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
        "dataset_id": release.get("dataset_id"),
        "sample_count": len(sample_ids),
        "source_image_count": len(source_paths),
        "class_counts": {
            name: int(class_counts[name]) for name in class_names
        },
        "class_source_image_counts": {
            name: len(class_sources[name]) for name in class_names
        },
        "acquisition_batch_count": len(observed_batches),
        "capture_session_count": len(observed_sessions),
        "sensor_record_count": len(sensor_by_id),
        "development_overlap": {
            "sample_id_count": len(sample_ids & development_sample_ids),
            "source_image_id_count": len(
                set(source_paths) & development_source_ids
            ),
            "exact_image_hash_count": len(
                set(source_hashes.values()) & development_hashes
            ),
            "severe_perceptual_candidate_count": severe_near_duplicates,
            "manual_review_candidate_count": review_near_duplicates,
        },
        "bindings": {
            name: {
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
    }
