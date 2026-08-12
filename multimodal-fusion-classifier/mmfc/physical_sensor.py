"""Validation and calibration for synchronized physical 4x4 sensor records."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


PROPERTY_NAMES = (
    "weight_g",
    "temperature_c",
    "moisture_pct",
    "conductivity_us_cm",
    "odor_index",
)
PROPERTY_RANGES = {
    "weight_g": (0.1, 5000.0),
    "temperature_c": (-20.0, 80.0),
    "moisture_pct": (0.0, 100.0),
    "conductivity_us_cm": (0.0, 5000.0),
    "odor_index": (0.0, 1.0),
}
FORBIDDEN_SUPERVISION_FIELDS = {
    "label",
    "labels",
    "class",
    "class_id",
    "class_name",
    "target",
    "target_id",
    "annotation",
    "annotations",
}
EXPECTED_RECORD_FIELDS = {
    "sample_id",
    "source_image_id",
    "sensor_id",
    "measurement_unit",
    "image_timestamp",
    "sensor_timestamp",
    "raw_grid",
    "properties",
    "calibration",
}
EXPECTED_CALIBRATION_FIELDS = {
    "calibration_id",
    "calibrated_at",
    "zero_grid",
    "full_scale_grid",
}


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _grid(value: object, field: str) -> np.ndarray:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise ValueError(f"{field} must be a strict 4x4 list")
    rows = []
    for row in value:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 4
        ):
            raise ValueError(f"{field} must be a strict 4x4 list")
        parsed_row = []
        for item in row:
            if isinstance(item, bool):
                raise ValueError(f"{field} values must be numeric")
            try:
                number = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} values must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{field} values must be finite")
            parsed_row.append(number)
        rows.append(parsed_row)
    return np.asarray(rows, dtype=np.float64)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _properties(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("properties must be an object")
    unexpected = sorted(set(value) - set(PROPERTY_NAMES))
    if unexpected:
        raise ValueError(
            "properties contains unexpected fields: " + ", ".join(unexpected)
        )
    result = {}
    for name in PROPERTY_NAMES:
        item = value.get(name)
        if isinstance(item, bool):
            raise ValueError(f"property {name} must be numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"property {name} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"property {name} must be finite")
        lower, upper = PROPERTY_RANGES[name]
        if not lower <= number <= upper:
            raise ValueError(
                f"property {name} must be in [{lower}, {upper}]"
            )
        result[name] = number
    return result


def calibrate_physical_observation(
    record: Mapping[str, Any],
    *,
    max_skew_ms: float = 100.0,
    max_calibration_age_days: float = 365.0,
) -> dict[str, Any]:
    """Convert one synchronized raw hardware record to the existing sidecar schema.

    Calibration is cell-wise ``(raw - zero) / (full_scale - zero)`` followed
    by clipping to ``[0, 1]``. Labels and annotations are explicitly rejected.
    """

    if not isinstance(record, Mapping):
        raise ValueError("physical observation must be an object")
    if max_skew_ms < 0.0 or not math.isfinite(max_skew_ms):
        raise ValueError("max_skew_ms must be finite and non-negative")
    if (
        max_calibration_age_days <= 0.0
        or not math.isfinite(max_calibration_age_days)
    ):
        raise ValueError(
            "max_calibration_age_days must be finite and positive"
        )
    forbidden = sorted(
        str(field)
        for field in record
        if str(field).casefold() in FORBIDDEN_SUPERVISION_FIELDS
    )
    if forbidden:
        raise ValueError(
            "physical observation must not contain supervision fields: "
            + ", ".join(forbidden)
        )
    unexpected = sorted(str(field) for field in set(record) - EXPECTED_RECORD_FIELDS)
    if unexpected:
        raise ValueError(
            "physical observation contains unexpected fields: "
            + ", ".join(unexpected)
        )

    sample_id = _identifier(record.get("sample_id"), "sample_id")
    source_image_id = _identifier(
        record.get("source_image_id"), "source_image_id"
    )
    sensor_id = _identifier(record.get("sensor_id"), "sensor_id")
    measurement_unit = _identifier(
        record.get("measurement_unit"), "measurement_unit"
    )
    image_time = _timestamp(
        record.get("image_timestamp"), "image_timestamp"
    )
    sensor_time = _timestamp(
        record.get("sensor_timestamp"), "sensor_timestamp"
    )
    skew_ms = abs((sensor_time - image_time).total_seconds()) * 1000.0
    if skew_ms > max_skew_ms:
        raise ValueError(
            f"image/sensor time skew {skew_ms:.3f} ms exceeds "
            f"{max_skew_ms:.3f} ms"
        )

    calibration = record.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration must be an object")
    unexpected_calibration = sorted(
        str(field)
        for field in set(calibration) - EXPECTED_CALIBRATION_FIELDS
    )
    if unexpected_calibration:
        raise ValueError(
            "calibration contains unexpected fields: "
            + ", ".join(unexpected_calibration)
        )
    calibration_id = _identifier(
        calibration.get("calibration_id"), "calibration.calibration_id"
    )
    calibrated_at = _timestamp(
        calibration.get("calibrated_at"), "calibration.calibrated_at"
    )
    age_days = (sensor_time - calibrated_at).total_seconds() / 86400.0
    if age_days < 0.0:
        raise ValueError("sensor timestamp precedes calibration timestamp")
    if age_days > max_calibration_age_days:
        raise ValueError(
            f"calibration age {age_days:.3f} days exceeds "
            f"{max_calibration_age_days:.3f} days"
        )

    raw = _grid(record.get("raw_grid"), "raw_grid")
    zero = _grid(calibration.get("zero_grid"), "calibration.zero_grid")
    full = _grid(
        calibration.get("full_scale_grid"),
        "calibration.full_scale_grid",
    )
    span = full - zero
    if np.any(span <= 0.0):
        raise ValueError(
            "calibration.full_scale_grid must exceed zero_grid in every cell"
        )
    unbounded = (raw - zero) / span
    clipping_mask = (unbounded < 0.0) | (unbounded > 1.0)
    normalized = np.clip(unbounded, 0.0, 1.0)
    normalized_grid = [
        [round(float(item), 8) for item in row]
        for row in normalized.tolist()
    ]
    properties = _properties(record.get("properties"))
    calibration_payload = {
        "calibration_id": calibration_id,
        "calibrated_at": calibrated_at.isoformat(),
        "zero_grid": zero.tolist(),
        "full_scale_grid": full.tolist(),
    }
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "grid_shape": [4, 4],
        "grid": normalized_grid,
        "normalized_grid": normalized_grid,
        "properties": properties,
        "source": {
            "mode": "physical-array-v1",
            "generator": "hardware-calibrated-v1",
            "source_image_id": source_image_id,
            "sensor_id": sensor_id,
            "measurement_unit": measurement_unit,
            "image_timestamp_utc": image_time.isoformat(),
            "sensor_timestamp_utc": sensor_time.isoformat(),
            "time_skew_ms": round(skew_ms, 6),
            "max_time_skew_ms": float(max_skew_ms),
            "calibration_id": calibration_id,
            "calibration_age_days": round(age_days, 6),
            "max_calibration_age_days": float(max_calibration_age_days),
            "raw_grid_sha256": _canonical_sha256(raw.tolist()),
            "calibration_sha256": _canonical_sha256(calibration_payload),
            "clipped_cell_count": int(clipping_mask.sum()),
            "clipping_fraction": float(clipping_mask.mean()),
            "annotation_inputs": False,
            "class_inputs": False,
            "physical_hardware_origin_declared": True,
            "physical_hardware_origin_verified": False,
        },
    }
