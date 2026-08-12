"""Convert synchronized raw 4x4 hardware observations to model sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from mmfc.physical_sensor import calibrate_physical_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare calibrated physical 4x4 sensor sidecars"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-skew-ms", type=float, default=100.0)
    parser.add_argument(
        "--max-calibration-age-days",
        type=float,
        default=365.0,
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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
        raise ValueError("physical observation JSONL is empty")
    return rows


def build_sidecars(
    input_path: Path,
    output_path: Path,
    report_path: Path | None = None,
    *,
    max_skew_ms: float = 100.0,
    max_calibration_age_days: float = 365.0,
) -> dict[str, Any]:
    input_path = _resolve(input_path)
    output_path = _resolve(output_path)
    report_path = (
        _resolve(report_path)
        if report_path is not None
        else output_path.with_name("BUILD_REPORT.json")
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"physical input does not exist: {input_path}")
    occupied = [path for path in (output_path, report_path) if path.exists()]
    if occupied:
        raise FileExistsError(
            "refusing to overwrite physical sidecar outputs: "
            + ", ".join(str(path) for path in occupied)
        )
    source_rows = _read_jsonl(input_path)
    sidecars = []
    seen = set()
    for index, record in enumerate(source_rows, start=1):
        sample_id = str(record.get("sample_id", ""))
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id at input row {index}: {sample_id}")
        seen.add(sample_id)
        try:
            sidecars.append(
                calibrate_physical_observation(
                    record,
                    max_skew_ms=max_skew_ms,
                    max_calibration_age_days=max_calibration_age_days,
                )
            )
        except ValueError as exc:
            raise ValueError(f"physical input row {index}: {exc}") from exc
    _atomic_text(
        output_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in sidecars
        ),
    )
    clipping_counts = [
        int(row["source"]["clipped_cell_count"]) for row in sidecars
    ]
    skews = [float(row["source"]["time_skew_ms"]) for row in sidecars]
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "physical_4x4_sensor_sidecar_preparation",
        "sample_count": len(sidecars),
        "unique_sample_count": len(seen),
        "source_mode": "physical-array-v1",
        "physical_hardware_origin_declared": True,
        "physical_hardware_origin_verified": False,
        "max_skew_ms": float(max_skew_ms),
        "observed_max_skew_ms": max(skews),
        "max_calibration_age_days": float(max_calibration_age_days),
        "samples_with_clipping": sum(count > 0 for count in clipping_counts),
        "total_clipped_cells": sum(clipping_counts),
        "input": {
            "path": input_path.as_posix(),
            "size_bytes": input_path.stat().st_size,
            "sha256": _sha256(input_path),
        },
        "output": {
            "path": output_path.as_posix(),
            "size_bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "claim_boundary": (
            "This report validates software ingestion, calibration, and "
            "timestamp alignment only. Hardware performance is established "
            "only when the input records originate from an actual synchronized "
            "array and the acquisition procedure is independently documented."
        ),
        "errors": [],
    }
    _atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> None:
    args = parse_args()
    report = build_sidecars(
        args.input,
        args.output,
        args.report,
        max_skew_ms=args.max_skew_ms,
        max_calibration_age_days=args.max_calibration_age_days,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "sample_count": report["sample_count"],
                "observed_max_skew_ms": report["observed_max_skew_ms"],
                "samples_with_clipping": report["samples_with_clipping"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
