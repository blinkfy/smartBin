"""Release physical sidecars for the frozen external-blind protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmfc.physical_release import release_sidecars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert calibrated physical sidecars to the preregistered "
            "synchronized-hardware release mode"
        )
    )
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = release_sidecars(
        args.sidecars,
        args.build_report,
        args.manifest,
        args.attestation,
        args.output,
        args.report,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "sample_count": report["sample_count"],
                "released_mode": report["released_mode"],
                "hardware_performance_validated": report[
                    "hardware_performance_validated"
                ],
                "external_blind_release_json_still_required": report[
                    "external_blind_release_json_still_required"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
