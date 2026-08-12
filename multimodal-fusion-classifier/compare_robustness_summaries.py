"""Compare two matched robustness multiseed summaries descriptively."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIOS = (
    "clean",
    "missing_100",
    "noise_015",
    "bias_008",
    "stuck",
    "mixed",
)
EXPECTED_SEEDS = [2026, 2027, 2028, 2029, 2030]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched robustness summary files"
    )
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def compare_summaries(
    full: Mapping[str, Any],
    ablation: Mapping[str, Any],
    *,
    scenarios: Sequence[str] = SCENARIOS,
) -> dict[str, Any]:
    if (
        full.get("seeds") != EXPECTED_SEEDS
        or ablation.get("seeds") != EXPECTED_SEEDS
        or full.get("seed_count") != 5
        or ablation.get("seed_count") != 5
    ):
        raise ValueError("robustness degradation-seed identity mismatch")
    full_rows = {
        item["scenario"]: item for item in full.get("summary", [])
    }
    ablation_rows = {
        item["scenario"]: item for item in ablation.get("summary", [])
    }
    rows = []
    for scenario in scenarios:
        if scenario not in full_rows or scenario not in ablation_rows:
            raise ValueError(f"missing robustness scenario: {scenario}")
        full_row = full_rows[scenario]
        ablation_row = ablation_rows[scenario]
        if (
            full_row.get("seed_count") != 5
            or ablation_row.get("seed_count") != 5
        ):
            raise ValueError(f"scenario seed count mismatch: {scenario}")
        rows.append(
            {
                "scenario": scenario,
                "full_accuracy_mean": full_row["accuracy_mean"],
                "ablation_accuracy_mean": ablation_row["accuracy_mean"],
                "full_minus_ablation_accuracy_pp": (
                    full_row["accuracy_mean"]
                    - ablation_row["accuracy_mean"]
                )
                * 100.0,
                "full_macro_f1_mean": full_row["macro_f1_mean"],
                "ablation_macro_f1_mean": ablation_row["macro_f1_mean"],
                "full_minus_ablation_macro_f1_pp": (
                    full_row["macro_f1_mean"]
                    - ablation_row["macro_f1_mean"]
                )
                * 100.0,
                "full_ece_mean": full_row["ece_mean"],
                "ablation_ece_mean": ablation_row["ece_mean"],
                "ablation_minus_full_ece_pp": (
                    ablation_row["ece_mean"] - full_row["ece_mean"]
                )
                * 100.0,
            }
        )
    return {
        "valid": True,
        "training_seed": 42,
        "degradation_seeds": EXPECTED_SEEDS,
        "scenario_count": len(rows),
        "rows": rows,
        "boundaries": [
            (
                "This is a descriptive seed-42 training comparison over five "
                "matched software-degradation seeds."
            ),
            (
                "Degradation seeds are repeated perturbations, not additional "
                "independent model-training seeds."
            ),
        ],
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    full_path = _resolve(args.full)
    ablation_path = _resolve(args.ablation)
    report = compare_summaries(
        _read_json(full_path),
        _read_json(ablation_path),
    )
    report["source_bindings"] = {
        "comparison_source": _binding(Path(__file__)),
        "full_summary": _binding(full_path),
        "ablation_summary": _binding(ablation_path),
    }
    output = _resolve(args.output)
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _write_csv(output.with_suffix(".csv"), report["rows"])
    _atomic_write(
        output.with_suffix(".md"),
        "\n".join(
            [
                "# v3 reliability-subsystem robustness comparison",
                "",
                "| Scenario | Accuracy effect (pp) | Macro F1 effect (pp) | ECE benefit (pp) |",
                "|---|---:|---:|---:|",
                *[
                    (
                        f"| {row['scenario']} | "
                        f"{row['full_minus_ablation_accuracy_pp']:+.2f} | "
                        f"{row['full_minus_ablation_macro_f1_pp']:+.2f} | "
                        f"{row['ablation_minus_full_ece_pp']:+.2f} |"
                    )
                    for row in report["rows"]
                ],
                "",
                "Positive Accuracy/Macro F1 effects favor the full reliability "
                "subsystem. Positive ECE benefit means lower ECE for the full "
                "system.",
                "",
            ]
        ),
    )
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "scenario_count": report["scenario_count"],
                "output": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
