"""Independently verify cross-artifact consistency of the recovery outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability-workers0"
)
DEFAULT_OUTCOME = REPORT_ROOT / "OUTCOME.json"
DEFAULT_OUTPUT = REPORT_ROOT / "OUTCOME_VERIFICATION.json"
METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "negative_log_likelihood",
    "multiclass_brier_score",
    "ece",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the workers=0 recovery outcome"
    )
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _verify_bindings(
    bindings: Mapping[str, Any],
    errors: list[str],
) -> None:
    for name, raw in bindings.items():
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            errors.append(f"malformed outcome binding: {name}")
            continue
        path = _resolve(Path(raw["path"]))
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"outcome binding escapes project: {name}")
            continue
        if not path.is_file():
            errors.append(f"outcome binding missing: {name}")
            continue
        if path.stat().st_size != raw.get("size_bytes"):
            errors.append(f"outcome binding size mismatch: {name}")
        if _sha256(path) != raw.get("sha256"):
            errors.append(f"outcome binding hash mismatch: {name}")


def verify_outcome(outcome_path: Path = DEFAULT_OUTCOME) -> dict[str, Any]:
    outcome_path = _resolve(outcome_path)
    errors: list[str] = []
    outcome = _read_json(outcome_path)
    if (
        outcome.get("schema_version") != 2
        or outcome.get("state") != "complete"
        or outcome.get("valid") is not True
        or outcome.get("errors") != []
        or outcome.get("training", {}).get("seeds") != [42, 43, 44]
        or outcome.get("training", {}).get("num_workers") != 0
        or outcome.get("training", {}).get("resume_used") is not False
        or outcome.get("recovery", {}).get("workers4_partial_run_reused")
        is not False
        or outcome.get("recovery", {}).get("workers0_fresh_epoch1_runs")
        is not True
    ):
        errors.append("recovery outcome identity mismatch")
    bindings = outcome.get("source_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        errors.append("outcome source bindings are missing")
        bindings = {}
    _verify_bindings(bindings, errors)

    validation_summary = outcome.get("ablation_validation_summary", {})
    test_summary = outcome.get("ablation_test_summary", {})
    matched_summary = outcome.get("matched_test_comparison", {})
    for split, summary in (
        ("val", validation_summary),
        ("test", test_summary),
    ):
        individuals = summary.get("individual", [])
        if [row.get("name") for row in individuals] != [
            "seed42",
            "seed43",
            "seed44",
        ]:
            errors.append(f"{split} summary run order mismatch")
            continue
        for seed, row in zip((42, 43, 44), individuals):
            evaluation_path = REPORT_ROOT / f"seed{seed}" / split / "evaluation.json"
            evaluation = _read_json(evaluation_path)
            if row.get("sample_count") != 826:
                errors.append(f"seed {seed} {split} summary sample count mismatch")
            for metric in METRICS:
                if row.get("metrics", {}).get(metric) != (
                    evaluation.get("metrics", {}).get(metric)
                ):
                    errors.append(
                        f"seed {seed} {split} summary metric mismatch: {metric}"
                    )

    matched_individuals = matched_summary.get("individual", [])
    if [row.get("name") for row in matched_individuals] != [
        "seed42",
        "seed43",
        "seed44",
    ]:
        errors.append("matched comparison run order mismatch")
    else:
        for seed, row in zip((42, 43, 44), matched_individuals):
            comparison = _read_json(
                REPORT_ROOT / f"seed{seed}" / "MATCHED_TEST_COMPARISON.json"
            )
            points = comparison.get("paired_bootstrap", {}).get(
                "differences", {}
            )
            for metric in METRICS:
                if row.get("differences", {}).get(metric) != (
                    points.get(metric, {}).get("point")
                ):
                    errors.append(
                        f"seed {seed} matched summary mismatch: {metric}"
                    )
    runs = outcome.get("training", {}).get("runs", [])
    if [row.get("seed") for row in runs] != [42, 43, 44]:
        errors.append("training run order mismatch")
    for row in runs:
        seed = row.get("seed")
        if seed not in (42, 43, 44):
            continue
        audit = _read_json(REPORT_ROOT / f"seed{seed}" / "CHECKPOINT_AUDIT.json")
        if (
            row.get("best_checkpoint_sha256")
            != audit.get("best", {}).get("sha256")
            or row.get("last_checkpoint_sha256")
            != audit.get("last", {}).get("sha256")
            or audit.get("valid") is not True
        ):
            errors.append(f"seed {seed} checkpoint outcome mismatch")
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "recovery_outcome_cross_artifact_verification",
        "outcome": {
            "path": outcome_path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": outcome_path.stat().st_size,
            "sha256": _sha256(outcome_path),
        },
        "source_binding_count": len(bindings),
        "training_seed_count": 3,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    report = verify_outcome(args.outcome)
    output_path = _resolve(args.output)
    _atomic_write(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# v3 reliability-ablation recovery outcome verification",
                "",
                f"- Valid: {report['valid']}",
                f"- Source bindings checked: {report['source_binding_count']}",
                f"- Error count: {len(report['errors'])}",
                (
                    "- Validation/test summaries, paired comparisons, and "
                    "checkpoint hashes were cross-checked against source files."
                ),
                "",
            ]
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["valid"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
