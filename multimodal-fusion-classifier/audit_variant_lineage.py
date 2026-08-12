"""Audit the locked configuration lineage used in the paper evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
VARIANT_FILES = {
    "full": "full.json",
    "v2b": "full_quality_aware_prior.json",
    "v3": "full_cross_modal_quality.json",
    "balanced": "full_cross_modal_quality_balanced.json",
}
COMPARISON_ORDER = (
    ("full", "v2b"),
    ("v2b", "v3"),
    ("v3", "balanced"),
)
METADATA_PREFIXES = ("experiment.",)
METADATA_PATHS = {"training.checkpoint_dir"}
_MISSING = object()

EXPECTED_SCIENTIFIC_CHANGES = {
    ("full", "v2b"): {
        "loss.calibration_clean_weight": (1.0, 3.0),
        "loss.calibration_weight": (0.1, 0.3),
        "loss.reconstruction_weight": (0.2, 0.1),
        "loss.variance_prior_weight": (0.0, 0.1),
        "model.sensor_grid_quality_enabled": (False, True),
        "model.sensor_log_variance_min": (-6.0, -3.0),
        "robustness.sensor_degradation.bias_probability": (0.15, 0.25),
        "robustness.sensor_degradation.bias_strength": (0.08, 0.15),
        "robustness.sensor_degradation.missing_probability": (0.05, 0.1),
        "robustness.sensor_degradation.noise_probability": (0.2, 0.5),
        "robustness.sensor_degradation.noise_std_range": (
            _MISSING,
            [0.02, 0.15],
        ),
        "robustness.sensor_degradation.stuck_probability": (0.05, 0.15),
    },
    ("v2b", "v3"): {
        "loss.calibration_ranking_margin": (_MISSING, 0.05),
        "loss.calibration_ranking_weight": (_MISSING, 0.05),
        "loss.sensor_grid_reconstruction_weight": (_MISSING, 0.5),
        "model.sensor_cross_modal_quality_enabled": (_MISSING, True),
        "model.sensor_cross_modal_quality_hidden_dim": (_MISSING, 32),
        "model.sensor_grid_decoder_hidden_dim": (_MISSING, 64),
    },
    ("v3", "balanced"): {
        "loss.class_weighting.apply_to_ce": (_MISSING, True),
        "loss.class_weighting.apply_to_focal": (_MISSING, False),
        "loss.class_weighting.method": (_MISSING, "sqrt_inverse"),
        "training.selection_metric": (_MISSING, "macro_f1"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the full-to-v2b-to-v3 configuration lineage"
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ablations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_ROOT / "VARIANT_LINEAGE_AUDIT.json",
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


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(value[key], path))
    return flattened


def _is_metadata(path: str) -> bool:
    return path in METADATA_PATHS or path.startswith(METADATA_PREFIXES)


def _encoded_value(value: Any) -> dict[str, Any]:
    if value is _MISSING:
        return {"present": False}
    return {"present": True, "value": value}


def _change_rows(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scientific: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in sorted(set(reference) | set(candidate)):
        before = reference.get(path, _MISSING)
        after = candidate.get(path, _MISSING)
        if before == after:
            continue
        row = {
            "path": path,
            "before": _encoded_value(before),
            "after": _encoded_value(after),
        }
        (metadata if _is_metadata(path) else scientific).append(row)
    return scientific, metadata


def _validate_expected_changes(
    reference_name: str,
    candidate_name: str,
    scientific: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    expected = EXPECTED_SCIENTIFIC_CHANGES[
        (reference_name, candidate_name)
    ]
    actual_paths = {str(row["path"]) for row in scientific}
    expected_paths = set(expected)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        if missing:
            errors.append(
                f"{reference_name}->{candidate_name} missing expected changes: "
                + ", ".join(missing)
            )
        if unexpected:
            errors.append(
                f"{reference_name}->{candidate_name} has unexpected changes: "
                + ", ".join(unexpected)
            )
    rows = {str(row["path"]): row for row in scientific}
    for path, (before, after) in expected.items():
        row = rows.get(path)
        if row is None:
            continue
        if row["before"] != _encoded_value(before):
            errors.append(
                f"{reference_name}->{candidate_name} before value drift: {path}"
            )
        if row["after"] != _encoded_value(after):
            errors.append(
                f"{reference_name}->{candidate_name} after value drift: {path}"
            )
    return errors


def build_variant_lineage(config_root: Path) -> dict[str, Any]:
    config_root = _resolve(config_root)
    errors: list[str] = []
    variants: dict[str, dict[str, Any]] = {}
    flattened: dict[str, dict[str, Any]] = {}

    for name, filename in VARIANT_FILES.items():
        path = config_root / filename
        if not path.is_file():
            errors.append(f"variant config is missing: {name}: {path}")
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(
                f"variant config cannot be parsed: {name}: {type(exc).__name__}"
            )
            continue
        if not isinstance(config, dict):
            errors.append(f"variant config root must be an object: {name}")
            continue
        flat = _flatten(config)
        flattened[name] = flat
        variants[name] = {
            "path": _display_path(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "leaf_count": len(flat),
            "experiment_name": config.get("experiment", {}).get("name"),
            "checkpoint_dir": config.get("training", {}).get(
                "checkpoint_dir"
            ),
        }
        expected_experiment = {
            "full": "full",
            "v2b": "full_quality_aware_prior",
            "v3": "full_cross_modal_quality",
            "balanced": "full_cross_modal_quality_balanced",
        }[name]
        if variants[name]["experiment_name"] != expected_experiment:
            errors.append(f"experiment name mismatch: {name}")

    comparisons: list[dict[str, Any]] = []
    for reference_name, candidate_name in COMPARISON_ORDER:
        if reference_name not in flattened or candidate_name not in flattened:
            continue
        scientific, metadata = _change_rows(
            flattened[reference_name],
            flattened[candidate_name],
        )
        comparison_errors = _validate_expected_changes(
            reference_name,
            candidate_name,
            scientific,
        )
        errors.extend(comparison_errors)
        comparisons.append(
            {
                "reference": reference_name,
                "candidate": candidate_name,
                "scientific_change_count": len(scientific),
                "metadata_change_count": len(metadata),
                "scientific_changes": scientific,
                "metadata_changes": metadata,
                "valid": not comparison_errors,
            }
        )

    common_leaves: dict[str, Any] = {}
    if len(flattened) == len(VARIANT_FILES):
        shared_paths = set.intersection(
            *(set(value) for value in flattened.values())
        )
        for path in sorted(shared_paths):
            if _is_metadata(path):
                continue
            values = [flattened[name][path] for name in VARIANT_FILES]
            if all(value == values[0] for value in values[1:]):
                common_leaves[path] = values[0]

    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "lineage": list(VARIANT_FILES),
        "variants": variants,
        "comparisons": comparisons,
        "shared_invariants": {
            "leaf_count": len(common_leaves),
            "sha256": _json_sha256(common_leaves),
            "paths": sorted(common_leaves),
        },
        "interpretation_boundary": (
            "This is a raw locked-config lineage audit. It prevents variant "
            "mislabeling, but independent-training differences are not "
            "single-component causal estimates. Absent keys inherit code "
            "defaults and are recorded as absent rather than silently filled."
        ),
    }


def _format_value(encoded: Mapping[str, Any]) -> str:
    if not encoded.get("present"):
        return "`<absent>`"
    return f"`{json.dumps(encoded.get('value'), ensure_ascii=False)}`"


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Variant Lineage Audit",
        "",
        f"- State: `{report['state']}`.",
        f"- Config variants: {len(report['variants'])}.",
        (
            "- Shared scientific leaves: "
            f"{report['shared_invariants']['leaf_count']}, SHA-256 "
            f"`{report['shared_invariants']['sha256']}`."
        ),
        "",
        "## Locked Lineage",
        "",
        "| Alias | Config | SHA-256 |",
        "|---|---|---|",
    ]
    for name in report["lineage"]:
        variant = report["variants"].get(name)
        if variant is None:
            continue
        lines.append(
            f"| `{name}` | `{variant['path']}` | `{variant['sha256']}` |"
        )
    for comparison in report["comparisons"]:
        lines.extend(
            [
                "",
                (
                    f"## {comparison['reference']} to "
                    f"{comparison['candidate']}"
                ),
                "",
                "| Scientific config path | Before | After |",
                "|---|---:|---:|",
            ]
        )
        for row in comparison["scientific_changes"]:
            lines.append(
                f"| `{row['path']}` | {_format_value(row['before'])} | "
                f"{_format_value(row['after'])} |"
            )
        lines.append("")
        lines.append(
            f"Metadata-only changes: {comparison['metadata_change_count']}."
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            str(report["interpretation_boundary"]),
            "",
        ]
    )
    if report["errors"]:
        lines.extend(
            [
                "## Errors",
                "",
                *[f"- {error}" for error in report["errors"]],
                "",
            ]
        )
    _atomic_write(path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    report = build_variant_lineage(args.config_root)
    output = _resolve(args.output)
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = output.with_suffix(".md")
    _write_markdown(report, markdown)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "variant_count": len(report["variants"]),
                "comparison_count": len(report["comparisons"]),
                "error_count": report["error_count"],
                "output": output.as_posix(),
                "markdown": markdown.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
