"""Build a source-bound post-hoc error analysis from saved predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = {
    "config": "configs/ablations/full_cross_modal_quality.json",
    "manifest": "data/qwen-material/test.jsonl",
    "reference_predictions": (
        "reports/qwen-material/full_quality_aware_prior/test/"
        "predictions.jsonl"
    ),
    "candidate_predictions": (
        "reports/qwen-material/full_cross_modal_quality/test/"
        "predictions.jsonl"
    ),
}
DEFAULT_GATE_SOURCES = {
    "config": "configs/ablations/full_cross_modal_quality.json",
    "manifest": "data/qwen-material/test.jsonl",
    "reference_predictions": (
        "reports/qwen-material/full_cross_modal_quality/"
        "posthoc_no_reliability/test/predictions.jsonl"
    ),
    "candidate_predictions": (
        "reports/qwen-material/full_cross_modal_quality/test/"
        "predictions.jsonl"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze aligned v2b/v3 prediction errors without inference"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_SOURCES["config"],
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_SOURCES["manifest"],
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_SOURCES["reference_predictions"],
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_SOURCES["candidate_predictions"],
    )
    parser.add_argument("--reference-name", default="v2b")
    parser.add_argument("--candidate-name", default="v3")
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "ERROR_ANALYSIS.json"
        ),
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row must be an object: {path}:{line_number}"
                )
            records.append(value)
    return records


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


def _index_records(
    records: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{name} contains a missing sample_id")
        if sample_id in indexed:
            raise ValueError(f"{name} contains duplicate sample_id: {sample_id}")
        indexed[sample_id] = record
    return indexed


def _confidence_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q10": None,
            "q90": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _counter_rows(
    counter: Counter[tuple[int, ...]],
    class_names: Sequence[str],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]:
        row: dict[str, Any] = {"count": int(count)}
        if len(key) == 2:
            row.update(
                {
                    "target_id": key[0],
                    "target": class_names[key[0]],
                    "predicted_id": key[1],
                    "predicted": class_names[key[1]],
                }
            )
        elif len(key) == 3:
            row.update(
                {
                    "target_id": key[0],
                    "target": class_names[key[0]],
                    "reference_predicted_id": key[1],
                    "reference_predicted": class_names[key[1]],
                    "candidate_predicted_id": key[2],
                    "candidate_predicted": class_names[key[2]],
                }
            )
        else:
            raise ValueError("unsupported counter key length")
        rows.append(row)
    return rows


def _model_summary(
    rows: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    correct_confidence: list[float] = []
    error_confidence: list[float] = []
    confusions: Counter[tuple[int, int]] = Counter()
    supports = np.zeros(len(class_names), dtype=np.int64)
    errors = np.zeros(len(class_names), dtype=np.int64)
    for row in rows:
        target = int(row["target_id"])
        predicted = int(row["predicted_id"])
        confidence = float(row["confidence"])
        supports[target] += 1
        if predicted == target:
            correct_confidence.append(confidence)
        else:
            error_confidence.append(confidence)
            errors[target] += 1
            confusions[(target, predicted)] += 1
    sample_count = len(rows)
    error_count = len(error_confidence)
    return {
        "sample_count": sample_count,
        "correct_count": sample_count - error_count,
        "error_count": error_count,
        "error_rate": float(error_count / max(sample_count, 1)),
        "confidence": {
            "correct": _confidence_summary(correct_confidence),
            "error": _confidence_summary(error_confidence),
            "high_confidence_error_count_ge_0_8": int(
                np.count_nonzero(np.asarray(error_confidence) >= 0.8)
            ),
            "high_confidence_error_count_ge_0_9": int(
                np.count_nonzero(np.asarray(error_confidence) >= 0.9)
            ),
        },
        "top_directional_confusions": _counter_rows(
            confusions,
            class_names,
            top_k=top_k,
        ),
        "per_class_error": [
            {
                "label_id": index,
                "label": class_names[index],
                "support": int(supports[index]),
                "error_count": int(errors[index]),
                "recall": float(
                    (supports[index] - errors[index]) / supports[index]
                )
                if supports[index]
                else None,
            }
            for index in range(len(class_names))
        ],
    }


def analyze_prediction_records(
    reference_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    manifest_records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    reference_name: str = "v2b",
    candidate_name: str = "v3",
    top_k: int = 12,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    class_names = tuple(str(value) for value in class_names)
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must contain unique classes")
    reference = _index_records(reference_records, name=reference_name)
    candidate = _index_records(candidate_records, name=candidate_name)
    manifest = _index_records(manifest_records, name="manifest")
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate sample_id sets differ")
    if set(reference) != set(manifest):
        raise ValueError("manifest and prediction sample_id sets differ")

    ordered_ids = [
        str(record["sample_id"])
        for record in manifest_records
    ]
    normalized_reference: list[dict[str, Any]] = []
    normalized_candidate: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    source_ids: list[str] = []
    for sample_id in ordered_ids:
        manifest_row = manifest[sample_id]
        reference_row = reference[sample_id]
        candidate_row = candidate[sample_id]
        source_id = str(
            manifest_row.get("source_image_id", sample_id)
        )
        source_ids.append(source_id)
        target = int(reference_row["target_id"])
        if int(candidate_row["target_id"]) != target:
            raise ValueError(f"target mismatch: {sample_id}")
        if target < 0 or target >= len(class_names):
            raise ValueError(f"target_id out of range: {sample_id}")

        normalized: list[dict[str, Any]] = []
        for name, row in (
            (reference_name, reference_row),
            (candidate_name, candidate_row),
        ):
            predicted = int(row["predicted_id"])
            confidence = float(row["confidence"])
            if predicted < 0 or predicted >= len(class_names):
                raise ValueError(
                    f"predicted_id out of range: {name}: {sample_id}"
                )
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"invalid confidence: {name}: {sample_id}"
                )
            expected_correct = predicted == target
            if bool(row.get("correct")) != expected_correct:
                raise ValueError(
                    f"correct flag mismatch: {name}: {sample_id}"
                )
            normalized.append(
                {
                    "sample_id": sample_id,
                    "source_image_id": source_id,
                    "target_id": target,
                    "predicted_id": predicted,
                    "confidence": confidence,
                    "correct": expected_correct,
                }
            )
        normalized_reference.append(normalized[0])
        normalized_candidate.append(normalized[1])
        paired_rows.append(
            {
                "sample_id": sample_id,
                "source_image_id": source_id,
                "target_id": target,
                "reference_predicted_id": normalized[0]["predicted_id"],
                "candidate_predicted_id": normalized[1]["predicted_id"],
                "reference_confidence": normalized[0]["confidence"],
                "candidate_confidence": normalized[1]["confidence"],
                "reference_correct": normalized[0]["correct"],
                "candidate_correct": normalized[1]["correct"],
            }
        )

    both_correct = [
        row for row in paired_rows
        if row["reference_correct"] and row["candidate_correct"]
    ]
    corrected = [
        row for row in paired_rows
        if not row["reference_correct"] and row["candidate_correct"]
    ]
    regressed = [
        row for row in paired_rows
        if row["reference_correct"] and not row["candidate_correct"]
    ]
    both_wrong = [
        row for row in paired_rows
        if not row["reference_correct"] and not row["candidate_correct"]
    ]
    changed_wrong = [
        row for row in both_wrong
        if row["reference_predicted_id"] != row["candidate_predicted_id"]
    ]
    transition_counter = Counter(
        (
            int(row["target_id"]),
            int(row["reference_predicted_id"]),
            int(row["candidate_predicted_id"]),
        )
        for row in paired_rows
        if row["reference_predicted_id"] != row["candidate_predicted_id"]
    )
    corrected_by_target = Counter(
        int(row["target_id"]) for row in corrected
    )
    regressed_by_target = Counter(
        int(row["target_id"]) for row in regressed
    )

    def example_rows(
        rows: Sequence[Mapping[str, Any]],
        *,
        descending_key: str,
    ) -> list[dict[str, Any]]:
        selected = sorted(
            rows,
            key=lambda row: (
                -float(row[descending_key]),
                str(row["sample_id"]),
            ),
        )[:top_k]
        return [
            {
                "sample_id": row["sample_id"],
                "source_image_id": row["source_image_id"],
                "target": class_names[int(row["target_id"])],
                "reference_predicted": class_names[
                    int(row["reference_predicted_id"])
                ],
                "candidate_predicted": class_names[
                    int(row["candidate_predicted_id"])
                ],
                "reference_confidence": float(row["reference_confidence"]),
                "candidate_confidence": float(row["candidate_confidence"]),
            }
            for row in selected
        ]

    sample_count = len(paired_rows)
    return {
        "dataset": {
            "sample_count": sample_count,
            "source_image_count": len(set(source_ids)),
            "class_count": len(class_names),
            "top_k": int(top_k),
        },
        "models": {
            reference_name: _model_summary(
                normalized_reference,
                class_names,
                top_k=top_k,
            ),
            candidate_name: _model_summary(
                normalized_candidate,
                class_names,
                top_k=top_k,
            ),
        },
        "paired_outcomes": {
            "both_correct": len(both_correct),
            "candidate_only_correct": len(corrected),
            "reference_only_correct": len(regressed),
            "both_wrong": len(both_wrong),
            "both_wrong_same_prediction": len(both_wrong) - len(changed_wrong),
            "both_wrong_changed_prediction": len(changed_wrong),
            "net_correct_count_change": len(corrected) - len(regressed),
            "accuracy_change": float(
                (len(corrected) - len(regressed)) / max(sample_count, 1)
            ),
        },
        "target_outcome_changes": [
            {
                "label_id": index,
                "label": class_names[index],
                "corrected": int(corrected_by_target[index]),
                "regressed": int(regressed_by_target[index]),
                "net": int(
                    corrected_by_target[index] - regressed_by_target[index]
                ),
            }
            for index in range(len(class_names))
        ],
        "top_prediction_transitions": _counter_rows(
            transition_counter,
            class_names,
            top_k=top_k,
        ),
        "qualitative_candidate_ids": {
            "candidate_corrections": example_rows(
                corrected,
                descending_key="candidate_confidence",
            ),
            "candidate_regressions": example_rows(
                regressed,
                descending_key="candidate_confidence",
            ),
            "selection_rule": (
                "highest candidate confidence, then sample_id; identifiers "
                "only, no source images copied"
            ),
        },
    }


def build_error_analysis(
    *,
    config_path: Path,
    manifest_path: Path,
    reference_path: Path,
    candidate_path: Path,
    reference_name: str,
    candidate_name: str,
    top_k: int,
) -> dict[str, Any]:
    paths = {
        "config": _resolve(config_path),
        "manifest": _resolve(manifest_path),
        "reference_predictions": _resolve(reference_path),
        "candidate_predictions": _resolve(candidate_path),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing error-analysis input: {name}: {path}")
    config = json.loads(paths["config"].read_text(encoding="utf-8-sig"))
    class_names = [
        str(row["name"]) if isinstance(row, Mapping) else str(row)
        for row in config.get("classes", [])
    ]
    analysis = analyze_prediction_records(
        _read_jsonl(paths["reference_predictions"]),
        _read_jsonl(paths["candidate_predictions"]),
        _read_jsonl(paths["manifest"]),
        class_names,
        reference_name=reference_name,
        candidate_name=candidate_name,
        top_k=top_k,
    )
    gate_intervention = (
        reference_name == "gate_disabled"
        and candidate_name == "gate_enabled"
    )
    if gate_intervention:
        analysis_scope = "posthoc_same_checkpoint_gate_error_analysis"
        claim_boundary = (
            "The models share one checkpoint, inputs and evaluation code; "
            "the only forward intervention is sensor reliability gating, so "
            "the aggregate correction pattern can contextualize that "
            "checkpoint's gate contribution. Sample example selection remains "
            "post-hoc, and this is not an independently retrained structural "
            "ablation, qualitative image validation, physical-sensor evidence "
            "or external generalization evidence."
        )
    else:
        analysis_scope = "posthoc_fixed_test_prediction_error_analysis"
        claim_boundary = (
            "This descriptive analysis reuses the fixed internal test set "
            "after model selection. It is not a preregistered endpoint, a "
            "causal component attribution, qualitative image validation or "
            "external generalization evidence."
        )
    return {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "analysis_scope": analysis_scope,
        "source_bindings": {
            name: {
                "path": _display_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "model_aliases": {
            "reference": reference_name,
            "candidate": candidate_name,
        },
        "analysis": analysis,
        "claim_boundary": claim_boundary,
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    analysis = report["analysis"]
    aliases = report["model_aliases"]
    reference_name = aliases["reference"]
    candidate_name = aliases["candidate"]
    reference = analysis["models"][reference_name]
    candidate = analysis["models"][candidate_name]
    paired = analysis["paired_outcomes"]
    lines = [
        "# Post-hoc Error Analysis",
        "",
        f"- Samples: {analysis['dataset']['sample_count']}.",
        f"- Source-image clusters: {analysis['dataset']['source_image_count']}.",
        f"- Classes: {analysis['dataset']['class_count']}.",
        (
            f"- `{reference_name}` errors: {reference['error_count']} "
            f"({100.0 * reference['error_rate']:.2f}%)."
        ),
        (
            f"- `{candidate_name}` errors: {candidate['error_count']} "
            f"({100.0 * candidate['error_rate']:.2f}%)."
        ),
        (
            f"- Candidate-only corrections / regressions: "
            f"{paired['candidate_only_correct']} / "
            f"{paired['reference_only_correct']}."
        ),
        "",
        "## Confidence",
        "",
        "| Model | Correct mean | Error mean | Errors >= .8 | Errors >= .9 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, model in (
        (reference_name, reference),
        (candidate_name, candidate),
    ):
        lines.append(
            f"| `{name}` | {model['confidence']['correct']['mean']:.4f} | "
            f"{model['confidence']['error']['mean']:.4f} | "
            f"{model['confidence']['high_confidence_error_count_ge_0_8']} | "
            f"{model['confidence']['high_confidence_error_count_ge_0_9']} |"
        )
    for name, model in (
        (reference_name, reference),
        (candidate_name, candidate),
    ):
        lines.extend(
            [
                "",
                f"## Top {name} Confusions",
                "",
                "| Target | Predicted | Count |",
                "|---|---|---:|",
            ]
        )
        for row in model["top_directional_confusions"]:
            lines.append(
                f"| `{row['target']}` | `{row['predicted']}` | "
                f"{row['count']} |"
            )
    lines.extend(
        [
            "",
            "## Largest Target-Level Changes",
            "",
            "| Class | Corrected | Regressed | Net |",
            "|---|---:|---:|---:|",
        ]
    )
    changes = sorted(
        analysis["target_outcome_changes"],
        key=lambda row: (-abs(row["net"]), row["label_id"]),
    )
    for row in changes[: analysis["dataset"]["top_k"]]:
        lines.append(
            f"| `{row['label']}` | {row['corrected']} | "
            f"{row['regressed']} | {row['net']:+d} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            str(report["claim_boundary"]),
            "",
            (
                "The JSON report records deterministic sample identifiers for "
                "later qualitative review, but this command does not read or "
                "copy source images."
            ),
            "",
        ]
    )
    _atomic_write(path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    report = build_error_analysis(
        config_path=args.config,
        manifest_path=args.manifest,
        reference_path=args.reference,
        candidate_path=args.candidate,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
        top_k=args.top_k,
    )
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
                "sample_count": report["analysis"]["dataset"]["sample_count"],
                "source_image_count": report["analysis"]["dataset"][
                    "source_image_count"
                ],
                "paired_outcomes": report["analysis"]["paired_outcomes"],
                "output": output.as_posix(),
                "markdown": markdown.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
