"""Index the audited v3 reliability-ablation evidence extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
ABLATION_ROOT = (
    REPORTS_ROOT
    / "confirmatory-ablations"
    / "v3-no-reliability-workers0"
)
DEFAULT_OUTPUT = REPORTS_ROOT / "RELIABILITY_ABLATION_EXTENSION_INDEX.json"
MANUSCRIPT_RESULTS = REPORTS_ROOT / "MANUSCRIPT_RESULTS_SECTIONS.md"
REQUIRED = {
    "recovery_protocol_addendum": (
        ABLATION_ROOT / "RECOVERY_PROTOCOL_ADDENDUM.json"
    ),
    "outcome": ABLATION_ROOT / "OUTCOME.json",
    "paper_addendum": ABLATION_ROOT / "PAPER_ADDENDUM.json",
    "readiness_addendum": ABLATION_ROOT / "PAPER_READINESS_ADDENDUM.json",
    "outcome_verification": ABLATION_ROOT / "OUTCOME_VERIFICATION.json",
    "manuscript_results_audit": (
        REPORTS_ROOT / "MANUSCRIPT_RESULTS_SECTIONS.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v3 reliability-ablation extension index"
    )
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


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_index(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    documents = {name: _read_json(path) for name, path in REQUIRED.items()}
    if (
        documents["recovery_protocol_addendum"].get("state")
        != "frozen_before_workers0_recovery"
        or documents["recovery_protocol_addendum"].get("valid") is not True
        or documents["outcome"].get("state") != "complete"
        or documents["outcome"].get("valid") is not True
        or documents["paper_addendum"].get("valid") is not True
        or documents["readiness_addendum"].get("valid") is not True
        or documents["outcome_verification"].get("valid") is not True
        or documents["manuscript_results_audit"].get("valid") is not True
    ):
        raise ValueError("ablation extension inputs are incomplete")
    matched = documents["outcome"]["matched_test_comparison"]["aggregate"]
    index = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "v3_reliability_subsystem_confirmatory_extension",
        "training": {
            "matched_seeds": [42, 43, 44],
            "epochs_per_seed": 100,
            "resume_used": False,
            "runtime_num_workers": 0,
            "workers4_partial_run_reused": False,
        },
        "clean_test_effects_full_minus_ablation": {
            metric: matched[metric]
            for metric in (
                "accuracy",
                "macro_f1",
                "negative_log_likelihood",
                "multiclass_brier_score",
                "ece",
            )
        },
        "robustness": {
            "training_seed": 42,
            "degradation_seeds": [2026, 2027, 2028, 2029, 2030],
            "scenarios": [
                row["scenario"]
                for row in documents["outcome"]["robustness_comparison"]["rows"]
            ],
        },
        "claim_boundaries": documents["outcome"]["boundaries"],
        "source_bindings": {
            **{name: _binding(path) for name, path in REQUIRED.items()},
            "manuscript_results": _binding(MANUSCRIPT_RESULTS),
            "builder_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
    )
    accuracy = matched["accuracy"]
    macro_f1 = matched["macro_f1"]
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# v3 reliability-subsystem confirmatory extension",
                "",
                (
                    "Three matched independent training seeds completed 100 "
                    "epochs without resume."
                ),
                "",
                "| Clean test effect (full minus ablation) | Mean | Sample SD |",
                "|---|---:|---:|",
                (
                    f"| Accuracy | {accuracy['mean']:+.6f} | "
                    f"{accuracy['sample_std']:.6f} |"
                ),
                (
                    f"| Macro F1 | {macro_f1['mean']:+.6f} | "
                    f"{macro_f1['sample_std']:.6f} |"
                ),
                "",
                "The internal test is inspected. The intervention removes the "
                "entire reliability subsystem, and seed-42 degradation repeats "
                "are supporting descriptive evidence.",
                "",
            ]
        ),
    )
    return index


def main() -> None:
    result = build_index(parse_args().output)
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "output": DEFAULT_OUTPUT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
