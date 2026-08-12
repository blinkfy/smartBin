"""Build a paper-readiness addendum for the v3 reliability ablation."""

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
DEFAULT_OUTCOME = ABLATION_ROOT / "OUTCOME.json"
DEFAULT_PAPER_ADDENDUM = ABLATION_ROOT / "PAPER_ADDENDUM.json"
DEFAULT_OUTPUT = ABLATION_ROOT / "PAPER_READINESS_ADDENDUM.json"
BASE_READINESS = REPORTS_ROOT / "PAPER_READINESS_AUDIT.json"
BASE_EVIDENCE = REPORTS_ROOT / "FINAL_EVIDENCE_AUDIT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v3 reliability-ablation readiness addendum"
    )
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    parser.add_argument(
        "--paper-addendum",
        type=Path,
        default=DEFAULT_PAPER_ADDENDUM,
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


def build_readiness_addendum(
    outcome_path: Path = DEFAULT_OUTCOME,
    paper_addendum_path: Path = DEFAULT_PAPER_ADDENDUM,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    outcome_path = _resolve(outcome_path)
    paper_addendum_path = _resolve(paper_addendum_path)
    output_path = _resolve(output_path)
    outcome = _read_json(outcome_path)
    paper_addendum = _read_json(paper_addendum_path)
    base_readiness = _read_json(BASE_READINESS)
    base_evidence = _read_json(BASE_EVIDENCE)
    if (
        outcome.get("state") != "complete"
        or outcome.get("valid") is not True
        or outcome.get("training", {}).get("seeds") != [42, 43, 44]
        or paper_addendum.get("state") != "complete"
        or paper_addendum.get("valid") is not True
        or base_readiness.get("valid") is not True
        or base_readiness.get("internal_paper_evidence_ready") is not True
        or base_evidence.get("valid") is not True
    ):
        raise ValueError("readiness addendum inputs are incomplete")
    matched = outcome["matched_test_comparison"]["aggregate"]
    addendum = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "internal_paper_evidence_ready": True,
        "full_patent_reproduction_ready": False,
        "external_deployment_claim_ready": False,
        "new_confirmatory_evidence": {
            "component": "entire_sensor_reliability_subsystem",
            "independent_training_seed_count": 3,
            "matched_training_seeds": [42, 43, 44],
            "epochs_per_run": 100,
            "clean_test_full_minus_ablation": {
                "accuracy_mean": matched["accuracy"]["mean"],
                "accuracy_sample_std": matched["accuracy"]["sample_std"],
                "macro_f1_mean": matched["macro_f1"]["mean"],
                "macro_f1_sample_std": matched["macro_f1"]["sample_std"],
                "ece_mean": matched["ece"]["mean"],
                "ece_sample_std": matched["ece"]["sample_std"],
            },
            "robustness_training_seed_count": 1,
            "robustness_degradation_seed_count": 5,
        },
        "legacy_warning_update": {
            "status": "refined_not_removed",
            "previous": (
                "Formal ablations currently quantify one training seed; paired "
                "test intervals do not include training-seed variance."
            ),
            "current": (
                "The legacy eight-comparison ablation suite remains a one-seed "
                "study. The new v3 reliability-subsystem comparison independently "
                "retrains three matched seeds and reports training-seed variation."
            ),
        },
        "unchanged_boundaries": list(base_readiness.get("warnings", [])),
        "patent_gaps_unchanged": [
            "RGB-D input is absent.",
            "Synchronized hardware sensors are absent.",
            "MMFC features are not injected into the YOLO detection head.",
            "Small-target and occlusion strata are diagnostic proxies.",
        ],
        "source_bindings": {
            "outcome": _binding(outcome_path),
            "paper_addendum": _binding(paper_addendum_path),
            "base_paper_readiness": _binding(BASE_READINESS),
            "base_final_evidence": _binding(BASE_EVIDENCE),
            "builder_source": _binding(Path(__file__)),
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n",
    )
    effects = addendum["new_confirmatory_evidence"][
        "clean_test_full_minus_ablation"
    ]
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# v3 reliability-ablation paper-readiness addendum",
                "",
                "- Internal paper evidence remains ready.",
                "- Full patent reproduction and external deployment claims remain "
                "not ready.",
                "- The v3 reliability-subsystem comparison now uses three matched "
                "independent training seeds.",
                "- The legacy eight-comparison ablation suite remains a one-seed "
                "study and must be described separately.",
                "",
                "| Clean test effect (full minus ablation) | Mean | Sample SD |",
                "|---|---:|---:|",
                (
                    f"| Accuracy | {effects['accuracy_mean']:+.6f} | "
                    f"{effects['accuracy_sample_std']:.6f} |"
                ),
                (
                    f"| Macro F1 | {effects['macro_f1_mean']:+.6f} | "
                    f"{effects['macro_f1_sample_std']:.6f} |"
                ),
                (
                    f"| ECE | {effects['ece_mean']:+.6f} | "
                    f"{effects['ece_sample_std']:.6f} |"
                ),
                "",
                "The fixed internal test is inspected, and Qwen sidecars remain "
                "image-conditioned soft-sensor observations.",
                "",
            ]
        ),
    )
    return addendum


def main() -> None:
    args = parse_args()
    result = build_readiness_addendum(
        args.outcome,
        args.paper_addendum,
        args.output,
    )
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "output": _resolve(args.output).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
