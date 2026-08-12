"""Build scoped semantic-fallback evidence from an existing matched study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RECONSTRUCTION_METRICS = ("latent_mse", "grid_mae", "grid_mse")
CLASSIFICATION_METRICS = ("accuracy", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--classification-margin-pp", type=float, default=0.25)
    return parser.parse_args()


def _mean(summary: dict[str, Any], mode: str, metric: str) -> float:
    return float(summary["modes"][mode][metric]["mean"])


def _values(summary: dict[str, Any], mode: str, metric: str) -> list[float]:
    return [float(value) for value in summary["modes"][mode][metric]["values"]]


def _reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def build_evidence(
    summary: dict[str, Any],
    *,
    classification_margin_pp: float,
) -> dict[str, Any]:
    correct = "descriptions"
    references = ("vision_only", "shuffled_descriptions")
    relative_reductions = {
        reference: {
            metric: _reduction(
                _mean(summary, reference, metric),
                _mean(summary, correct, metric),
            )
            for metric in RECONSTRUCTION_METRICS
        }
        for reference in references
    }
    per_seed_dominance = {
        reference: {
            metric: all(
                candidate < baseline
                for candidate, baseline in zip(
                    _values(summary, correct, metric),
                    _values(summary, reference, metric),
                )
            )
            for metric in RECONSTRUCTION_METRICS
        }
        for reference in references
    }
    classification_deltas_pp = {
        reference: {
            metric: 100.0
            * (
                _mean(summary, correct, metric)
                - _mean(summary, reference, metric)
            )
            for metric in CLASSIFICATION_METRICS
        }
        for reference in references
    }
    fallback_fidelity_supported = all(
        relative_reductions[reference][metric] > 0.0
        and per_seed_dominance[reference][metric]
        for reference in references
        for metric in RECONSTRUCTION_METRICS
    )
    classification_preserved = all(
        abs(classification_deltas_pp[reference][metric])
        <= classification_margin_pp
        for reference in references
        for metric in CLASSIFICATION_METRICS
    )
    return {
        "scope": "development validation; missing sensor probability = 100%",
        "run_count_per_condition": int(
            summary["protocol"]["run_count_per_condition"]
        ),
        "strict_all_metrics_claim_supported": bool(
            summary["semantic_guidance_claim_supported"]
        ),
        "fallback_fidelity_claim_supported": fallback_fidelity_supported,
        "classification_preserved_within_margin": classification_preserved,
        "classification_margin_pp": classification_margin_pp,
        "relative_reconstruction_reduction_pct": relative_reductions,
        "per_seed_reconstruction_dominance": per_seed_dominance,
        "classification_delta_pp": classification_deltas_pp,
        "supported_wording": (
            "Correct class descriptions improve missing-sensor latent and "
            "spatial-grid reconstruction relative to vision-only and shuffled "
            "semantic controls, while validation classification remains within "
            f"a {classification_margin_pp:.2f}-percentage-point descriptive margin."
        ),
        "unsupported_wording": (
            "Semantic descriptions improve missing-sensor classification "
            "accuracy or Macro F1."
        ),
        "confirmatory_boundary": (
            "The current split has been used for method development. Final "
            "confirmatory wording requires a newly locked external or untouched "
            "evaluation set."
        ),
    }


def write_paper_table(
    path: Path,
    summary: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    labels = {
        "vision_only": "Vision-conditioned fallback",
        "descriptions": "Vision + correct descriptions",
        "class_names": "Vision + class names only",
        "shuffled_descriptions": "Vision + shuffled descriptions",
    }
    lines = [
        "# Semantic Guidance Validation Evidence",
        "",
        "| Fallback condition | Missing Acc. (%) | Missing Macro F1 (%) | "
        "Latent MSE | Grid MAE | Grid MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in (
        "vision_only",
        "descriptions",
        "class_names",
        "shuffled_descriptions",
    ):
        values = summary["modes"][mode]
        lines.append(
            "| "
            + labels[mode]
            + " | "
            + f"{100 * values['accuracy']['mean']:.2f} +/- "
            + f"{100 * values['accuracy']['std']:.2f} | "
            + f"{100 * values['macro_f1']['mean']:.2f} +/- "
            + f"{100 * values['macro_f1']['std']:.2f} | "
            + f"{values['latent_mse']['mean']:.3e} +/- "
            + f"{values['latent_mse']['std']:.1e} | "
            + f"{values['grid_mae']['mean']:.4f} +/- "
            + f"{values['grid_mae']['std']:.4f} | "
            + f"{values['grid_mse']['mean']:.4f} +/- "
            + f"{values['grid_mse']['std']:.4f} |"
        )
    reductions = evidence["relative_reconstruction_reduction_pct"]
    lines.extend(
        [
            "",
            "Correct descriptions versus vision-only reduce latent MSE, grid "
            f"MAE, and grid MSE by {reductions['vision_only']['latent_mse']:.1f}%, "
            f"{reductions['vision_only']['grid_mae']:.1f}%, and "
            f"{reductions['vision_only']['grid_mse']:.1f}%, respectively.",
            "",
            "Correct descriptions versus shuffled descriptions reduce latent "
            f"MSE, grid MAE, and grid MSE by "
            f"{reductions['shuffled_descriptions']['latent_mse']:.1f}%, "
            f"{reductions['shuffled_descriptions']['grid_mae']:.1f}%, and "
            f"{reductions['shuffled_descriptions']['grid_mse']:.1f}%, "
            "respectively.",
            "",
            "This is development-validation evidence for fallback fidelity, "
            "not an independent test-set claim of classification improvement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.classification_margin_pp < 0.0:
        raise ValueError("classification margin must be non-negative")
    summary_path = args.summary.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = build_evidence(
        summary,
        classification_margin_pp=args.classification_margin_pp,
    )
    output_dir = summary_path.parent
    (output_dir / "EVIDENCE.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_paper_table(output_dir / "PAPER_TABLE.md", summary, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
