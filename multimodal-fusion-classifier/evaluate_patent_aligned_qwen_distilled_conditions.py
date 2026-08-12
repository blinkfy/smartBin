"""Evaluate v6 object-Qwen-distilled adaptation under sensor conditions."""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_contrastive_external_conditions as shared
from mmfc.model import PROJECT_ROOT
from train_patent_aligned_qwen_distilled import (
    DEFAULT_NAME,
    DEFAULT_PROJECT,
    DEFAULT_REPORT,
    DEFAULT_SENSOR_OBSERVATIONS,
    RUN_PROTOCOL_NAME,
)


DEFAULT_CHECKPOINT = DEFAULT_PROJECT / DEFAULT_NAME / "weights" / "best.pt"
DEFAULT_SHUFFLED = (
    DEFAULT_SENSOR_OBSERVATIONS.parent / "SHUFFLED_OBSERVATIONS.jsonl"
)
DEFAULT_OUTPUT = (
    DEFAULT_REPORT / DEFAULT_NAME / "RELIABILITY_CONDITIONS.json"
)


def parse_args():
    args = shared.parse_args()
    if Path(args.checkpoint) == shared.DEFAULT_CHECKPOINT:
        args.checkpoint = DEFAULT_CHECKPOINT
    if Path(args.sensor_observations) == shared.DEFAULT_OBSERVATIONS:
        args.sensor_observations = DEFAULT_SENSOR_OBSERVATIONS
    if Path(args.shuffled_observations) == shared.DEFAULT_SHUFFLED:
        args.shuffled_observations = DEFAULT_SHUFFLED
    if Path(args.output) == shared.DEFAULT_OUTPUT:
        args.output = DEFAULT_OUTPUT
    return args


def evaluate(args):
    original_protocol = shared.PROTOCOL_NAME
    shared.PROTOCOL_NAME = RUN_PROTOCOL_NAME
    try:
        result = shared.evaluate(args)
    finally:
        shared.PROTOCOL_NAME = original_protocol

    output = shared._resolve(args.output)
    result["evidence_role"] = (
        "object_qwen_distilled_patent_stack_development"
    )
    result["bindings"]["distilled_evaluator_source"] = shared._binding(
        Path(__file__)
    )
    shared._atomic_json(output, result)
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "formal_paper_metric": result["formal_paper_metric"],
                "gate_ordering_all_passed": result[
                    "gate_ordering_all_passed"
                ],
                "clean_minus_shuffled": result["comparisons"][
                    "clean_minus_shuffled"
                ],
                "clean_minus_rgb_fallback": result["comparisons"][
                    "clean_minus_rgb_fallback"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
