"""Evaluate the v6 patent-aligned stack under sensor conditions."""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_contrastive_external_conditions as shared
from mmfc.model import PROJECT_ROOT
from train_patent_aligned_external_sensor_joint_seg import (
    DEFAULT_NAME,
    PROTOCOL_NAME,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
    / "RELIABILITY_CONDITIONS.json"
)


def parse_args():
    args = shared.parse_args()
    if Path(args.checkpoint) == shared.DEFAULT_CHECKPOINT:
        args.checkpoint = DEFAULT_CHECKPOINT
    if Path(args.output) == shared.DEFAULT_OUTPUT:
        args.output = DEFAULT_OUTPUT
    return args


def evaluate(args):
    original_protocol = shared.PROTOCOL_NAME
    shared.PROTOCOL_NAME = PROTOCOL_NAME
    try:
        result = shared.evaluate(args)
    finally:
        shared.PROTOCOL_NAME = original_protocol

    output = shared._resolve(args.output)
    result["evidence_role"] = (
        "rule_sidecar_patent_stack_mechanism_development"
    )
    result["bindings"]["patent_evaluator_source"] = shared._binding(
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
