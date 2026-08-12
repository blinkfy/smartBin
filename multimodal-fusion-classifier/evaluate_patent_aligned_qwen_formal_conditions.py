"""Evaluate one registered v6 Qwen adaptation seed under all conditions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluate_contrastive_external_conditions as shared
import evaluate_patent_aligned_conditions as v6_evaluator
from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from prepare_patent_stack_qwen_formal_protocol import (
    DEFAULT_OUTPUT as DEFAULT_FORMAL_PROTOCOL,
    RUN_PROTOCOL_NAME,
)


DEFAULT_NAME = "yolo11s-seg-joint-patent-qwen-adapt-seed42"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-patent-qwen"
    / DEFAULT_NAME
    / "weights"
    / "best.pt"
)
DEFAULT_OBSERVATIONS = (
    PROJECT_ROOT / "data" / "scene-qwen-sensor-sidecars" / "observations.jsonl"
)
DEFAULT_SHUFFLED = (
    PROJECT_ROOT
    / "data"
    / "scene-qwen-sensor-sidecars"
    / "SHUFFLED_OBSERVATIONS.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent-qwen"
    / DEFAULT_NAME
    / "RELIABILITY_CONDITIONS.json"
)


def parse_args():
    args = v6_evaluator.parse_args()
    if Path(args.checkpoint) == v6_evaluator.DEFAULT_CHECKPOINT:
        args.checkpoint = DEFAULT_CHECKPOINT
    if Path(args.sensor_observations) == shared.DEFAULT_OBSERVATIONS:
        args.sensor_observations = DEFAULT_OBSERVATIONS
    if Path(args.shuffled_observations) == shared.DEFAULT_SHUFFLED:
        args.shuffled_observations = DEFAULT_SHUFFLED
    if Path(args.output) == v6_evaluator.DEFAULT_OUTPUT:
        args.output = DEFAULT_OUTPUT
    return args


def validate_formal_run_bindings(
    protocol: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    checkpoint_sha256: str,
    observations_sha256: str,
) -> None:
    if (
        protocol.get("protocol") != RUN_PROTOCOL_NAME
        or protocol.get("formal_paper_metric") is not True
        or protocol.get("sensor_source", {}).get("mode") != "qwen"
        or protocol.get("qwen_formal_gate", {}).get(
            "observations_sha256"
        )
        != observations_sha256
        or evaluation.get("valid") is not True
        or evaluation.get("formal_paper_metric") is not True
        or evaluation.get("checkpoint", {}).get("sha256")
        != checkpoint_sha256
        or evaluation.get("sensor_observations", {}).get("mode")
        != "qwen"
        or evaluation.get("sensor_observations", {}).get("sha256")
        != observations_sha256
    ):
        raise ValueError("formal v6 Qwen run bindings are inconsistent")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def evaluate(args):
    formal_path = DEFAULT_FORMAL_PROTOCOL.resolve()
    shuffled = shared._resolve(args.shuffled_observations)
    shuffle_report = shuffled.parent / "SHUFFLE_REPORT.json"
    for path in (formal_path, shuffle_report):
        if not path.is_file():
            raise FileNotFoundError(f"formal condition input is missing: {path}")
    formal = _read_json(formal_path)
    derangement = _read_json(shuffle_report)
    observations = shared._resolve(args.sensor_observations)
    if (
        formal.get("run_protocol") != RUN_PROTOCOL_NAME
        or derangement.get("valid") is not True
        or derangement.get("self_assignment_count") != 0
        or derangement.get("split_membership_preserved") is not True
        or derangement.get("sensor_payload_multiset_preserved") is not True
        or derangement.get("bindings", {})
        .get("original_observations", {})
        .get("sha256")
        != sha256_file(observations)
        or derangement.get("bindings", {})
        .get("shuffled_observations", {})
        .get("sha256")
        != sha256_file(shuffled)
    ):
        raise ValueError("formal Qwen derangement contract is inconsistent")

    original_validator = shared.validate_run_bindings
    original_protocol = v6_evaluator.PROTOCOL_NAME
    shared.validate_run_bindings = validate_formal_run_bindings
    v6_evaluator.PROTOCOL_NAME = RUN_PROTOCOL_NAME
    try:
        result = v6_evaluator.evaluate(args)
    finally:
        shared.validate_run_bindings = original_validator
        v6_evaluator.PROTOCOL_NAME = original_protocol

    output = shared._resolve(args.output)
    result.update(
        {
            "formal_paper_metric": True,
            "evidence_role": (
                "formal_qwen_patent_stack_adaptation_conditions"
            ),
            "lineage": formal["lineage"],
            "claim_boundaries": formal["claim_boundaries"],
        }
    )
    result["bindings"].update(
        {
            "formal_extension_protocol": shared._binding(formal_path),
            "qwen_derangement_report": shared._binding(shuffle_report),
            "formal_evaluator_source": shared._binding(Path(__file__)),
        }
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
