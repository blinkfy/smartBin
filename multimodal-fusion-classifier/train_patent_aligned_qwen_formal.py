"""Run one registered Qwen adaptation seed for the v6 patent stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmfc.model import PROJECT_ROOT
from prepare_joint_multimodal_seg import sha256_file
from prepare_patent_stack_qwen_formal_protocol import (
    DEFAULT_OUTPUT as DEFAULT_FORMAL_PROTOCOL,
    RUN_PROTOCOL_NAME,
)
import train_contrastive_reliable_external_sensor_joint_seg as v4
import train_patent_aligned_external_sensor_joint_seg as v6


DEFAULT_OBSERVATIONS = (
    PROJECT_ROOT / "data" / "scene-qwen-sensor-sidecars" / "observations.jsonl"
)
DEFAULT_PROJECT = (
    PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg-patent-qwen"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent-qwen"
)
DEFAULT_NAME = "yolo11s-seg-joint-patent-qwen-adapt-seed42"
DEVELOPMENT_PROTOCOL = v6._protocol


def parse_args():
    args = v6.parse_args()
    if Path(args.sensor_observations) == v4.DEFAULT_OBSERVATIONS:
        args.sensor_observations = DEFAULT_OBSERVATIONS
    if Path(args.project) == v6.DEFAULT_PROJECT:
        args.project = DEFAULT_PROJECT
    if Path(args.report_dir) == v6.DEFAULT_REPORT:
        args.report_dir = DEFAULT_REPORT
    if args.name == v6.DEFAULT_NAME:
        args.name = DEFAULT_NAME
    return args


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_qwen_gate(paths: dict[str, Path]) -> dict[str, Any]:
    build = _read_json(paths["sensor_build"])
    validation = _read_json(paths["sensor_validation"])
    expected = 5990
    qwen_count = int(validation.get("qwen_success_count", -1))
    coverage = qwen_count / expected if qwen_count >= 0 else -1.0
    contract = validation.get("source_contract", {})
    observations_sha256 = sha256_file(paths["sensor_observations"])
    if (
        build.get("valid") is not True
        or build.get("mode") != "qwen"
        or build.get("source_image_count") != expected
        or validation.get("valid") is not True
        or validation.get("mode") != "qwen"
        or validation.get("source_image_count") != expected
        or validation.get("error_count") != 0
        or coverage < 0.99
        or contract.get("full_image_only") is not True
        or contract.get("annotation_files_read") is not False
        or contract.get("label_files_read") is not False
        or contract.get("bbox_inputs") is not False
        or contract.get("polygon_inputs") is not False
        or contract.get("class_inputs") is not False
        or validation.get("observations_sha256") != observations_sha256
        or build.get("output", {}).get("sha256") != observations_sha256
    ):
        raise ValueError("Qwen observations fail the formal adaptation gate")
    return {
        "source_image_count": expected,
        "qwen_success_count": qwen_count,
        "fallback_count": int(validation.get("fallback_count", -1)),
        "coverage": coverage,
        "observations_sha256": observations_sha256,
    }


def _protocol(
    args,
    paths: dict[str, Path],
    train_args: dict[str, Any],
    degradation: dict[str, Any],
) -> dict[str, Any]:
    formal_path = DEFAULT_FORMAL_PROTOCOL.resolve()
    if not formal_path.is_file():
        raise FileNotFoundError(
            f"formal extension protocol is missing: {formal_path}"
        )
    formal = _read_json(formal_path)
    if (
        formal.get("run_protocol") != RUN_PROTOCOL_NAME
        or formal.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
        or int(args.seed) not in formal.get("training", {}).get("seeds", [])
    ):
        raise ValueError("formal extension protocol is inconsistent")
    qwen_gate = validate_qwen_gate(paths)
    protocol = DEVELOPMENT_PROTOCOL(args, paths, train_args, degradation)
    protocol.update(
        {
            "protocol": RUN_PROTOCOL_NAME,
            "status": "formal_qwen_adaptation_frozen_before_training",
            "formal_paper_metric": True,
            "evidence_role": "formal_qwen_patent_stack_adaptation_seed",
            "formal_extension_protocol": {
                "path": formal_path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": formal_path.stat().st_size,
                "sha256": sha256_file(formal_path),
            },
            "qwen_formal_gate": qwen_gate,
            "lineage": formal["lineage"],
            "claim_boundaries": formal["claim_boundaries"],
        }
    )
    protocol["source_bindings"][
        Path(__file__).resolve().as_posix()
    ] = {
        "size_bytes": Path(__file__).stat().st_size,
        "sha256": sha256_file(Path(__file__)),
    }
    protocol["source_bindings"][formal_path.as_posix()] = {
        "size_bytes": formal_path.stat().st_size,
        "sha256": sha256_file(formal_path),
    }
    return protocol


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args) -> dict[str, Any]:
    original_protocol = v6._protocol
    v6._protocol = _protocol
    try:
        result = v6.run(args)
    finally:
        v6._protocol = original_protocol
    if result.get("state") != "complete":
        return result

    evaluation_path = Path(result["evaluation"]).resolve()
    evaluation = _read_json(evaluation_path)
    protocol = _read_json(evaluation_path.parent / "PROTOCOL.json")
    if (
        protocol.get("protocol") != RUN_PROTOCOL_NAME
        or protocol.get("formal_paper_metric") is not True
        or evaluation.get("sensor_observations", {}).get("mode") != "qwen"
        or evaluation.get("sensor_observations", {}).get("sha256")
        != protocol.get("qwen_formal_gate", {}).get(
            "observations_sha256"
        )
    ):
        raise ValueError("completed Qwen adaptation binding is inconsistent")
    evaluation.update(
        {
            "protocol": RUN_PROTOCOL_NAME,
            "formal_paper_metric": True,
            "evidence_role": "formal_qwen_patent_stack_adaptation_seed",
            "formal_extension_protocol": protocol[
                "formal_extension_protocol"
            ],
            "lineage": protocol["lineage"],
            "limitations": protocol["claim_boundaries"],
        }
    )
    _atomic_json(evaluation_path, evaluation)
    result.update(
        {
            "formal_paper_metric": True,
            "evidence_role": evaluation["evidence_role"],
            "evaluation": evaluation_path.as_posix(),
        }
    )
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
