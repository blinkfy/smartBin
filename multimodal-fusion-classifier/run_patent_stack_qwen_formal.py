"""Execute the additive preregistered v6 Qwen adaptation protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from prepare_patent_stack_qwen_formal_protocol import (
    DEFAULT_OUTPUT as DEFAULT_PROTOCOL,
    PROTOCOL_NAME,
    RUN_PROTOCOL_NAME,
)
from run_external_sensor_qwen_formal import (
    PROJECT_ROOT,
    _atomic_json,
    _binding,
    _read_json,
    _resolve,
    _run_command,
    _sha256,
    qwen_data_status,
    training_runtime_status,
    validate_bindings,
)


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-patent-qwen"
)
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints" / "joint-multimodal-seg-patent-qwen"
)
DEFAULT_PREFLIGHT = REPORT_ROOT / "QWEN_FORMAL_EXTENSION_PREFLIGHT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered v6 Qwen adaptation extension"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stop-after-data", action="store_true")
    return parser.parse_args()


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors = validate_bindings(protocol.get("source_bindings", {}))
    generation = protocol.get("data_generation", {})
    training = protocol.get("training", {})
    fusion = protocol.get("fusion", {})
    evaluation = protocol.get("evaluation", {})
    lineage = protocol.get("lineage", {})
    derangement = protocol.get("derangement_control", {})
    if (
        protocol.get("protocol") != PROTOCOL_NAME
        or protocol.get("run_protocol") != RUN_PROTOCOL_NAME
        or protocol.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
    ):
        errors.append("unexpected extension protocol identity or status")
    if (
        generation.get("mode") != "qwen"
        or generation.get("expected_source_image_count") != 5990
        or generation.get("expected_split_counts")
        != {"train": 4823, "val": 597, "test": 570}
        or generation.get("api_key_value_persisted") is not False
        or generation.get("minimum_qwen_coverage") != 0.99
        or generation.get("required_valid_record_fraction") != 1.0
    ):
        errors.append("data-generation protocol is inconsistent")
    if (
        lineage.get("experiment_type")
        != "fixed_parent_multiseed_qwen_adaptation"
        or lineage.get("shared_parent_across_seeds") is not True
        or lineage.get("independent_end_to_end_replicates") is not False
        or not isinstance(
            lineage.get("parent_checkpoint", {}).get("sha256"), str
        )
    ):
        errors.append("fixed-parent lineage is not explicit")
    if (
        training.get("seeds") != [42, 43, 44]
        or training.get("batch_size") != 32
        or training.get("workers") != 4
        or training.get("epochs") != 20
        or training.get("patience") != 5
        or training.get("freeze_layers") != 24
        or training.get("output_project")
        != "checkpoints/joint-multimodal-seg-patent-qwen"
    ):
        errors.append("training protocol is inconsistent")
    required_fusion = {
        "semantic_dim": 64,
        "sensor_hidden_dim": 256,
        "text_temperature": 0.1,
        "sensor_reconstruction_weight": 0.25,
        "sensor_calibration_weight": 0.05,
        "reliability_calibration_weight": 0.5,
        "reliability_hidden_dim": 32,
        "reliability_init": 0.85,
        "mismatch_probability": 0.35,
    }
    if any(fusion.get(key) != value for key, value in required_fusion.items()):
        errors.append("fusion protocol is inconsistent")
    if (
        derangement.get("method")
        != "within_split_one_position_cyclic_derangement"
        or derangement.get("self_assignments_allowed") is not False
        or derangement.get("split_membership_preserved") is not True
        or derangement.get("sensor_payload_multiset_preserved") is not True
    ):
        errors.append("derangement protocol is inconsistent")
    primary_conditions = set(evaluation.get("primary_conditions", []))
    if primary_conditions != {
        "clean matched Qwen full-image sidecar",
        "same-checkpoint RGB predicted-sensor fallback",
        "same-checkpoint within-split shuffled Qwen sidecar",
    }:
        errors.append("primary evaluation conditions are incomplete")
    return errors


def derangement_status(
    protocol: Mapping[str, Any],
    data_status: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ready": False,
        "errors": [],
    }
    shuffled = _resolve(
        Path(str(protocol["derangement_control"]["output"]))
    )
    report_path = _resolve(
        Path(str(protocol["derangement_control"]["report"]))
    )
    result.update(
        {
            "output": shuffled.relative_to(PROJECT_ROOT).as_posix(),
            "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
        }
    )
    if not shuffled.is_file() or not report_path.is_file():
        result["errors"].append("Qwen derangement outputs are incomplete")
        return result
    report = _read_json(report_path)
    if (
        not data_status.get("ready")
        or report.get("valid") is not True
        or report.get("source_image_count") != 5990
        or report.get("split_counts")
        != {"train": 4823, "val": 597, "test": 570}
        or report.get("self_assignment_count") != 0
        or report.get("split_violation_count") != 0
        or report.get("sensor_payload_multiset_preserved") is not True
        or report.get("bindings", {})
        .get("original_observations", {})
        .get("sha256")
        != data_status.get("observations_sha256")
        or report.get("bindings", {})
        .get("shuffled_observations", {})
        .get("sha256")
        != _sha256(shuffled)
    ):
        result["errors"].append("Qwen derangement evidence fails the gate")
        return result
    result.update(
        {
            "ready": True,
            "shuffled_observations_sha256": _sha256(shuffled),
            "bindings": {
                "shuffled_observations": _binding(shuffled),
                "shuffle_report": _binding(report_path),
            },
        }
    )
    return result


def build_command_plan(
    protocol: Mapping[str, Any],
    *,
    python_executable: str,
    device: str,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    generation = protocol["data_generation"]
    training = protocol["training"]
    fusion = protocol["fusion"]
    degradation = protocol["software_sensor_degradation"]
    output = str(generation["output"])
    base_weights = str(
        protocol["lineage"]["parent_checkpoint"]["path"]
    )
    project = str(training["output_project"])
    report_root = str(training["report_root"])
    build = [
        python_executable,
        "build_scene_sensor_sidecars.py",
        "--mode",
        "qwen",
        "--api-key-env",
        str(generation["api_key_environment_variable_name"]),
        "--output",
        output,
        "--resume",
        "--workers",
        str(generation["workers"]),
        "--retries",
        str(generation["retries"]),
        "--timeout",
        str(generation["timeout_seconds"]),
        "--checkpoint-every",
        "20",
        "--progress-every",
        "20",
    ]
    validate = [
        python_executable,
        "validate_scene_sensor_sidecars.py",
        "--output",
        output,
    ]
    derange = [
        python_executable,
        "prepare_qwen_sensor_derangement.py",
        "--input",
        f"{output}/observations.jsonl",
        "--output",
        str(protocol["derangement_control"]["output"]),
    ]
    runs = []
    for seed in training["seeds"]:
        name = f"yolo11s-seg-joint-patent-qwen-adapt-seed{seed}"
        checkpoint = (
            f"{project}/{name}/weights/best.pt"
        )
        report_run = f"{report_root}/{name}"
        train = [
            python_executable,
            "train_patent_aligned_qwen_formal.py",
            "--base-weights",
            base_weights,
            "--sensor-observations",
            f"{output}/observations.jsonl",
            "--project",
            project,
            "--report-dir",
            report_root,
            "--name",
            name,
            "--epochs",
            str(training["epochs"]),
            "--patience",
            str(training["patience"]),
            "--batch-size",
            str(training["batch_size"]),
            "--image-size",
            str(training["image_size"]),
            "--workers",
            str(training["workers"]),
            "--freeze-layers",
            str(training["freeze_layers"]),
            "--seed",
            str(seed),
            "--device",
            device,
            "--learning-rate",
            str(training["learning_rate"]),
            "--weight-decay",
            str(training["weight_decay"]),
            "--cache",
            str(training["cache"]),
            "--semantic-dim",
            str(fusion["semantic_dim"]),
            "--sensor-hidden-dim",
            str(fusion["sensor_hidden_dim"]),
            "--text-temperature",
            str(fusion["text_temperature"]),
            "--sensor-reconstruction-weight",
            str(fusion["sensor_reconstruction_weight"]),
            "--sensor-calibration-weight",
            str(fusion["sensor_calibration_weight"]),
            "--reliability-calibration-weight",
            str(fusion["reliability_calibration_weight"]),
            "--reliability-hidden-dim",
            str(fusion["reliability_hidden_dim"]),
            "--reliability-init",
            str(fusion["reliability_init"]),
            "--mismatch-probability",
            str(fusion["mismatch_probability"]),
            "--missing-probability",
            str(degradation["missing_probability"]),
            "--noise-probability",
            str(degradation["noise_probability"]),
            "--noise-std-low",
            str(degradation["noise_std_low"]),
            "--noise-std-high",
            str(degradation["noise_std_high"]),
            "--bias-probability",
            str(degradation["bias_probability"]),
            "--bias-strength",
            str(degradation["bias_strength"]),
            "--stuck-probability",
            str(degradation["stuck_probability"]),
            "--quality-scale",
            str(degradation["quality_scale"]),
        ]
        condition = [
            python_executable,
            "evaluate_patent_aligned_qwen_formal_conditions.py",
            "--checkpoint",
            checkpoint,
            "--sensor-observations",
            f"{output}/observations.jsonl",
            "--shuffled-observations",
            str(protocol["derangement_control"]["output"]),
            "--output",
            f"{report_run}/RELIABILITY_CONDITIONS.json",
            "--image-size",
            str(training["image_size"]),
            "--batch-size",
            str(protocol["evaluation"]["evaluation_batch_size"]),
            "--workers",
            str(training["workers"]),
            "--device",
            device,
            "--seed",
            str(seed),
        ]
        runs.append(
            {
                "seed": seed,
                "name": name,
                "train": train,
                "condition": condition,
            }
        )
    resolved_protocol = protocol_path.resolve()
    protocol_argument = (
        resolved_protocol.relative_to(PROJECT_ROOT).as_posix()
        if resolved_protocol.is_relative_to(PROJECT_ROOT)
        else str(resolved_protocol)
    )
    aggregate = [
        python_executable,
        "aggregate_patent_stack_qwen_formal.py",
        "--protocol",
        protocol_argument,
    ]
    return {
        "build": build,
        "validate": validate,
        "derange": derange,
        "runs": runs,
        "aggregate": aggregate,
    }


def _completed_run_matches(
    *,
    run: Mapping[str, Any],
    data_status: Mapping[str, Any],
) -> bool:
    name = str(run["name"])
    report_run = REPORT_ROOT / name
    checkpoint = CHECKPOINT_ROOT / name / "weights" / "best.pt"
    protocol_path = report_run / "PROTOCOL.json"
    evaluation_path = report_run / "evaluation.json"
    if not all(
        path.is_file() for path in (checkpoint, protocol_path, evaluation_path)
    ):
        return False
    run_protocol = _read_json(protocol_path)
    evaluation = _read_json(evaluation_path)
    return (
        run_protocol.get("protocol") == RUN_PROTOCOL_NAME
        and run_protocol.get("formal_paper_metric") is True
        and run_protocol.get("seed") == run["seed"]
        and run_protocol.get("sensor_source", {}).get("mode") == "qwen"
        and run_protocol.get("qwen_formal_gate", {}).get(
            "observations_sha256"
        )
        == data_status.get("observations_sha256")
        and evaluation.get("valid") is True
        and evaluation.get("formal_paper_metric") is True
        and evaluation.get("sensor_observations", {}).get("mode") == "qwen"
        and evaluation.get("sensor_observations", {}).get("sha256")
        == data_status.get("observations_sha256")
        and evaluation.get("checkpoint", {}).get("sha256")
        == _sha256(checkpoint)
    )


def _write_preflight(result: Mapping[str, Any]) -> None:
    _atomic_json(DEFAULT_PREFLIGHT, dict(result))
    lines = [
        "# v6 Qwen 正式适配预检",
        "",
        f"- 状态：{result['state']}",
        f"- 训练运行时可用：{result['training_runtime']['ready']}",
        f"- API 凭据可用：{result['api_credentials_available']}",
        f"- Qwen 数据可用：{result['qwen_data']['ready']}",
        f"- 乱序负对照可用：{result['derangement']['ready']}",
        "- 计划种子：42、43、44",
        f"- batch/workers：{result['training_batch_size']}/{result['training_workers']}",
        "",
        (
            "预检只记录凭据是否存在，不记录凭据值。三种子共享固定 v5 "
            "父检查点。"
        ),
        "",
    ]
    DEFAULT_PREFLIGHT.with_suffix(".md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = _resolve(args.protocol)
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"formal extension protocol is missing: {protocol_path}"
        )
    protocol = _read_json(protocol_path)
    errors = validate_protocol(protocol)
    if errors:
        raise ValueError("; ".join(errors))
    command_plan = build_command_plan(
        protocol,
        python_executable=str(args.python),
        device=str(args.device),
        protocol_path=protocol_path,
    )
    data_status = qwen_data_status(protocol)
    shuffle_status = derangement_status(protocol, data_status)
    runtime_status = training_runtime_status(str(args.python))
    environment_name = str(
        protocol["data_generation"]["api_key_environment_variable_name"]
    )
    credential_available = bool(os.environ.get(environment_name))
    ready = bool(
        runtime_status["ready"]
        and (
            (data_status["ready"] and shuffle_status["ready"])
            or credential_available
        )
    )
    state = (
        "ready"
        if ready
        else (
            "awaiting_api_credentials"
            if runtime_status["ready"]
            and not data_status["ready"]
            and not credential_available
            else (
                "awaiting_derangement"
                if runtime_status["ready"]
                and data_status["ready"]
                and not shuffle_status["ready"]
                else "invalid_training_runtime"
            )
        )
    )
    preflight = {
        "schema_version": 1,
        "state": state,
        "valid": runtime_status["ready"],
        "protocol": _binding(protocol_path),
        "lineage": protocol["lineage"],
        "api_key_environment_variable_name": environment_name,
        "api_key_value_persisted": False,
        "api_credentials_available": credential_available,
        "qwen_data": data_status,
        "derangement": shuffle_status,
        "training_runtime": runtime_status,
        "planned_seeds": list(protocol["training"]["seeds"]),
        "training_workers": int(protocol["training"]["workers"]),
        "training_batch_size": int(protocol["training"]["batch_size"]),
        "command_plan": command_plan,
    }
    _write_preflight(preflight)
    if args.preflight_only or (
        not credential_available and not data_status["ready"]
    ):
        return preflight
    if not runtime_status["ready"]:
        raise RuntimeError("formal adaptation runtime is not ready")

    if not data_status["ready"]:
        _run_command(command_plan["build"])
        _run_command(command_plan["validate"])
        data_status = qwen_data_status(protocol)
        if not data_status["ready"]:
            raise ValueError("; ".join(data_status["errors"]))
    shuffle_status = derangement_status(protocol, data_status)
    if not shuffle_status["ready"]:
        _run_command(command_plan["derange"])
        shuffle_status = derangement_status(protocol, data_status)
        if not shuffle_status["ready"]:
            raise ValueError("; ".join(shuffle_status["errors"]))
    if args.stop_after_data:
        result = {
            **preflight,
            "state": "data_ready",
            "qwen_data": data_status,
            "derangement": shuffle_status,
        }
        _write_preflight(result)
        return result

    run_states = []
    for run in command_plan["runs"]:
        complete = _completed_run_matches(
            run=run, data_status=data_status
        )
        if not complete:
            report_run = REPORT_ROOT / str(run["name"])
            checkpoint_run = CHECKPOINT_ROOT / str(run["name"])
            if report_run.exists() or checkpoint_run.exists():
                raise FileExistsError(
                    "partial or incompatible formal adaptation run "
                    f"requires manual audit: {run['name']}"
                )
            _run_command(run["train"])
            complete = _completed_run_matches(
                run=run, data_status=data_status
            )
            if not complete:
                raise ValueError(
                    f"formal adaptation audit failed: {run['name']}"
                )
        condition_path = (
            REPORT_ROOT
            / str(run["name"])
            / "RELIABILITY_CONDITIONS.json"
        )
        if not condition_path.is_file():
            _run_command(run["condition"])
        run_states.append(
            {
                "seed": run["seed"],
                "name": run["name"],
                "training_complete": True,
                "condition_complete": condition_path.is_file(),
            }
        )
    _run_command(command_plan["aggregate"])
    aggregate_path = REPORT_ROOT / "FORMAL_ADAPTATION_MULTISEED_SUMMARY.json"
    result = {
        **preflight,
        "state": "complete",
        "qwen_data": data_status,
        "derangement": shuffle_status,
        "runs": run_states,
        "aggregate": _binding(aggregate_path),
    }
    _atomic_json(REPORT_ROOT / "QWEN_FORMAL_EXTENSION_EXECUTION.json", result)
    return result


def main() -> None:
    result = execute(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "api_credentials_available": result[
                    "api_credentials_available"
                ],
                "qwen_data_ready": result["qwen_data"]["ready"],
                "derangement_ready": result["derangement"]["ready"],
                "planned_seeds": result["planned_seeds"],
                "training_workers": result["training_workers"],
                "training_batch_size": result["training_batch_size"],
                "shared_parent": result["lineage"][
                    "shared_parent_across_seeds"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
