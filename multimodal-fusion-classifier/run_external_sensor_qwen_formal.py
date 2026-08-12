"""Execute the preregistered full-image Qwen v2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg-external"
)
DEFAULT_PROTOCOL = REPORT_ROOT / "QWEN_FORMAL_PROTOCOL.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered full-image Qwen v2 experiment"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stop-after-data", action="store_true")
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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_bindings(bindings: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not bindings:
        return ["protocol source bindings are missing"]
    for name, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping):
            errors.append(f"malformed source binding: {name}")
            continue
        raw_path = raw_binding.get("path", name)
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            path.resolve().relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"source binding leaves project: {name}")
            continue
        if not path.is_file():
            errors.append(f"source binding is missing: {name}")
        elif (
            path.stat().st_size != raw_binding.get("size_bytes")
            or _sha256(path) != raw_binding.get("sha256")
        ):
            errors.append(f"source binding drift: {name}")
    return errors


def validate_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors = validate_bindings(protocol.get("source_bindings", {}))
    generation = protocol.get("data_generation", {})
    training = protocol.get("training", {})
    evaluation = protocol.get("evaluation", {})
    fusion = protocol.get("fusion", {})
    if (
        protocol.get("protocol")
        != "full_image_qwen_external_sensor_joint_seg_v2"
        or protocol.get("status")
        != "preregistered_awaiting_api_credentials_and_data"
    ):
        errors.append("unexpected protocol identity or status")
    if (
        generation.get("mode") != "qwen"
        or generation.get("expected_source_image_count") != 5990
        or generation.get("api_key_value_persisted") is not False
        or generation.get("minimum_qwen_coverage") != 0.99
        or generation.get("required_valid_record_fraction") != 1.0
    ):
        errors.append("data-generation protocol is inconsistent")
    if (
        training.get("seeds") != [42, 43, 44]
        or training.get("batch_size") != 32
        or training.get("workers") != 4
        or training.get("epochs") != 20
        or training.get("freeze_layers") != 24
    ):
        errors.append("training protocol is inconsistent")
    if fusion != {
        "semantic_dim": 64,
        "sensor_hidden_dim": 256,
        "text_temperature": 0.1,
        "sensor_reconstruction_weight": 0.25,
        "sensor_calibration_weight": 0.05,
    }:
        errors.append("fusion protocol is inconsistent")
    required_conditions = set(evaluation.get("required_conditions", []))
    if required_conditions != {
        "external Qwen full-image sidecar",
        "same-checkpoint RGB predicted-sensor fallback",
        "same-checkpoint shuffled-sidecar negative control",
    }:
        errors.append("evaluation conditions are incomplete")
    return errors


def qwen_data_status(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    generation = protocol["data_generation"]
    output = _resolve(Path(str(generation["output"])))
    observations = output / "observations.jsonl"
    build_path = output / "BUILD_REPORT.json"
    validation_path = output / "VALIDATION_REPORT.json"
    result = {
        "ready": False,
        "output": output.relative_to(PROJECT_ROOT).as_posix(),
        "observations": observations.relative_to(PROJECT_ROOT).as_posix(),
        "errors": [],
    }
    if not all(
        path.is_file() for path in (observations, build_path, validation_path)
    ):
        result["errors"].append("Qwen sidecar outputs are incomplete")
        return result
    build = _read_json(build_path)
    validation = _read_json(validation_path)
    expected = int(generation["expected_source_image_count"])
    qwen_count = int(validation.get("qwen_success_count", -1))
    coverage = qwen_count / expected if qwen_count >= 0 else -1.0
    contract = validation.get("source_contract", {})
    if (
        build.get("valid") is not True
        or build.get("mode") != "qwen"
        or build.get("source_image_count") != expected
        or validation.get("valid") is not True
        or validation.get("error_count") != 0
        or validation.get("source_image_count") != expected
        or coverage < float(generation["minimum_qwen_coverage"])
        or contract.get("full_image_only") is not True
        or contract.get("annotation_files_read") is not False
        or contract.get("label_files_read") is not False
        or contract.get("bbox_inputs") is not False
        or contract.get("polygon_inputs") is not False
        or contract.get("class_inputs") is not False
        or validation.get("observations_sha256") != _sha256(observations)
        or build.get("output", {}).get("sha256") != _sha256(observations)
    ):
        result["errors"].append("Qwen sidecar evidence fails the formal gate")
        return result
    result.update(
        {
            "ready": True,
            "qwen_success_count": qwen_count,
            "fallback_count": int(validation.get("fallback_count", -1)),
            "coverage": coverage,
            "observations_sha256": _sha256(observations),
            "bindings": {
                "observations": _binding(observations),
                "build_report": _binding(build_path),
                "validation_report": _binding(validation_path),
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
    output = str(generation["output"])
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
    runs = []
    for seed in training["seeds"]:
        name = f"yolo11s-seg-joint-external-qwen-seed{seed}"
        report_run = (
            Path("reports")
            / "qwen-material"
            / "joint-multimodal-seg-external"
            / name
        )
        train = [
            python_executable,
            "train_external_sensor_joint_seg.py",
            "--sensor-observations",
            f"{output}/observations.jsonl",
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
        ]
        condition = [
            python_executable,
            "evaluate_external_sensor_conditions.py",
            "--checkpoint",
            (
                "checkpoints/joint-multimodal-seg-external/"
                f"{name}/weights/best.pt"
            ),
            "--sensor-observations",
            f"{output}/observations.jsonl",
            "--output",
            (report_run / "conditions").as_posix(),
            "--batch-size",
            "64",
            "--workers",
            str(training["workers"]),
            "--device",
            device,
        ]
        runs.append(
            {
                "seed": seed,
                "name": name,
                "train": train,
                "condition": condition,
            }
        )
    aggregate = [
        python_executable,
        "aggregate_external_sensor_formal.py",
        "--protocol",
        (
            protocol_path.resolve().relative_to(PROJECT_ROOT).as_posix()
            if protocol_path.resolve().is_relative_to(PROJECT_ROOT)
            else str(protocol_path.resolve())
        ),
    ]
    return {
        "build": build,
        "validate": validate,
        "runs": runs,
        "aggregate": aggregate,
    }


def _run_command(command: Sequence[str]) -> None:
    subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        check=True,
    )


def training_runtime_status(python_executable: str) -> dict[str, Any]:
    probe = (
        "import json, torch, ultralytics; "
        "print(json.dumps({'torch': torch.__version__, "
        "'ultralytics': ultralytics.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'gpu': torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else None}))"
    )
    try:
        completed = subprocess.run(
            [python_executable, "-c", probe],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ready": False,
            "python": python_executable,
            "error": type(exc).__name__,
        }
    if completed.returncode != 0:
        return {
            "ready": False,
            "python": python_executable,
            "error": "required training imports or CUDA are unavailable",
        }
    try:
        details = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {
            "ready": False,
            "python": python_executable,
            "error": "training runtime probe output is invalid",
        }
    ready = bool(details.get("cuda_available") and details.get("gpu"))
    return {
        "ready": ready,
        "python": python_executable,
        **details,
    }


def _completed_run_matches(
    *,
    run: Mapping[str, Any],
    data_status: Mapping[str, Any],
) -> bool:
    name = str(run["name"])
    report_run = REPORT_ROOT / name
    checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / "joint-multimodal-seg-external"
        / name
        / "weights"
        / "best.pt"
    )
    protocol_path = report_run / "PROTOCOL.json"
    evaluation_path = report_run / "evaluation.json"
    if not all(
        path.is_file() for path in (checkpoint, protocol_path, evaluation_path)
    ):
        return False
    run_protocol = _read_json(protocol_path)
    evaluation = _read_json(evaluation_path)
    return (
        run_protocol.get("seed") == run["seed"]
        and run_protocol.get("sensor_source", {}).get("mode") == "qwen"
        and evaluation.get("valid") is True
        and evaluation.get("sensor_observations", {}).get("mode") == "qwen"
        and evaluation.get("sensor_observations", {}).get("sha256")
        == data_status.get("observations_sha256")
        and evaluation.get("checkpoint", {}).get("sha256")
        == _sha256(checkpoint)
    )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = _resolve(args.protocol)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"formal protocol is missing: {protocol_path}")
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
    runtime_status = training_runtime_status(str(args.python))
    environment_name = str(
        protocol["data_generation"]["api_key_environment_variable_name"]
    )
    credential_available = bool(os.environ.get(environment_name))
    preflight = {
        "schema_version": 1,
        "state": (
            "ready"
            if runtime_status["ready"]
            and (credential_available or data_status["ready"])
            else (
                "awaiting_api_credentials"
                if runtime_status["ready"]
                else "invalid_training_runtime"
            )
        ),
        "valid": runtime_status["ready"],
        "protocol": _binding(protocol_path),
        "api_key_environment_variable_name": environment_name,
        "api_key_value_persisted": False,
        "api_credentials_available": credential_available,
        "qwen_data": data_status,
        "training_runtime": runtime_status,
        "planned_seeds": list(protocol["training"]["seeds"]),
        "training_workers": int(protocol["training"]["workers"]),
        "training_batch_size": int(protocol["training"]["batch_size"]),
        "command_plan": command_plan,
    }
    if args.preflight_only or (
        not credential_available and not data_status["ready"]
    ):
        return preflight
    if not runtime_status["ready"]:
        raise RuntimeError("formal training runtime is not ready")

    if not data_status["ready"]:
        _run_command(command_plan["build"])
        _run_command(command_plan["validate"])
        data_status = qwen_data_status(protocol)
        if not data_status["ready"]:
            raise ValueError("; ".join(data_status["errors"]))
    if args.stop_after_data:
        return {
            **preflight,
            "state": "data_ready",
            "qwen_data": data_status,
        }

    run_states = []
    for run in command_plan["runs"]:
        complete = _completed_run_matches(run=run, data_status=data_status)
        if not complete:
            report_run = REPORT_ROOT / str(run["name"])
            checkpoint_run = (
                PROJECT_ROOT
                / "checkpoints"
                / "joint-multimodal-seg-external"
                / str(run["name"])
            )
            if report_run.exists() or checkpoint_run.exists():
                raise FileExistsError(
                    "partial or incompatible formal run requires manual audit: "
                    f"{run['name']}"
                )
            _run_command(run["train"])
            complete = _completed_run_matches(
                run=run,
                data_status=data_status,
            )
            if not complete:
                raise ValueError(f"formal training audit failed: {run['name']}")
        condition_path = (
            REPORT_ROOT
            / str(run["name"])
            / "conditions"
            / "CONDITION_DIAGNOSTICS.json"
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
    result = {
        **preflight,
        "state": "complete",
        "qwen_data": data_status,
        "runs": run_states,
        "aggregate": _binding(REPORT_ROOT / "FORMAL_MULTISEED_SUMMARY.json"),
    }
    _atomic_json(REPORT_ROOT / "QWEN_FORMAL_EXECUTION.json", result)
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
                "planned_seeds": result["planned_seeds"],
                "training_workers": result["training_workers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
