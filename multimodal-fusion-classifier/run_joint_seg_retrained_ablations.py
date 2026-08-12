"""Run the frozen independently retrained joint-head ablation queue."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from mmfc.model import PROJECT_ROOT
from mmfc.training import resolve_project_path
from prepare_joint_multimodal_seg import sha256_file


DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "joint-multimodal-seg"
    / "retrained-ablations"
    / "PROTOCOL.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen independently retrained joint-head ablations."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_bindings(protocol: dict[str, Any]) -> None:
    errors = []
    for relative, expected in protocol["source_bindings"].items():
        path = resolve_project_path(Path(relative))
        if not path.is_file():
            errors.append(f"missing source binding: {relative}")
        elif sha256_file(path) != expected:
            errors.append(f"source binding drift: {relative}")
    if errors:
        raise ValueError("; ".join(errors))


def build_command(run: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "train_joint_multimodal_seg.py",
        "--name",
        str(run["run_name"]),
        "--fusion-mode",
        str(run["fusion_mode"]),
        "--seed",
        str(run["seed"]),
        "--epochs",
        str(policy["epochs"]),
        "--patience",
        str(policy["patience"]),
        "--batch-size",
        str(policy["batch_size"]),
        "--image-size",
        str(policy["image_size"]),
        "--workers",
        str(policy["workers"]),
        "--freeze-layers",
        str(policy["freeze_layers"]),
        "--device",
        "0",
        "--learning-rate",
        str(policy["learning_rate"]),
        "--weight-decay",
        str(policy["weight_decay"]),
        "--semantic-dim",
        str(policy["semantic_dim"]),
        "--sensor-hidden-dim",
        str(policy["sensor_hidden_dim"]),
        "--text-temperature",
        str(policy["text_temperature"]),
        "--sensor-reconstruction-weight",
        str(policy["sensor_reconstruction_weight"]),
        "--sensor-calibration-weight",
        str(policy["sensor_calibration_weight"]),
        "--cache",
        "false" if policy["cache"] is False else str(policy["cache"]),
    ]
    if not policy["amp"]:
        command.append("--no-amp")
    return command


def run_queue(protocol_path: Path) -> dict[str, Any]:
    protocol_path = resolve_project_path(protocol_path)
    protocol = _read_object(protocol_path)
    if protocol.get("status") != "frozen_before_formal_ablation_training":
        raise ValueError("ablation protocol is not frozen")
    _verify_bindings(protocol)

    root = protocol_path.parent
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = root / "STATUS.json"
    completed = []
    skipped = []
    for run in protocol["formal_runs"]:
        run_name = str(run["run_name"])
        report_run = (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "joint-multimodal-seg"
            / run_name
        )
        checkpoint = (
            PROJECT_ROOT
            / "checkpoints"
            / "joint-multimodal-seg"
            / run_name
            / "weights"
            / "best.pt"
        )
        evaluation = report_run / "evaluation.json"
        if checkpoint.is_file() and evaluation.is_file():
            saved = _read_object(evaluation)
            if saved.get("fusion_mode") != run["fusion_mode"]:
                raise ValueError(f"completed run mode mismatch: {run_name}")
            skipped.append(run_name)
            continue
        if report_run.exists() or checkpoint.parent.parent.exists():
            raise FileExistsError(
                f"partial formal run requires manual audit: {run_name}"
            )

        state = {
            "schema_version": 1,
            "protocol": protocol_path.resolve().as_posix(),
            "state": "running",
            "current_run": run_name,
            "completed": completed,
            "skipped_existing_complete": skipped,
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        _atomic_json(status_path, state)
        command = build_command(run, protocol["training_policy"])
        stdout_path = logs / f"{run_name}.stdout.log"
        stderr_path = logs / f"{run_name}.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if result.returncode != 0:
            state.update(
                {
                    "state": "failed",
                    "returncode": result.returncode,
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
            )
            _atomic_json(status_path, state)
            raise RuntimeError(
                f"formal ablation failed: {run_name}; see {stderr_path}"
            )
        if not checkpoint.is_file() or not evaluation.is_file():
            raise FileNotFoundError(
                f"formal ablation outputs are incomplete: {run_name}"
            )
        completed.append(run_name)

    state = {
        "schema_version": 1,
        "protocol": protocol_path.resolve().as_posix(),
        "state": "complete",
        "current_run": None,
        "completed": completed,
        "skipped_existing_complete": skipped,
        "run_count": len(protocol["formal_runs"]),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    _atomic_json(status_path, state)
    return state


def main() -> None:
    print(json.dumps(run_queue(parse_args().protocol), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
