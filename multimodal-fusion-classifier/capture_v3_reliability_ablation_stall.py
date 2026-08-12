"""Capture immutable evidence for the workers=4 long-horizon training stall."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import psutil


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = PROJECT_ROOT / "reports" / "qwen-material" / "training"
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability"
)
DEFAULT_OUTPUT = REPORT_ROOT / "WORKERS4_STALL_LIVENESS.json"
PROTOCOL = REPORT_ROOT / "PROTOCOL.json"
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "confirmatory-ablations"
    / "v3_no_reliability_seed42.json"
)
STDOUT = TRAINING_ROOT / "v3_no_reliability_seed42.stdout.log"
STDERR = TRAINING_ROOT / "v3_no_reliability_seed42.stderr.log"
STATUS = TRAINING_ROOT / "v3_reliability_ablation_status.json"
CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "confirmatory-ablations"
    / "v3_no_reliability"
    / "seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture workers=4 training-stall liveness evidence"
    )
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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
        "last_write_time_ns": resolved.stat().st_mtime_ns,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _last_epoch(path: Path) -> int | None:
    last = None
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("epoch"), int):
                last = int(value["epoch"])
    return last


def _gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parts = [item.strip() for item in result.stdout.strip().split(",")]
    return {
        "utilization_pct": int(parts[0]),
        "memory_used_mib": int(parts[1]),
        "power_w": float(parts[2]),
    }


def _process_snapshot(process: psutil.Process) -> dict[str, Any]:
    cpu = process.cpu_times()
    children = []
    for child in process.children(recursive=True):
        try:
            children.append(
                {
                    "pid": child.pid,
                    "name": child.name(),
                    "command_line": child.cmdline(),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return {
        "pid": process.pid,
        "status": process.status(),
        "cpu_user_seconds": float(cpu.user),
        "cpu_system_seconds": float(cpu.system),
        "rss_bytes": int(process.memory_info().rss),
        "children": children,
        "command_line": process.cmdline(),
    }


def capture(pid: int, seconds: int, output_path: Path) -> dict[str, Any]:
    if seconds < 10:
        raise ValueError("liveness interval must be at least 10 seconds")
    process = psutil.Process(pid)
    command_line = " ".join(process.cmdline())
    if (
        "train.py" not in command_line
        or "v3_no_reliability_seed42.json" not in command_line
    ):
        raise ValueError("PID does not match the expected seed42 training process")
    before = {
        "process": _process_snapshot(process),
        "gpu": _gpu_snapshot(),
        "stdout": _binding(STDOUT),
        "stderr": _binding(STDERR),
        "last_epoch": _last_epoch(STDOUT),
        "captured_unix_ns": time.time_ns(),
    }
    time.sleep(seconds)
    after = {
        "process": _process_snapshot(process),
        "gpu": _gpu_snapshot(),
        "stdout": _binding(STDOUT),
        "stderr": _binding(STDERR),
        "last_epoch": _last_epoch(STDOUT),
        "captured_unix_ns": time.time_ns(),
    }
    cpu_before = (
        before["process"]["cpu_user_seconds"]
        + before["process"]["cpu_system_seconds"]
    )
    cpu_after = (
        after["process"]["cpu_user_seconds"]
        + after["process"]["cpu_system_seconds"]
    )
    no_log_progress = (
        before["stdout"]["size_bytes"] == after["stdout"]["size_bytes"]
        and before["stdout"]["sha256"] == after["stdout"]["sha256"]
        and before["last_epoch"] == after["last_epoch"]
    )
    before_python_children = [
        child
        for child in before["process"]["children"]
        if str(child["name"]).lower().startswith("python")
    ]
    after_python_children = [
        child
        for child in after["process"]["children"]
        if str(child["name"]).lower().startswith("python")
    ]
    no_worker_children = (
        before_python_children == [] and after_python_children == []
    )
    report = {
        "schema_version": 1,
        "state": "captured_before_termination",
        "valid": True,
        "scope": "v3_reliability_ablation_workers4_long_horizon_stall",
        "interval_seconds": seconds,
        "before": before,
        "after": after,
        "diagnosis": {
            "no_log_progress": no_log_progress,
            "last_completed_epoch": after["last_epoch"],
            "no_worker_children": no_worker_children,
            "before_python_worker_children": before_python_children,
            "after_python_worker_children": after_python_children,
            "process_cpu_seconds_delta": cpu_after - cpu_before,
            "stdout_age_seconds_after": (
                time.time_ns() - after["stdout"]["last_write_time_ns"]
            )
            / 1_000_000_000.0,
            "long_horizon_workers4_not_operationally_valid": (
                no_log_progress
                and no_worker_children
                and after["last_epoch"] == 69
            ),
        },
        "frozen_inputs": {
            "protocol": _binding(PROTOCOL),
            "seed42_config": _binding(CONFIG),
            "queue_status": _binding(STATUS),
        },
        "boundary": (
            "The two-epoch input-equivalence benchmark remains valid, but it "
            "did not establish 100-epoch operational liveness on Windows."
        ),
        "errors": [],
    }
    output_path = _resolve(output_path)
    _atomic_write(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# workers=4 long-horizon stall evidence",
                "",
                f"- Last completed epoch: {after['last_epoch']}",
                f"- Observation interval: {seconds} seconds",
                f"- Log progress: {not no_log_progress}",
                f"- Worker child processes present: {not no_worker_children}",
                (
                    "- Main-process CPU-time delta: "
                    f"{cpu_after - cpu_before:.6f} seconds"
                ),
                (
                    "- The short input-equivalence benchmark remains valid but "
                    "did not prove 100-epoch operational liveness."
                ),
                "",
            ]
        ),
    )
    return report


def main() -> None:
    args = parse_args()
    report = capture(args.pid, args.seconds, args.output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "diagnosis": report["diagnosis"],
                "output": _resolve(args.output).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if (
        report["diagnosis"]["long_horizon_workers4_not_operationally_valid"]
        is not True
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
