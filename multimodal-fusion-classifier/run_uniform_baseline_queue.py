"""Run the matched three-seed baseline train/test queue with restart support."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from prepare_recent_robust_baselines import (
    build_configs as build_recent_configs,
)
from prepare_standard_fusion_baselines import (
    METHODS as STANDARD_METHODS,
    SEEDS,
    build_configs as build_standard_configs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
METHODS = (*STANDARD_METHODS, "qmf", "tmc")
CONFIG_ROOTS = {
    **{
        method: PROJECT_ROOT / "configs" / "standard-fusion-baselines"
        for method in STANDARD_METHODS
    },
    "qmf": PROJECT_ROOT / "configs" / "recent-robust-baselines",
    "tmc": PROJECT_ROOT / "configs" / "recent-robust-baselines",
}
REPORT_ROOTS = {
    **{
        method: PROJECT_ROOT / "reports" / "qwen-material"
        / "standard-fusion-baselines" / method
        for method in STANDARD_METHODS
    },
    "qmf": PROJECT_ROOT / "reports" / "qwen-material"
    / "recent-robust-baselines" / "qmf",
    "tmc": PROJECT_ROOT / "reports" / "qwen-material"
    / "recent-robust-baselines" / "tmc",
}
DEFAULT_STATE = (
    PROJECT_ROOT / "reports" / "qwen-material"
    / "uniform-three-seed-baseline-queue" / "QUEUE_STATE.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run missing matched baseline training and test evaluation"
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _training_complete(config: Mapping[str, Any]) -> bool:
    checkpoint_dir = PROJECT_ROOT / config["training"]["checkpoint_dir"]
    summary_path = checkpoint_dir / "training_summary.json"
    if not summary_path.exists() or not (checkpoint_dir / "best.pt").exists():
        return False
    summary = _read_json(summary_path)
    return bool(summary.get("valid")) and int(summary.get("completed_epochs", 0)) == int(
        config["training"]["epochs"]
    )


def _evaluation_complete(output_dir: Path) -> bool:
    evaluation = output_dir / "evaluation.json"
    predictions = output_dir / "predictions.jsonl"
    if not evaluation.exists() or not predictions.exists():
        return False
    metrics = _read_json(evaluation).get("metrics", {})
    return all(key in metrics for key in ("accuracy", "macro_f1", "ece"))


def _run(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    build_standard_configs(num_workers=args.num_workers)
    build_recent_configs()

    state_path = args.state.resolve()
    log_root = state_path.parent / "logs"
    state: dict[str, Any] = {
        "valid": False,
        "started_at": _now(),
        "updated_at": _now(),
        "methods": list(args.methods),
        "seeds": list(args.seeds),
        "tasks": [],
    }
    _atomic_write(state_path, state)

    try:
        for method in args.methods:
            for seed in args.seeds:
                task_name = f"{method}_seed{seed}"
                config_path = CONFIG_ROOTS[method] / f"{task_name}.json"
                config = _read_json(config_path)
                checkpoint_dir = PROJECT_ROOT / config["training"]["checkpoint_dir"]
                output_dir = REPORT_ROOTS[method] / f"seed{seed}" / "test"
                task = {
                    "task": task_name,
                    "config": str(config_path.relative_to(PROJECT_ROOT)),
                    "status": "running",
                    "started_at": _now(),
                }
                state["tasks"].append(task)
                state["updated_at"] = _now()
                _atomic_write(state_path, state)
                print(f"START {task_name}", flush=True)

                if _training_complete(config):
                    task["training"] = "reused_complete"
                else:
                    train_command = [
                        sys.executable,
                        "train_recent_baseline.py",
                        "--config",
                        str(config_path),
                    ]
                    if (checkpoint_dir / "last.pt").exists():
                        train_command.append("--resume")
                    _run(
                        train_command,
                        log_root / f"{task_name}.train.stdout.log",
                        log_root / f"{task_name}.train.stderr.log",
                    )
                    if not _training_complete(config):
                        raise RuntimeError(f"incomplete training bundle: {task_name}")
                    task["training"] = "completed"

                if _evaluation_complete(output_dir):
                    task["test"] = "reused_complete"
                else:
                    _run(
                        [
                            sys.executable,
                            "evaluate_recent_baseline.py",
                            "--config",
                            str(config_path),
                            "--checkpoint",
                            str(checkpoint_dir / "best.pt"),
                            "--split",
                            "test",
                            "--output",
                            str(output_dir),
                            "--num-workers",
                            "0",
                        ],
                        log_root / f"{task_name}.test.stdout.log",
                        log_root / f"{task_name}.test.stderr.log",
                    )
                    if not _evaluation_complete(output_dir):
                        raise RuntimeError(f"incomplete test artifacts: {task_name}")
                    task["test"] = "completed"

                task["status"] = "completed"
                task["finished_at"] = _now()
                state["updated_at"] = _now()
                _atomic_write(state_path, state)
                print(f"DONE {task_name}", flush=True)
    except Exception as exc:
        state["error"] = str(exc)
        state["updated_at"] = _now()
        _atomic_write(state_path, state)
        raise

    state["valid"] = True
    state["finished_at"] = _now()
    state["updated_at"] = _now()
    _atomic_write(state_path, state)
    print(f"QUEUE COMPLETE: {len(state['tasks'])} tasks", flush=True)


if __name__ == "__main__":
    main()
