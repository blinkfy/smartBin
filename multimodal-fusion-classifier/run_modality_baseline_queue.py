"""Run the restartable uniform three-seed modality-baseline queue."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

from mmfc.training import PROJECT_ROOT
from prepare_modality_baselines import (
    SEEDS,
    VARIANTS,
    config_path,
    write_configs,
)
from summarize_modality_baselines import OUTPUT_ROOT, summarize


STATE_PATH = OUTPUT_ROOT / "QUEUE_STATE.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
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


def training_complete(checkpoint_dir: Path) -> bool:
    summary_path = checkpoint_dir / "training_summary.json"
    if not all(
        path.is_file()
        for path in (summary_path, checkpoint_dir / "best.pt", checkpoint_dir / "last.pt")
    ):
        return False
    summary = _read(summary_path)
    return (
        int(summary.get("completed_epochs", 0)) == 100
        and bool(summary.get("checkpoints_saved", False))
    )


def evaluation_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    report = _read(path)
    return (
        report.get("dataset", {}).get("split") == "test"
        and report.get("metrics", {}).get("sample_count") == 826
        and all(
            name in report.get("metrics", {})
            for name in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "negative_log_likelihood",
                "multiclass_brier_score",
                "ece",
            )
        )
    )


def build_train_command(config: Path, checkpoint_dir: Path) -> list[str]:
    command = [sys.executable, "-u", "train.py", "--config", str(config)]
    if (checkpoint_dir / "last.pt").is_file():
        command.append("--resume")
    return command


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


def _run_task(variant: str, seed: int) -> dict[str, Any]:
    config = config_path(variant, seed)
    resolved = _read(config)
    checkpoint_dir = PROJECT_ROOT / resolved["training"]["checkpoint_dir"]
    output_dir = OUTPUT_ROOT / variant / f"seed{seed}" / "test"
    log_root = OUTPUT_ROOT / "logs"
    task_name = f"{variant}_seed{seed}"
    result: dict[str, Any] = {"task": task_name, "reused_training": False}
    if training_complete(checkpoint_dir):
        result["reused_training"] = True
    else:
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            if not (checkpoint_dir / "last.pt").is_file():
                raise RuntimeError(
                    f"non-empty incomplete checkpoint has no last.pt: {checkpoint_dir}"
                )
        _run(
            build_train_command(config, checkpoint_dir),
            log_root / f"{task_name}.train.stdout.log",
            log_root / f"{task_name}.train.stderr.log",
        )
        if not training_complete(checkpoint_dir):
            raise RuntimeError(f"training did not complete: {task_name}")
    evaluation_path = output_dir / "evaluation.json"
    if evaluation_complete(evaluation_path):
        result["reused_evaluation"] = True
    else:
        _run(
            [
                sys.executable,
                "evaluate.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint_dir / "best.pt"),
                "--split",
                "test",
                "--bootstrap-repeats",
                "0",
                "--num-workers",
                "0",
                "--output",
                str(output_dir),
            ],
            log_root / f"{task_name}.test.stdout.log",
            log_root / f"{task_name}.test.stderr.log",
        )
        if not evaluation_complete(evaluation_path):
            raise RuntimeError(f"evaluation did not complete: {task_name}")
        result["reused_evaluation"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independently trained modality baselines"
    )
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    parser.add_argument("--max-parallel", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(args.variants)
    seeds = tuple(args.seeds)
    write_configs(variants, seeds)
    tasks = [(variant, seed) for variant in variants for seed in seeds]
    state: dict[str, Any] = {
        "schema_version": 1,
        "valid": False,
        "protocol": "uniform_three_seed_independently_trained_modality_matrix_v1",
        "started_at": _now(),
        "updated_at": _now(),
        "max_parallel": args.max_parallel,
        "tasks": {
            f"{variant}_seed{seed}": {"status": "pending"}
            for variant, seed in tasks
        },
    }
    lock = threading.Lock()
    _atomic_json(STATE_PATH, state)

    def execute(variant: str, seed: int) -> dict[str, Any]:
        name = f"{variant}_seed{seed}"
        with lock:
            state["tasks"][name] = {"status": "running", "started_at": _now()}
            state["updated_at"] = _now()
            _atomic_json(STATE_PATH, state)
        print(f"START {name}", flush=True)
        result = _run_task(variant, seed)
        print(f"DONE {name}", flush=True)
        return result

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(execute, variant, seed): (variant, seed)
            for variant, seed in tasks
        }
        for future in as_completed(futures):
            variant, seed = futures[future]
            name = f"{variant}_seed{seed}"
            try:
                result = future.result()
                task_state = {"status": "completed", "finished_at": _now(), **result}
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                task_state = {
                    "status": "failed",
                    "finished_at": _now(),
                    "error": str(exc),
                }
            with lock:
                state["tasks"][name] = task_state
                state["updated_at"] = _now()
                _atomic_json(STATE_PATH, state)
    if errors:
        state["errors"] = errors
        state["finished_at"] = _now()
        _atomic_json(STATE_PATH, state)
        raise RuntimeError("; ".join(errors))
    summary = summarize(OUTPUT_ROOT)
    state["valid"] = bool(summary["valid"])
    state["finished_at"] = _now()
    state["updated_at"] = _now()
    _atomic_json(STATE_PATH, state)
    print(f"QUEUE COMPLETE: {len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
