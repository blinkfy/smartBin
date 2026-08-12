"""Run matched mixed-fault interventions across training and perturbation seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from run_fine_grained_intervention_queue import (
    INTERVENTIONS,
    OUTPUT_ROOT,
    RUNS,
    SEEDS,
)


PERTURBATION_SEEDS = (2026, 2027, 2028, 2029, 2030)
FAULT_VARIANTS = (
    "full",
    "trimodal_no_reliability",
    "no_raw_grid_quality",
    "no_cross_modal_quality",
    "no_latent_discrepancy",
    "no_latent_consistency",
    "no_variance_evidence",
    "no_availability_evidence",
    "no_grid_quality_heads",
    "no_latent_quality_evidence",
    "constant_reliability_gate",
)
DEFAULT_STATE = OUTPUT_ROOT / "faults" / "QUEUE_STATE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run restartable mixed-fault path interventions"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=FAULT_VARIANTS,
        default=list(FAULT_VARIANTS),
    )
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=list(SEEDS),
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _complete(output_dir: Path) -> bool:
    summary = output_dir / "robustness_multiseed_summary.json"
    if not summary.is_file():
        return False
    report = _read_json(summary)
    if report.get("seed_count") != len(PERTURBATION_SEEDS):
        return False
    for seed in PERTURBATION_SEEDS:
        evaluation = (
            output_dir / "seeds" / f"seed-{seed}" / "mixed"
            / "evaluation.json"
        )
        if not evaluation.is_file():
            return False
        if _read_json(evaluation).get("metrics", {}).get("sample_count") != 826:
            return False
    return True


def _run(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=OUTPUT_ROOT.parents[2],
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
    state_path = args.state.resolve()
    log_root = state_path.parent / "logs"
    state: dict[str, Any] = {
        "schema_version": 1,
        "valid": False,
        "protocol": "mixed_fault_inference_path_intervention",
        "started_at": _now(),
        "updated_at": _now(),
        "training_seeds": list(args.training_seeds),
        "perturbation_seeds": list(PERTURBATION_SEEDS),
        "variants": list(args.variants),
        "tasks": [],
    }
    _atomic_write(state_path, state)

    try:
        for variant in args.variants:
            intervention_args = (
                [] if variant == "full" else INTERVENTIONS[variant]["args"]
            )
            for training_seed in args.training_seeds:
                task_name = f"{variant}_training_seed{training_seed}"
                output_dir = (
                    OUTPUT_ROOT / "faults" / variant
                    / f"training-seed{training_seed}"
                )
                task: dict[str, Any] = {
                    "task": task_name,
                    "variant": variant,
                    "training_seed": training_seed,
                    "status": "running",
                    "started_at": _now(),
                }
                state["tasks"].append(task)
                state["updated_at"] = _now()
                _atomic_write(state_path, state)
                print(f"START {task_name}", flush=True)

                if _complete(output_dir):
                    task["evaluation"] = "reused_complete"
                else:
                    run = RUNS[training_seed]
                    _run(
                        [
                            sys.executable,
                            "evaluate_robustness.py",
                            "--config",
                            str(run["config"]),
                            "--checkpoint",
                            str(run["checkpoint"]),
                            "--split",
                            "test",
                            "--output",
                            str(output_dir),
                            "--scenarios",
                            "mixed",
                            "--seeds",
                            *[str(seed) for seed in PERTURBATION_SEEDS],
                            "--num-workers",
                            "0",
                            *intervention_args,
                        ],
                        log_root / f"{task_name}.stdout.log",
                        log_root / f"{task_name}.stderr.log",
                    )
                    if not _complete(output_dir):
                        raise RuntimeError(f"incomplete fault task: {task_name}")
                    task["evaluation"] = "completed"

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
