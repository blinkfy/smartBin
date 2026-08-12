"""Evaluate fine-grained path interventions on all three full checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "fine-grained-interventions"
)
DEFAULT_STATE = OUTPUT_ROOT / "QUEUE_STATE.json"
SEEDS = (42, 43, 44)
RUNS = {
    42: {
        "config": PROJECT_ROOT / "configs" / "ablations"
        / "full_cross_modal_quality.json",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "ablations"
        / "full_cross_modal_quality" / "best.pt",
    },
    43: {
        "config": PROJECT_ROOT / "configs" / "replicates"
        / "full_cross_modal_quality_seed43.json",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "ablations"
        / "v3_seed_replicates" / "full_cross_modal_quality_seed43" / "best.pt",
    },
    44: {
        "config": PROJECT_ROOT / "configs" / "replicates"
        / "full_cross_modal_quality_seed44.json",
        "checkpoint": PROJECT_ROOT / "checkpoints" / "ablations"
        / "v3_seed_replicates" / "full_cross_modal_quality_seed44" / "best.pt",
    },
}
INTERVENTIONS = {
    "vision_only": {
        "group": "modality_path",
        "label": "Visual path only",
        "args": ["--posthoc-modalities", "visual"],
    },
    "vision_text": {
        "group": "modality_path",
        "label": "Visual + text paths",
        "args": ["--posthoc-modalities", "visual_text"],
    },
    "vision_sensor": {
        "group": "modality_path",
        "label": "Visual + sensor paths",
        "args": ["--posthoc-modalities", "visual_sensor"],
    },
    "trimodal_no_reliability": {
        "group": "fusion_path",
        "label": "No reliability/fallback path",
        "args": ["--disable-sensor-reliability"],
    },
    "no_raw_grid_quality": {
        "group": "reliability_evidence",
        "label": "No raw-grid quality branch",
        "args": ["--disable-sensor-grid-quality"],
    },
    "no_cross_modal_quality": {
        "group": "reliability_evidence",
        "label": "No predicted-observed grid branch",
        "args": ["--disable-sensor-cross-modal-quality"],
    },
    "no_latent_discrepancy": {
        "group": "reliability_evidence",
        "label": "No latent discrepancy evidence",
        "args": ["--disable-reliability-evidence", "discrepancy"],
    },
    "no_latent_consistency": {
        "group": "reliability_evidence",
        "label": "No latent consistency evidence",
        "args": ["--disable-reliability-evidence", "consistency"],
    },
    "no_variance_evidence": {
        "group": "reliability_evidence",
        "label": "No variance evidence",
        "args": ["--disable-reliability-evidence", "uncertainty"],
    },
    "no_availability_evidence": {
        "group": "reliability_evidence",
        "label": "No availability evidence in learned head",
        "args": ["--disable-reliability-evidence", "availability"],
    },
    "no_grid_quality_heads": {
        "group": "reliability_combination",
        "label": "No raw or cross-modal grid-quality heads",
        "args": [
            "--disable-sensor-grid-quality",
            "--disable-sensor-cross-modal-quality",
        ],
    },
    "no_latent_quality_evidence": {
        "group": "reliability_combination",
        "label": "No latent or availability evidence in learned head",
        "args": [
            "--disable-reliability-evidence",
            "discrepancy",
            "consistency",
            "uncertainty",
            "availability",
        ],
    },
    "constant_reliability_gate": {
        "group": "reliability_combination",
        "label": "Fixed learned-bias reliability gate",
        "args": [
            "--disable-sensor-grid-quality",
            "--disable-sensor-cross-modal-quality",
            "--disable-reliability-evidence",
            "discrepancy",
            "consistency",
            "uncertainty",
            "availability",
        ],
    },
    "no_film": {
        "group": "fusion_path",
        "label": "No FiLM path",
        "args": ["--disable-film"],
    },
    "no_attention": {
        "group": "fusion_path",
        "label": "Equal-mean fusion at inference",
        "args": ["--disable-attention"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run restartable three-checkpoint path interventions"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(INTERVENTIONS),
        default=list(INTERVENTIONS),
    )
    parser.add_argument(
        "--seeds",
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


def _evaluation_complete(path: Path, variant: str) -> bool:
    if not path.is_file():
        return False
    report = _read_json(path)
    metrics = report.get("metrics", {})
    intervention = report.get("posthoc_intervention", {})
    return (
        metrics.get("sample_count") == 826
        and all(
            key in metrics
            for key in (
                "accuracy",
                "macro_f1",
                "negative_log_likelihood",
                "multiclass_brier_score",
                "ece",
            )
        )
        and isinstance(intervention, dict)
        and variant != "full"
    )


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
    for seed in args.seeds:
        for name, path in RUNS[seed].items():
            if not path.is_file():
                raise FileNotFoundError(f"missing seed {seed} {name}: {path}")

    state_path = args.state.resolve()
    log_root = state_path.parent / "logs"
    state: dict[str, Any] = {
        "schema_version": 1,
        "valid": False,
        "protocol": "inference_path_intervention_no_retraining",
        "started_at": _now(),
        "updated_at": _now(),
        "variants": list(args.variants),
        "seeds": list(args.seeds),
        "tasks": [],
    }
    _atomic_write(state_path, state)

    try:
        for variant in args.variants:
            specification = INTERVENTIONS[variant]
            for seed in args.seeds:
                task_name = f"{variant}_seed{seed}"
                output_dir = OUTPUT_ROOT / variant / f"seed{seed}" / "test"
                evaluation_path = output_dir / "evaluation.json"
                task: dict[str, Any] = {
                    "task": task_name,
                    "variant": variant,
                    "seed": seed,
                    "status": "running",
                    "started_at": _now(),
                }
                state["tasks"].append(task)
                state["updated_at"] = _now()
                _atomic_write(state_path, state)
                print(f"START {task_name}", flush=True)

                if _evaluation_complete(evaluation_path, variant):
                    task["evaluation"] = "reused_complete"
                else:
                    run = RUNS[seed]
                    command = [
                        sys.executable,
                        "evaluate.py",
                        "--config",
                        str(run["config"]),
                        "--checkpoint",
                        str(run["checkpoint"]),
                        "--split",
                        "test",
                        "--output",
                        str(output_dir),
                        "--num-workers",
                        "0",
                        *specification["args"],
                    ]
                    _run(
                        command,
                        log_root / f"{task_name}.stdout.log",
                        log_root / f"{task_name}.stderr.log",
                    )
                    if not _evaluation_complete(evaluation_path, variant):
                        raise RuntimeError(
                            f"incomplete intervention evaluation: {task_name}"
                        )
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
