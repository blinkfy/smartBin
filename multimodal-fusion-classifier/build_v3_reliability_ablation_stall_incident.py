"""Finalize the workers=4 stall incident after the queue has failed closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mmfc.checkpoints import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = PROJECT_ROOT / "reports" / "qwen-material" / "training"
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "confirmatory-ablations"
    / "v3-no-reliability"
)
DEFAULT_OUTPUT = REPORT_ROOT / "WORKERS4_STALL_INCIDENT.json"
ARCHIVED_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "confirmatory-ablations"
    / "interrupted"
    / "v3_reliability_workers4_stall_20260728"
    / "seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize the workers=4 long-horizon stall incident"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, "cpu")
    return {
        **_binding(path),
        "epoch": checkpoint.get("epoch"),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "history_rows": len(checkpoint.get("history", [])),
        "config_fingerprint": checkpoint.get("config_fingerprint"),
    }


def build_incident(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    paths = {
        "liveness": REPORT_ROOT / "WORKERS4_STALL_LIVENESS.json",
        "protocol": REPORT_ROOT / "PROTOCOL.json",
        "seed42_config": (
            PROJECT_ROOT
            / "configs"
            / "confirmatory-ablations"
            / "v3_no_reliability_seed42.json"
        ),
        "partial_checkpoint_audit": REPORT_ROOT / "PARTIAL_CHECKPOINT_AUDIT.json",
        "process_exit": REPORT_ROOT / "seed42" / "TRAINING_PROCESS_EXIT.json",
        "queue_status": (
            TRAINING_ROOT / "v3_reliability_ablation_status.json"
        ),
        "queue_log": TRAINING_ROOT / "v3_reliability_ablation.log",
        "training_stdout": (
            TRAINING_ROOT / "v3_no_reliability_seed42.stdout.log"
        ),
        "training_stderr": (
            TRAINING_ROOT / "v3_no_reliability_seed42.stderr.log"
        ),
        "wrapper_stderr": (
            TRAINING_ROOT / "v3_reliability_ablation.wrapper.stderr.log"
        ),
        "history": ARCHIVED_CHECKPOINT / "history.json",
        "best_checkpoint": ARCHIVED_CHECKPOINT / "best.pt",
        "last_checkpoint": ARCHIVED_CHECKPOINT / "last.pt",
        "capture_source": (
            PROJECT_ROOT / "capture_v3_reliability_ablation_stall.py"
        ),
        "incident_builder_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("incident inputs missing: " + ", ".join(missing))
    liveness = _read_json(paths["liveness"])
    status = _read_json(paths["queue_status"])
    process_exit = _read_json(paths["process_exit"])
    history = _read_json(paths["history"]).get("history")
    partial_audit = _read_json(paths["partial_checkpoint_audit"])
    if (
        liveness.get("valid") is not True
        or liveness.get("diagnosis", {}).get(
            "long_horizon_workers4_not_operationally_valid"
        )
        is not True
        or liveness.get("diagnosis", {}).get("last_completed_epoch") != 69
        or status.get("state") != "failed"
        or process_exit.get("exit_code") != -1
        or process_exit.get("complete_training_bundle") is not False
        or not isinstance(history, list)
        or len(history) != 69
        or history[-1].get("epoch") != 69
        or partial_audit.get("valid") is not False
    ):
        raise ValueError("stall incident identity mismatch")
    best = _checkpoint_identity(paths["best_checkpoint"])
    last = _checkpoint_identity(paths["last_checkpoint"])
    if (
        last.get("epoch") != 69
        or last.get("checkpoint_role") != "last_resume"
        or last.get("history_rows") != 69
        or not isinstance(best.get("epoch"), int)
        or best.get("epoch", 100) > 69
    ):
        raise ValueError("archived partial checkpoints do not match epoch 69")
    report = {
        "schema_version": 1,
        "state": "closed_incomplete_and_archived",
        "valid": True,
        "scope": "v3_reliability_ablation_workers4_long_horizon_stall",
        "facts": {
            "training_seed": 42,
            "num_workers": 4,
            "last_completed_epoch": 69,
            "target_epochs": 100,
            "resume_used": False,
            "process_exit_code_after_controlled_termination": -1,
            "complete_training_bundle": False,
            "training_summary_exists": False,
            "stderr_size_bytes": paths["training_stderr"].stat().st_size,
            "liveness_interval_seconds": liveness["interval_seconds"],
            "liveness_cpu_seconds_delta": liveness["diagnosis"][
                "process_cpu_seconds_delta"
            ],
            "liveness_python_worker_count": len(
                liveness["diagnosis"]["after_python_worker_children"]
            ),
            "stdout_age_seconds_at_diagnosis": liveness["diagnosis"][
                "stdout_age_seconds_after"
            ],
        },
        "archived_partial_checkpoints": {
            "best": best,
            "last": last,
            "history": _binding(paths["history"]),
        },
        "recovery_decision": {
            "reuse_or_resume_partial_run": False,
            "delete_partial_run": False,
            "fresh_training_from_epoch_1": True,
            "replacement_num_workers": 0,
            "reason": (
                "workers=0 is the original stable v3 training setting; the "
                "workers=4 short benchmark established input equivalence but "
                "not 100-epoch Windows liveness"
            ),
            "new_isolated_checkpoint_and_report_roots_required": True,
        },
        "source_bindings": {
            name: _binding(path) for name, path in paths.items()
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        output_path.with_suffix(".md"),
        "\n".join(
            [
                "# workers=4 long-horizon training stall incident",
                "",
                "- Seed 42 stopped making progress after epoch 69.",
                "- A 30-second liveness capture found no log-byte change, no "
                "Python DataLoader workers, and zero main-process CPU-time growth.",
                "- The queue failed closed after controlled termination and "
                "recorded exit code -1.",
                "- Partial best/last/history artifacts were preserved in an "
                "isolated interrupted-checkpoint directory.",
                "- The partial run will not be resumed or used as paper evidence.",
                "- Recovery must start at epoch 1 with workers=0 in new output "
                "roots; scientific model/data/optimizer settings remain unchanged.",
                "",
            ]
        ),
    )
    return report


def main() -> None:
    report = build_incident(parse_args().output)
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "last_completed_epoch": report["facts"]["last_completed_epoch"],
                "output": DEFAULT_OUTPUT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
