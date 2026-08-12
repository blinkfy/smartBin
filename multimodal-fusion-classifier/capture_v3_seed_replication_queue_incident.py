"""Capture the seed-43 queue false negative without touching its checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from audit_v3_seed_replication import validate_protocol


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "full_cross_modal_quality"
    / "seed-replication"
)
TRAINING_ROOT = PROJECT_ROOT / "reports" / "qwen-material" / "training"
CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "ablations"
    / "v3_seed_replicates"
    / "full_cross_modal_quality_seed43"
)
ARCHIVE_ROOT = (
    TRAINING_ROOT
    / "interrupted"
    / "v3_seed_replication_queue_false_negative_20260728"
)
DEFAULT_OUTPUT = REPORT_ROOT / "QUEUE_INCIDENT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the v3 seed-replication queue false negative"
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


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _display(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _archive_inputs(inputs: dict[str, Path]) -> dict[str, Path]:
    if ARCHIVE_ROOT.exists() and any(ARCHIVE_ROOT.iterdir()):
        raise FileExistsError(f"incident archive is not empty: {ARCHIVE_ROOT}")
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    archived = {}
    for name, source in inputs.items():
        destination = ARCHIVE_ROOT / f"{name}{source.suffix}"
        shutil.copy2(source, destination)
        archived[name] = destination
    return archived


def build_incident(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("seed-replication incident is immutable")
    protocol_path = REPORT_ROOT / "PROTOCOL.json"
    queue_source = TRAINING_ROOT / "run_v3_seed_replication.ps1"
    status_path = (
        TRAINING_ROOT / "full_cross_modal_quality.seed_replicates_status.json"
    )
    queue_log = TRAINING_ROOT / "full_cross_modal_quality.seed_replicates.log"
    stdout_path = TRAINING_ROOT / "full_cross_modal_quality_seed43.stdout.log"
    stderr_path = TRAINING_ROOT / "full_cross_modal_quality_seed43.stderr.log"
    postprocess_status = (
        TRAINING_ROOT
        / "full_cross_modal_quality.seed_replicates_postprocess_status.json"
    )
    nested_exit = REPORT_ROOT / "seed43" / "seed43" / "TRAINING_PROCESS_EXIT.json"
    checkpoint_audit_path = (
        REPORT_ROOT / "seed43" / "CHECKPOINT_AUDIT.pre_recovery.json"
    )
    summary_path = CHECKPOINT_DIR / "training_summary.json"
    history_path = CHECKPOINT_DIR / "history.json"
    best_path = CHECKPOINT_DIR / "best.pt"
    last_path = CHECKPOINT_DIR / "last.pt"
    inputs = {
        "queue_status": status_path,
        "queue_log": queue_log,
        "seed43_stdout": stdout_path,
        "seed43_stderr": stderr_path,
        "postprocess_status": postprocess_status,
        "nested_process_exit": nested_exit,
        "checkpoint_audit": checkpoint_audit_path,
    }
    required = {
        "protocol": protocol_path,
        "queue_source": queue_source,
        "training_summary": summary_path,
        "history": history_path,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        **inputs,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("incident inputs are missing: " + ", ".join(missing))

    protocol_validation = validate_protocol(protocol_path)
    status = _read_json(status_path)
    postprocess = _read_json(postprocess_status)
    process_exit = _read_json(nested_exit)
    summary = _read_json(summary_path)
    history = _read_json(history_path)
    checkpoint_audit = _read_json(checkpoint_audit_path)
    queue_text = queue_source.read_text(encoding="utf-8-sig")
    errors: list[str] = []
    history_rows = history.get("history", [])
    if protocol_validation["valid"] is not True:
        errors.append("original frozen protocol no longer validates")
    if (
        status.get("state") != "failed"
        or "complete 100-epoch bundle" not in str(status.get("message", ""))
        or postprocess.get("state") != "failed"
    ):
        errors.append("queue failure state does not match the incident")
    if (
        process_exit.get("seed") != 43
        or process_exit.get("exit_code") != 0
        or process_exit.get("zero_exit") is not True
        or process_exit.get("complete_training_bundle") is not False
    ):
        errors.append("nested process-exit record does not capture the false negative")
    if (
        summary.get("completed_epochs") != 100
        or summary.get("resumed_from") is not None
        or summary.get("checkpoints_saved") is not True
        or not isinstance(history_rows, list)
        or len(history_rows) != 100
        or history_rows[-1].get("epoch") != 100
    ):
        errors.append("seed 43 training files are not a complete fresh run")
    if (
        checkpoint_audit.get("valid") is not True
        or checkpoint_audit.get("training_summary_completed_epochs") != 100
        or checkpoint_audit.get("history_json_rows") != 100
        or checkpoint_audit.get("last", {}).get("epoch") != 100
        or checkpoint_audit.get("best", {}).get("sha256") != _sha256(best_path)
        or checkpoint_audit.get("last", {}).get("sha256") != _sha256(last_path)
    ):
        errors.append("independent seed 43 checkpoint audit mismatch")
    if (
        "@($history).Count -eq 100" not in queue_text
        or '$reportRoot = "$ReportRoot/seed${seed}"' not in queue_text
    ):
        errors.append("frozen queue source no longer exposes both incident causes")
    seed44_dir = CHECKPOINT_DIR.parent / "full_cross_modal_quality_seed44"
    if seed44_dir.exists() and any(seed44_dir.iterdir()):
        errors.append("seed 44 artifacts existed before incident recovery")
    if any(
        path.exists()
        for path in (
            REPORT_ROOT / "VALIDATION_SEED_SUMMARY.json",
            REPORT_ROOT / "TEST_SEED_SUMMARY.json",
            REPORT_ROOT / "OUTCOME.json",
        )
    ):
        errors.append("downstream seed summaries existed before incident recovery")
    if errors:
        raise ValueError("; ".join(errors))

    archived = _archive_inputs(inputs)
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "v3_seed_replication_seed43_queue_false_negative",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "failure": {
            "queue_state": status,
            "postprocess_state": postprocess,
            "training_process_exit_code": 0,
            "cause_history_shape": (
                "history.json is an object with a 100-row history member, while "
                "the queue counted the root object as an array"
            ),
            "cause_powershell_scope": (
                "PowerShell variable names are case-insensitive; loop-local "
                "$reportRoot overwrote global $ReportRoot and duplicated seed43"
            ),
        },
        "completed_seed43": {
            "completed_epochs": 100,
            "resume_used": False,
            "best_epoch": checkpoint_audit["best"]["epoch"],
            "last_epoch": 100,
            "best_checkpoint": _binding(best_path),
            "last_checkpoint": _binding(last_path),
            "training_summary": _binding(summary_path),
            "history": _binding(history_path),
            "checkpoint_audit": _binding(checkpoint_audit_path),
        },
        "recovery": {
            "seed43_training_rerun": False,
            "seed43_requires_independent_audit": True,
            "seed43_requires_validation_and_test_evaluation": True,
            "seed44_training_not_started": True,
            "original_protocol_and_queue_remain_immutable": True,
        },
        "source_bindings": {
            "incident_builder": _binding(Path(__file__)),
            "protocol": _binding(protocol_path),
            "frozen_queue_source": _binding(queue_source),
            **{
                f"archived_{name}": _binding(path)
                for name, path in archived.items()
            },
        },
        "errors": [],
    }
    _atomic_write(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        markdown_path,
        "\n".join(
            [
                "# v3 seed replication queue incident",
                "",
                "Seed 43 completed all 100 epochs and returned exit code 0. "
                "Independent checkpoint audit validated both checkpoints.",
                "",
                "The queue stopped on two reporting-control defects: it counted "
                "the history JSON root instead of its `history` member, and a "
                "case-insensitive PowerShell variable collision duplicated the "
                "`seed43` report directory.",
                "",
                "Seed 43 training and checkpoints are retained without rerun. "
                "Seed 44 had not started when this incident was captured.",
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
                "best_epoch": report["completed_seed43"]["best_epoch"],
                "archive": _display(ARCHIVE_ROOT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
