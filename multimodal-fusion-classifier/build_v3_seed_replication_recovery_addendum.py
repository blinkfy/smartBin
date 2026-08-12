"""Freeze the recovery amendment for the v3 seed-replication queue."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "full_cross_modal_quality"
    / "seed-replication"
)
DEFAULT_OUTPUT = REPORT_ROOT / "QUEUE_RECOVERY_ADDENDUM.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the seed-replication queue recovery addendum"
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


def build_addendum(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("seed-replication recovery addendum is immutable")
    training_root = PROJECT_ROOT / "reports" / "qwen-material" / "training"
    paths = {
        "builder_source": Path(__file__),
        "protocol": REPORT_ROOT / "PROTOCOL.json",
        "incident": REPORT_ROOT / "QUEUE_INCIDENT.json",
        "frozen_queue_source": training_root / "run_v3_seed_replication.ps1",
        "frozen_audit_source": PROJECT_ROOT / "audit_v3_seed_replication.py",
        "recovery_queue_source": (
            training_root / "run_v3_seed_replication_recovery.ps1"
        ),
        "recovery_audit_source": (
            PROJECT_ROOT / "audit_v3_seed_replication_recovery.py"
        ),
        "seed43_pre_recovery_audit": (
            REPORT_ROOT / "seed43" / "CHECKPOINT_AUDIT.pre_recovery.json"
        ),
        "seed43_training_summary": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "v3_seed_replicates"
            / "full_cross_modal_quality_seed43"
            / "training_summary.json"
        ),
        "seed43_history": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "v3_seed_replicates"
            / "full_cross_modal_quality_seed43"
            / "history.json"
        ),
        "seed43_best_checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "v3_seed_replicates"
            / "full_cross_modal_quality_seed43"
            / "best.pt"
        ),
        "seed43_last_checkpoint": (
            PROJECT_ROOT
            / "checkpoints"
            / "ablations"
            / "v3_seed_replicates"
            / "full_cross_modal_quality_seed43"
            / "last.pt"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("recovery addendum inputs missing: " + ", ".join(missing))
    protocol = _read_json(paths["protocol"])
    incident = _read_json(paths["incident"])
    checkpoint_audit = _read_json(paths["seed43_pre_recovery_audit"])
    recovery_queue = paths["recovery_queue_source"].read_text(encoding="utf-8-sig")
    recovery_audit = paths["recovery_audit_source"].read_text(encoding="utf-8-sig")
    errors: list[str] = []
    if (
        protocol.get("status") != "frozen_before_replicate_training"
        or protocol.get("replicates", {}).get("seeds") != [43, 44]
        or protocol.get("replicates", {}).get("all_training_seeds")
        != [42, 43, 44]
    ):
        errors.append("original seed-replication protocol identity mismatch")
    if incident.get("state") != "complete" or incident.get("valid") is not True:
        errors.append("seed-replication queue incident is not valid")
    completed = incident.get("completed_seed43", {})
    if (
        checkpoint_audit.get("valid") is not True
        or checkpoint_audit.get("history_json_rows") != 100
        or checkpoint_audit.get("last", {}).get("epoch") != 100
        or checkpoint_audit.get("best", {}).get("sha256")
        != _sha256(paths["seed43_best_checkpoint"])
        or checkpoint_audit.get("last", {}).get("sha256")
        != _sha256(paths["seed43_last_checkpoint"])
        or completed.get("best_checkpoint", {}).get("sha256")
        != _sha256(paths["seed43_best_checkpoint"])
        or completed.get("last_checkpoint", {}).get("sha256")
        != _sha256(paths["seed43_last_checkpoint"])
    ):
        errors.append("seed 43 complete bundle changed before recovery")
    required_queue_fragments = (
        "$historyRows = @($historyDocument.history)",
        "$ReportBase = ",
        "$seedReportDir = Join-Path",
        "REUSE seed43_training",
        "Invoke-FreshTraining -Seed 44",
        "audit_v3_seed_replication_recovery.py",
        "build_v3_seed_replication_addendum.py",
    )
    if any(fragment not in recovery_queue for fragment in required_queue_fragments):
        errors.append("recovery queue is missing an authorized control")
    if '$reportRoot = "$ReportRoot/seed${seed}"' in recovery_queue:
        errors.append("recovery queue retains the case-insensitive path collision")
    if (
        "def _history_rows" not in recovery_audit
        or 'document.get("history", [])' not in recovery_audit
        or "seed43_training_reused_after_queue_false_negative" not in recovery_audit
    ):
        errors.append("recovery audit does not handle the recorded incident")
    seed44_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "ablations"
        / "v3_seed_replicates"
        / "full_cross_modal_quality_seed44"
    )
    if seed44_dir.exists() and any(seed44_dir.iterdir()):
        errors.append("seed 44 training started before the recovery addendum")
    if any(
        path.exists()
        for path in (
            REPORT_ROOT / "VALIDATION_SEED_SUMMARY.json",
            REPORT_ROOT / "TEST_SEED_SUMMARY.json",
            REPORT_ROOT / "OUTCOME.json",
        )
    ):
        errors.append("downstream summaries existed before recovery addendum")
    if errors:
        raise ValueError("; ".join(errors))

    addendum = {
        "schema_version": 1,
        "status": (
            "frozen_after_seed43_queue_false_negative_before_recovery"
        ),
        "valid": True,
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": _binding(paths["protocol"]),
        "incident": _binding(paths["incident"]),
        "authorized_implementation_changes": [
            (
                "read 100 training rows from the history member of "
                "history.json"
            ),
            (
                "use distinct PowerShell names for the immutable report base "
                "and per-seed report directory"
            ),
            (
                "replace the final outcome auditor with the incident-aware "
                "recovery auditor"
            ),
        ],
        "invariants": {
            "seed43_training_rerun": False,
            "seed43_completed_epochs": 100,
            "seed43_best_epoch": checkpoint_audit["best"]["epoch"],
            "seed43_last_epoch": 100,
            "seed43_checkpoints_unchanged": True,
            "seed44_training_started": False,
            "seed44_fresh_training_required": True,
            "scientific_config_unchanged": True,
            "model_loss_and_data_unchanged": True,
            "epochs_and_evaluation_unchanged": True,
            "best_seed_replacement_forbidden": True,
        },
        "source_bindings": {
            name: _binding(path) for name, path in paths.items()
        },
        "errors": [],
        "boundaries": [
            (
                "This addendum changes queue control and outcome auditing only; "
                "it does not change model training or evaluation semantics."
            ),
            (
                "Seed 43 is reused only after an independent checkpoint audit "
                "proved a complete fresh 100-epoch run."
            ),
            (
                "The original protocol, failed queue, nested process-exit "
                "record, and incident archive remain immutable."
            ),
        ],
    }
    _atomic_write(
        output_path,
        json.dumps(addendum, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        markdown_path,
        "\n".join(
            [
                "# v3 seed replication queue recovery addendum",
                "",
                "This addendum was frozen after the seed 43 queue false "
                "negative and before seed 44 training.",
                "",
                "It authorizes only two control-plane fixes: reading the "
                "`history` member and separating the report-base variable from "
                "the per-seed report directory. Seed 43 is not retrained.",
                "",
                "All model, loss, data, optimizer, schedule, epoch, checkpoint "
                "selection, and evaluation settings remain unchanged.",
                "",
            ]
        ),
    )
    return addendum


def main() -> None:
    report = build_addendum(parse_args().output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "valid": report["valid"],
                "seed43_training_rerun": report["invariants"][
                    "seed43_training_rerun"
                ],
                "seed44_training_started": report["invariants"][
                    "seed44_training_started"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
