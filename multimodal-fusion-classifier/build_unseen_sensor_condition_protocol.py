"""Freeze the unseen sensor and availability-failure evaluation protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from evaluate_unseen_sensor_conditions import (
    BOOTSTRAP_REPEATS,
    CONDITIONS,
    CORRUPTION_SEED,
    DEFAULT_PROTOCOL,
    REPORT_ROOT,
    SEED_RUNS,
)
from mmfc.training import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze unseen sensor-condition protocol"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_PROTOCOL)
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
        "path": Path(os.path.relpath(resolved, PROJECT_ROOT)).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build(output: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    output = _resolve(output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite unseen-condition protocol")
    if REPORT_ROOT.exists() and any(REPORT_ROOT.iterdir()):
        raise ValueError("unseen-condition report root is not empty")
    sources = {
        "protocol_builder": Path(__file__),
        "evaluator": PROJECT_ROOT / "evaluate_unseen_sensor_conditions.py",
        "deferred_queue": (
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "training"
            / "run_unseen_sensor_conditions_after_components.ps1"
        ),
        "model_source": PROJECT_ROOT / "mmfc" / "model.py",
        "checkpoint_loader": PROJECT_ROOT / "mmfc" / "checkpoints.py",
        "training_data_source": PROJECT_ROOT / "mmfc" / "training.py",
        "evaluation_metrics_source": PROJECT_ROOT / "mmfc" / "evaluation.py",
        "test_manifest": PROJECT_ROOT / "data" / "qwen-material" / "test.jsonl",
        "sensor_manifest": (
            PROJECT_ROOT / "data" / "qwen-material" / "sensors.jsonl"
        ),
    }
    for seed, paths in SEED_RUNS.items():
        sources[f"seed{seed}_config"] = paths["config"]
        sources[f"seed{seed}_checkpoint"] = paths["checkpoint"]
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("protocol inputs missing: " + ", ".join(missing))
    report = {
        "schema_version": 1,
        "state": "frozen_before_unseen_condition_evaluation",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "three_seed_unseen_sensor_and_availability_failures",
        "model_seeds": list(SEED_RUNS),
        "corruption_seed": CORRUPTION_SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "cluster_key": "source_image_id",
        "sample_count_per_seed": 826,
        "source_image_count_per_seed": 570,
        "conditions": CONDITIONS,
        "primary_questions": [
            (
                "How does full v3 behave under corruption families absent from "
                "training and under severity beyond the registered range?"
            ),
            (
                "How does the deterministic fallback behave when availability "
                "has false-negative or false-positive errors?"
            ),
        ],
        "execution_policy": {
            "wait_for_component_training_queue": True,
            "do_not_share_gpu_with_training": True,
            "model_selection_or_tuning_from_results": False,
            "retain_all_conditions": True,
        },
        "claim_rules": [
            (
                "Results describe deterministic software interventions on the "
                "reused internal test only."
            ),
            (
                "Out-of-range and unseen conditions cannot be described as "
                "measured hardware fault distributions."
            ),
            (
                "Availability error effects must be reported separately from "
                "correctly detected missingness."
            ),
            (
                "Null and adverse outcomes are retained without changing the "
                "model or corruption definitions."
            ),
        ],
        "source_bindings": {
            name: _binding(path) for name, path in sorted(sources.items())
        },
        "errors": [],
    }
    _atomic_write(output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(
        markdown,
        "\n".join(
            [
                "# Unseen sensor and availability condition protocol",
                "",
                "- Status: frozen before evaluation.",
                "- Full v3 model seeds: 42, 43, 44.",
                f"- Conditions: {len(CONDITIONS)} including clean.",
                "- Paired source-image clustered bootstrap: 2,000 repeats.",
                "- Execution waits for the component-training GPU queue.",
                "",
                (
                    "The extension covers severity extrapolation, unseen "
                    "software corruptions, and availability false positives/"
                    "negatives. It does not represent hardware fault validation."
                ),
                "",
            ]
        ),
    )
    return report


def main() -> None:
    report = build(parse_args().output)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "state": report["state"],
                "condition_count": len(report["conditions"]),
                "model_seeds": report["model_seeds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
