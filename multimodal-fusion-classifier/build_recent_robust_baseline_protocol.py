"""Freeze the matched QMF/TMC comparison before any baseline training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_recent_robust_baselines import (
    BOOTSTRAP_REPEATS,
    CONDITIONS,
    PRIMARY_PAIRED_CONDITIONS,
)
from mmfc.model import PROJECT_ROOT
from prepare_recent_robust_baselines import (
    DEFAULT_OUTPUT as CONFIG_ROOT,
    METHODS,
    REFERENCE_CONFIG,
    SEEDS,
    SOURCE_METADATA,
    normalized_matched_settings,
)


REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "qwen-material" / "recent-robust-baselines"
)
DEFAULT_OUTPUT = REPORT_ROOT / "PROTOCOL.json"
SOURCE_FILES = {
    "model_implementation": PROJECT_ROOT / "mmfc" / "recent_baselines.py",
    "training_implementation": (
        PROJECT_ROOT / "mmfc" / "recent_baseline_training.py"
    ),
    "training_entrypoint": PROJECT_ROOT / "train_recent_baseline.py",
    "clean_evaluation": PROJECT_ROOT / "evaluate_recent_baseline.py",
    "condition_evaluation": (
        PROJECT_ROOT / "evaluate_recent_baseline_conditions.py"
    ),
    "config_preparation": (
        PROJECT_ROOT / "prepare_recent_robust_baselines.py"
    ),
    "protocol_builder": (
        PROJECT_ROOT / "build_recent_robust_baseline_protocol.py"
    ),
    "outcome_audit": PROJECT_ROOT / "audit_recent_robust_baselines.py",
    "queue_runner": (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "training"
        / "run_recent_robust_baselines_after_unseen.ps1"
    ),
    "shared_model": PROJECT_ROOT / "mmfc" / "model.py",
    "shared_training": PROJECT_ROOT / "mmfc" / "training.py",
    "shared_evaluation": PROJECT_ROOT / "mmfc" / "evaluation.py",
    "unseen_condition_definition": (
        PROJECT_ROOT / "evaluate_unseen_sensor_conditions.py"
    ),
    "upstream_unseen_protocol": (
        PROJECT_ROOT
        / "reports"
        / "qwen-material"
        / "unseen-sensor-conditions-v1"
        / "PROTOCOL.json"
    ),
    "reference_v3_config": REFERENCE_CONFIG,
    "train_manifest": (
        PROJECT_ROOT / "data" / "qwen-material" / "train.jsonl"
    ),
    "val_manifest": (
        PROJECT_ROOT / "data" / "qwen-material" / "val.jsonl"
    ),
    "test_manifest": (
        PROJECT_ROOT / "data" / "qwen-material" / "test.jsonl"
    ),
    "sensor_manifest": (
        PROJECT_ROOT / "data" / "qwen-material" / "sensors.jsonl"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the recent robust-fusion baseline protocol"
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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    markdown_path = output_path.with_suffix(".md")
    if output_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite recent baseline protocol")
    missing_sources = [
        name for name, path in SOURCE_FILES.items() if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "recent baseline protocol sources missing: "
            + ", ".join(missing_sources)
        )

    configs = {}
    normalized = None
    class_names = None
    for method in METHODS:
        for seed in SEEDS:
            name = f"{method}_seed{seed}"
            path = CONFIG_ROOT / f"{name}.json"
            if not path.is_file():
                raise FileNotFoundError(f"baseline config missing: {path}")
            config = _read_json(path)
            if (
                config.get("recent_baseline", {}).get("method") != method
                or config.get("training", {}).get("seed") != seed
                or config.get("training", {}).get("epochs") != 100
                or config.get("training", {}).get("num_workers") != 0
                or config.get("training", {}).get("batch_size") != 16
                or config.get("training", {}).get("selection_metric")
                != "accuracy"
            ):
                raise ValueError(f"baseline config identity mismatch: {name}")
            current = normalized_matched_settings(config)
            if normalized is None:
                normalized = current
                class_names = [
                    value["name"] if isinstance(value, dict) else str(value)
                    for value in config["classes"]
                ]
            elif current != normalized:
                raise ValueError("baseline matched settings differ")
            configs[name] = _binding(path)

    upstream = _read_json(SOURCE_FILES["upstream_unseen_protocol"])
    if (
        upstream.get("valid") is not True
        or upstream.get("state")
        != "frozen_before_unseen_condition_evaluation"
        or upstream.get("conditions") != CONDITIONS
    ):
        raise ValueError("upstream unseen-condition protocol differs")

    protocol = {
        "schema_version": 1,
        "state": "frozen_before_baseline_training",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Does the proposed full v3 reliability fusion outperform "
            "established quality-aware/evidential decision-fusion adaptations "
            "when encoders, candidate prototypes, sidecars, split, corruption "
            "training, optimization schedule, and seeds are matched?"
        ),
        "methods": list(METHODS),
        "training_seeds": list(SEEDS),
        "run_count": len(METHODS) * len(SEEDS),
        "epochs_per_run": 100,
        "selection_metric": "validation_accuracy",
        "num_workers": 0,
        "conditions": CONDITIONS,
        "primary_paired_conditions": list(PRIMARY_PAIRED_CONDITIONS),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "class_names": class_names,
        "sample_counts": {
            "train": 7002,
            "validation": 826,
            "test": 826,
        },
        "matched_settings": normalized,
        "method_sources": {
            method: {
                key: value
                for key, value in SOURCE_METADATA[method].items()
                if key
                in {
                    "paper",
                    "venue",
                    "paper_url",
                    "official_repository",
                    "inspected_commit",
                }
            }
            for method in METHODS
        },
        "adaptation_contract": {
            "branches": ["visual", "visual_text", "sensor"],
            "visual_text_definition": (
                "all candidate CLIP class prototypes queried only by the "
                "object visual feature; no per-object free-text observation"
            ),
            "qmf": (
                "energy confidence, detached confidence-weighted decision "
                "fusion, per-branch/joint CE, cumulative-loss ranking"
            ),
            "tmc": (
                "softplus evidence, Dirichlet opinions, sequential "
                "Dempster-Shafer combination, annealed evidential objective"
            ),
            "verbatim_original_code_claimed": False,
            "independent_text_observation_claimed": False,
        },
        "execution_dependency": {
            "required_status": (
                "reports/qwen-material/training/"
                "unseen_sensor_conditions_status.json"
            ),
            "required_state": "complete",
            "reason": "avoid concurrent GPU work and bind the v3 condition output",
        },
        "result_inclusion_rule": (
            "Report favorable, null, and adverse baseline outcomes only after "
            "all six 100-epoch runs, clean tests, 11-condition evaluations, "
            "and this frozen audit pass."
        ),
        "configuration_bindings": configs,
        "source_bindings": {
            name: _binding(path) for name, path in SOURCE_FILES.items()
        },
        "boundaries": [
            (
                "Same-input adaptation means the fusion/loss equations are "
                "ported onto this project's branches; it is not a verbatim "
                "reproduction of original paper datasets and encoders."
            ),
            (
                "Candidate class descriptions are fixed semantic prototypes, "
                "not independently collected sample text."
            ),
            (
                "Qwen sidecars and deterministic software corruption are not "
                "physical sensor measurements."
            ),
            (
                "The reused internal test split is not external blind evidence."
            ),
        ],
        "errors": [],
    }
    _atomic_text(
        output_path,
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(
        markdown_path,
        "\n".join(
            [
                "# Recent robust-fusion baseline protocol",
                "",
                "- State: frozen before QMF/TMC baseline training.",
                "- Methods: QMF (ICML 2023) and TMC (ICLR 2021).",
                "- Runs: 2 methods × seeds 42/43/44 × 100 epochs.",
                "- Matched: encoders, candidate prototypes, data, preprocessing, "
                "sensor degradation, optimizer schedule, batch size and seeds.",
                "- Evaluation: clean test plus all 11 frozen sensor conditions.",
                "- Primary paired conditions were fixed before training: "
                + ", ".join(PRIMARY_PAIRED_CONDITIONS)
                + ".",
                "- Execution starts only after the v3 unseen-condition queue "
                "completes.",
                "",
                "The implementations are transparent same-input adaptations. "
                "They are not presented as verbatim reproduction of the "
                "authors' original datasets or encoders.",
                "",
            ]
        ),
    )
    return protocol


def main() -> None:
    protocol = build(parse_args().output)
    print(
        json.dumps(
            {
                "valid": protocol["valid"],
                "state": protocol["state"],
                "methods": protocol["methods"],
                "training_seeds": protocol["training_seeds"],
                "run_count": protocol["run_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
