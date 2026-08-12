from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aggregate_patent_stack_qwen_formal import (
    PRIMARY_METRICS,
    summarize_conditions,
)
from evaluate_external_sensor_conditions import derange_sensor_records
from evaluate_patent_aligned_qwen_formal_conditions import (
    validate_formal_run_bindings,
)
from prepare_patent_stack_qwen_formal_protocol import (
    PROTOCOL_NAME,
    RUN_PROTOCOL_NAME,
    build_protocol,
)
from prepare_qwen_sensor_derangement import validate_derangement
from run_patent_stack_qwen_formal import (
    build_command_plan,
    validate_protocol,
)
from train_patent_aligned_qwen_formal import validate_qwen_gate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(split: str, index: int) -> dict:
    value = index / 10.0
    grid = [[value for _ in range(4)] for _ in range(4)]
    return {
        "source_image_id": f"{split}-{index}",
        "split": split,
        "image_path": f"images/{split}-{index}.jpg",
        "grid": grid,
        "normalized_grid": grid,
        "properties": {
            "weight_g": index + 1.0,
            "temperature_c": 20.0 + index,
            "moisture_pct": 30.0 + index,
            "conductivity_us_cm": 100.0 + index,
            "odor_index": value,
        },
        "source": {
            "observation_scope": "full_image",
            "full_image_only": True,
            "annotation_inputs": [],
        },
    }


def test_extension_protocol_is_additive_fixed_parent_adaptation() -> None:
    protocol = build_protocol()

    assert protocol["protocol"] == PROTOCOL_NAME
    assert protocol["run_protocol"] == RUN_PROTOCOL_NAME
    assert protocol["relationship_to_legacy_v2"][
        "legacy_protocol_modified"
    ] is False
    assert protocol["lineage"]["shared_parent_across_seeds"] is True
    assert protocol["lineage"]["independent_end_to_end_replicates"] is False
    assert protocol["training"]["seeds"] == [42, 43, 44]
    assert protocol["training"]["workers"] == 4
    assert validate_protocol(protocol) == []


def test_extension_command_plan_uses_current_v6_lineage() -> None:
    protocol = build_protocol()
    plan = build_command_plan(
        protocol,
        python_executable="python",
        device="0",
    )

    assert [run["seed"] for run in plan["runs"]] == [42, 43, 44]
    assert plan["derange"][1] == "prepare_qwen_sensor_derangement.py"
    for run in plan["runs"]:
        assert run["train"][1] == "train_patent_aligned_qwen_formal.py"
        assert "--base-weights" in run["train"]
        assert protocol["lineage"]["parent_checkpoint"]["path"] in run["train"]
        assert run["train"][run["train"].index("--workers") + 1] == "4"
        assert run["train"][run["train"].index("--batch-size") + 1] == "32"
        assert (
            run["condition"][1]
            == "evaluate_patent_aligned_qwen_formal_conditions.py"
        )
        assert "--shuffled-observations" in run["condition"]
    assert plan["aggregate"][1] == "aggregate_patent_stack_qwen_formal.py"


def test_derangement_preserves_split_and_sensor_payload_multiset() -> None:
    original = [
        _record(split, index)
        for split in ("train", "val", "test")
        for index in range(3)
    ]

    shuffled = derange_sensor_records(original)
    audit = validate_derangement(original, shuffled)

    assert audit["valid"] is True
    assert audit["self_assignment_count"] == 0
    assert audit["split_violation_count"] == 0
    assert audit["sensor_payload_multiset_preserved"] is True
    assert audit["split_counts"] == {"train": 3, "val": 3, "test": 3}


def test_qwen_gate_requires_mode_coverage_and_hash(tmp_path: Path) -> None:
    observations = tmp_path / "observations.jsonl"
    observations.write_text("{}\n", encoding="utf-8")
    digest = _sha256(observations)
    contract = {
        "full_image_only": True,
        "annotation_files_read": False,
        "label_files_read": False,
        "bbox_inputs": False,
        "polygon_inputs": False,
        "class_inputs": False,
    }
    build = {
        "valid": True,
        "mode": "qwen",
        "source_image_count": 5990,
        "output": {"sha256": digest},
    }
    validation = {
        "valid": True,
        "mode": "qwen",
        "error_count": 0,
        "source_image_count": 5990,
        "qwen_success_count": 5990,
        "fallback_count": 0,
        "source_contract": contract,
        "observations_sha256": digest,
    }
    build_path = tmp_path / "BUILD_REPORT.json"
    validation_path = tmp_path / "VALIDATION_REPORT.json"
    build_path.write_text(json.dumps(build), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    paths = {
        "sensor_observations": observations,
        "sensor_build": build_path,
        "sensor_validation": validation_path,
    }

    gate = validate_qwen_gate(paths)

    assert gate["coverage"] == 1.0
    validation["qwen_success_count"] = 5929
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(ValueError, match="formal adaptation gate"):
        validate_qwen_gate(paths)


def test_formal_evaluator_rejects_development_evidence() -> None:
    protocol = {
        "protocol": RUN_PROTOCOL_NAME,
        "formal_paper_metric": True,
        "sensor_source": {"mode": "qwen"},
        "qwen_formal_gate": {"observations_sha256": "observations"},
    }
    evaluation = {
        "valid": True,
        "formal_paper_metric": True,
        "checkpoint": {"sha256": "checkpoint"},
        "sensor_observations": {
            "mode": "qwen",
            "sha256": "observations",
        },
    }

    validate_formal_run_bindings(
        protocol,
        evaluation,
        checkpoint_sha256="checkpoint",
        observations_sha256="observations",
    )
    evaluation["formal_paper_metric"] = False
    with pytest.raises(ValueError, match="formal v6 Qwen"):
        validate_formal_run_bindings(
            protocol,
            evaluation,
            checkpoint_sha256="checkpoint",
            observations_sha256="observations",
        )


def test_formal_summary_uses_same_seed_paired_differences() -> None:
    rows = []
    for seed_index in range(3):
        clean = {
            metric: 0.9 + seed_index * 0.01 for metric in PRIMARY_METRICS
        }
        conditions = {
            "clean_matched": clean,
            "rgb_fallback": {
                metric: value - 0.01 for metric, value in clean.items()
            },
            "shuffled": {
                metric: value - 0.02 for metric, value in clean.items()
            },
            "missing": {
                metric: value - 0.03 for metric, value in clean.items()
            },
            "noise_0_15": {
                metric: value - 0.04 for metric, value in clean.items()
            },
            "bias_drift_0_25": {
                metric: value - 0.05 for metric, value in clean.items()
            },
            "stuck": {
                metric: value - 0.06 for metric, value in clean.items()
            },
        }
        rows.append({"metrics": conditions})

    summary = summarize_conditions(rows)

    fallback = summary["paired_differences"][
        "clean_minus_rgb_fallback"
    ]["metrics/mAP50-95(M)"]
    shuffled = summary["paired_differences"]["clean_minus_shuffled"][
        "metrics/mAP50-95(M)"
    ]
    assert abs(fallback["mean"] - 0.01) < 1e-12
    assert abs(shuffled["mean"] - 0.02) < 1e-12
    assert fallback["count"] == 3
