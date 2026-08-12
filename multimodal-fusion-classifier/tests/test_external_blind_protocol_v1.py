from __future__ import annotations

import json
from pathlib import Path
import sys

import evaluate_external_blind_v1 as evaluator
import prepare_external_blind_protocol_v1 as protocol_builder


def test_external_blind_protocol_freezes_data_and_analysis_gates() -> None:
    protocol = protocol_builder.build_protocol()
    assert protocol["status"] == "preregistered_awaiting_external_data"
    assert protocol["formal_paper_metric"] is True
    assert len(protocol["class_names"]) == 19
    criteria = protocol["intake_criteria"]
    assert criteria["minimum_total_samples"] == 380
    assert criteria["minimum_source_images"] == 190
    assert criteria["minimum_samples_per_class"] == 20
    assert criteria["minimum_source_images_per_class"] == 10
    assert criteria["exact_development_overlap_allowed"] is False
    assert protocol["evaluation"]["one_time_primary_analysis"] is True
    assert (
        protocol["evaluation"]["model_selection_or_retraining_after_results"]
        is False
    )
    assert protocol["reporting"]["negative_or_null_result_retained"] is True


def test_development_dhash_index_is_complete_and_bound() -> None:
    report = json.loads(
        protocol_builder.DHASH_REPORT.read_text(encoding="utf-8-sig")
    )
    assert report["valid"] is True
    assert report["source_image_count"] == 5990
    assert report["filename_persisted"] is False
    with protocol_builder.DHASH_INDEX.open("r", encoding="utf-8") as handle:
        assert sum(1 for line in handle if line.strip()) == 5990


def test_external_evaluator_defaults_match_protocol(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_external_blind_v1.py"])
    args = evaluator.parse_args()
    assert Path(args.protocol) == protocol_builder.DEFAULT_OUTPUT
    assert Path(args.config) == protocol_builder.CONFIG
    assert Path(args.checkpoint) == protocol_builder.CHECKPOINT
    assert args.batch_size == 16
    assert args.num_workers == 0


def test_external_protocol_binds_every_runtime_source() -> None:
    protocol = protocol_builder.build_protocol()
    expected = {
        "development_integrity",
        "development_dhash_index",
        "contract_source",
        "validator_source",
        "evaluator_source",
        "evaluation_core",
        "dataset_core",
        "model_core",
        "checkpoint_core",
    }
    assert expected <= set(protocol["bindings"])
