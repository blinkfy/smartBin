from __future__ import annotations

import json
from pathlib import Path
import sys

import evaluate_patent_aligned_qwen_distilled_conditions as evaluator
import prepare_object_qwen_scene_derangement as derangement
import prepare_object_qwen_scene_joint_protocol as protocol_builder
import train_patent_aligned_qwen_distilled as trainer


def test_distilled_derangement_report_is_strict() -> None:
    report_path = (
        derangement.DEFAULT_OUTPUT.parent / "SHUFFLE_REPORT.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    assert report["valid"] is True
    assert report["mode"] == derangement.MODE
    assert report["source_image_count"] == 5990
    assert report["split_counts"] == {
        "train": 4823,
        "val": 597,
        "test": 570,
    }
    assert report["self_assignment_count"] == 0
    assert report["split_violation_count"] == 0
    assert report["sensor_payload_multiset_preserved"] is True


def test_development_protocol_preserves_claim_boundaries() -> None:
    protocol = protocol_builder.build_protocol()
    assert protocol["formal_paper_metric"] is False
    assert protocol["training"] == {
        "seed": 42,
        "epochs": 20,
        "patience": 5,
        "batch_size": 32,
        "image_size": 640,
        "workers": 4,
        "freeze_layers": 24,
        "amp": True,
        "learning_rate": 0.0001,
        "weight_decay": 0.0005,
    }
    assert protocol["lineage"]["teacher_uses_training_annotations"] is True
    assert protocol["lineage"]["joint_inference_uses_annotations"] is False
    assert protocol["data"]["direct_full_image_qwen_api"] is False
    assert (
        protocol["evaluation"][
            "development_only_due_prior_test_diagnostic_access"
        ]
        is True
    )


def test_training_defaults_match_frozen_runtime(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_patent_aligned_qwen_distilled.py"])
    args = trainer.parse_args()
    assert Path(args.sensor_observations) == trainer.DEFAULT_SENSOR_OBSERVATIONS
    assert Path(args.project) == trainer.DEFAULT_PROJECT
    assert Path(args.report_dir) == trainer.DEFAULT_REPORT
    assert args.name == trainer.DEFAULT_NAME
    assert args.epochs == 20
    assert args.batch_size == 32
    assert args.workers == 4
    assert args.seed == 42


def test_condition_evaluator_defaults_are_isolated(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_patent_aligned_qwen_distilled_conditions.py"],
    )
    args = evaluator.parse_args()
    assert Path(args.checkpoint) == evaluator.DEFAULT_CHECKPOINT
    assert (
        Path(args.sensor_observations)
        == trainer.DEFAULT_SENSOR_OBSERVATIONS
    )
    assert Path(args.shuffled_observations) == evaluator.DEFAULT_SHUFFLED
    assert Path(args.output) == evaluator.DEFAULT_OUTPUT
