from __future__ import annotations

import copy

import pytest

from analyze_error_patterns import analyze_prediction_records


CLASSES = ("a", "b", "c")


def _record(sample_id, target, predicted, confidence):
    return {
        "sample_id": sample_id,
        "target_id": target,
        "predicted_id": predicted,
        "confidence": confidence,
        "correct": target == predicted,
    }


def test_error_analysis_reports_aligned_corrections_and_regressions():
    manifest = [
        {"sample_id": "s1", "source_image_id": "g1"},
        {"sample_id": "s2", "source_image_id": "g1"},
        {"sample_id": "s3", "source_image_id": "g2"},
        {"sample_id": "s4", "source_image_id": "g3"},
    ]
    reference = [
        _record("s1", 0, 0, 0.9),
        _record("s2", 1, 0, 0.8),
        _record("s3", 2, 1, 0.7),
        _record("s4", 2, 2, 0.6),
    ]
    candidate = [
        _record("s1", 0, 0, 0.95),
        _record("s2", 1, 1, 0.85),
        _record("s3", 2, 0, 0.75),
        _record("s4", 2, 1, 0.65),
    ]

    result = analyze_prediction_records(
        reference,
        candidate,
        manifest,
        CLASSES,
        top_k=3,
    )

    assert result["dataset"]["sample_count"] == 4
    assert result["dataset"]["source_image_count"] == 3
    assert result["paired_outcomes"] == {
        "both_correct": 1,
        "candidate_only_correct": 1,
        "reference_only_correct": 1,
        "both_wrong": 1,
        "both_wrong_same_prediction": 0,
        "both_wrong_changed_prediction": 1,
        "net_correct_count_change": 0,
        "accuracy_change": 0.0,
    }
    assert result["models"]["v2b"]["error_count"] == 2
    assert result["models"]["v3"]["error_count"] == 2
    assert result["qualitative_candidate_ids"][
        "candidate_corrections"
    ][0]["sample_id"] == "s2"


def test_error_analysis_rejects_target_and_correct_flag_drift():
    manifest = [{"sample_id": "s1", "source_image_id": "g1"}]
    reference = [_record("s1", 0, 0, 0.9)]
    candidate = [_record("s1", 1, 1, 0.9)]

    with pytest.raises(ValueError, match="target mismatch"):
        analyze_prediction_records(
            reference,
            candidate,
            manifest,
            CLASSES,
        )

    candidate = [copy.deepcopy(reference[0])]
    candidate[0]["correct"] = False
    with pytest.raises(ValueError, match="correct flag mismatch"):
        analyze_prediction_records(
            reference,
            candidate,
            manifest,
            CLASSES,
        )
