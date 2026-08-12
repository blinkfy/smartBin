import hashlib

import pytest

from evaluate_sensor_sources import (
    compare_source_predictions,
    verify_protocol_files,
)


def _prediction(sample_id, source_id, target, probabilities):
    return {
        "sample_id": sample_id,
        "source_image_id": source_id,
        "target_id": target,
        "probabilities": probabilities,
    }


def test_sensor_source_comparison_reports_candidate_minus_reference():
    reference = [
        _prediction("a", "source-a", 1, [0.8, 0.2]),
        _prediction("b", "source-b", 0, [0.9, 0.1]),
    ]
    candidate = [
        _prediction("a", "source-a", 1, [0.1, 0.9]),
        _prediction("b", "source-b", 0, [0.8, 0.2]),
    ]

    result = compare_source_predictions(
        reference,
        candidate,
        class_names=["zero", "one"],
        reference_name="qwen",
        candidate_name="rule-v1",
        repeats=10,
        seed=7,
    )

    assert result["candidate"]["metrics"]["accuracy"] == 1.0
    assert result["reference"]["metrics"]["accuracy"] == 0.5
    assert result["outcome_changes"]["candidate_only_correct"] == 1
    assert (
        result["paired_bootstrap"]["differences"]["accuracy"]["point"]
        == pytest.approx(0.5)
    )


def test_sensor_source_comparison_rejects_sample_order_drift():
    reference = [_prediction("a", "source", 0, [0.8, 0.2])]
    candidate = [_prediction("b", "source", 0, [0.8, 0.2])]

    with pytest.raises(ValueError, match="sample order"):
        compare_source_predictions(
            reference,
            candidate,
            class_names=["zero", "one"],
            reference_name="qwen",
            candidate_name="rule-v1",
            repeats=10,
            seed=7,
        )


def test_sensor_source_protocol_hash_rejects_drift(tmp_path):
    source = tmp_path / "sensors.jsonl"
    source.write_bytes(b"locked")
    protocol = {
        "inputs": {
            "sensor": {
                "path": "sensors.jsonl",
                "sha256": hashlib.sha256(b"locked").hexdigest(),
            }
        }
    }
    assert verify_protocol_files(protocol, project_root=tmp_path)

    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_protocol_files(protocol, project_root=tmp_path)
