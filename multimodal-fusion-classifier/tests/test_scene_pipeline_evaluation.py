import hashlib
from pathlib import Path

import pytest

from evaluate_scene_pipeline import (
    _result_path_matches,
    box_iou,
    greedy_iou_match,
    paired_source_bootstrap,
    summarize_end_to_end,
    verify_protocol_file_hashes,
)


def test_box_iou_handles_overlap_and_disjoint_boxes():
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(
        25 / 175
    )
    assert box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_greedy_iou_match_is_one_to_one_and_prefers_highest_iou():
    ground_truth = [
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [20, 20, 30, 30]},
    ]
    detections = [
        {"bbox": [1, 1, 9, 9]},
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [20, 20, 30, 30]},
    ]

    result = greedy_iou_match(
        ground_truth,
        detections,
        iou_threshold=0.5,
    )

    assert result["matches"] == [
        {"ground_truth_index": 0, "detection_index": 1, "iou": 1.0},
        {"ground_truth_index": 1, "detection_index": 2, "iou": 1.0},
    ]
    assert result["unmatched_detection_indices"] == [0]


def test_end_to_end_summary_counts_wrong_and_unmatched_classes():
    images = [
        {
            "source_image_id": "source",
            "ground_truth": [
                {"class_id": 0, "bbox": [0, 0, 10, 10]},
                {"class_id": 1, "bbox": [20, 20, 30, 30]},
            ],
            "detections": [
                {
                    "bbox": [0, 0, 10, 10],
                    "detector_confidence": 0.9,
                    "detector_label_id": 0,
                    "classifications": {
                        "missing": {"label_id": 0},
                        "rule-v1": {"label_id": 0},
                    },
                },
                {
                    "bbox": [20, 20, 30, 30],
                    "detector_confidence": 0.8,
                    "detector_label_id": 0,
                    "classifications": {
                        "missing": {"label_id": 1},
                        "rule-v1": {"label_id": 1},
                    },
                },
                {
                    "bbox": [40, 40, 50, 50],
                    "detector_confidence": 0.7,
                    "detector_label_id": 1,
                    "classifications": {
                        "missing": {"label_id": 0},
                        "rule-v1": {"label_id": 1},
                    },
                },
            ],
        }
    ]

    detector = summarize_end_to_end(
        images,
        class_names=["a", "b"],
        confidence_threshold=0.5,
        match_iou=0.5,
        prediction_source="detector",
    )
    missing = summarize_end_to_end(
        images,
        class_names=["a", "b"],
        confidence_threshold=0.5,
        match_iou=0.5,
        prediction_source="missing",
    )

    assert detector["localized_count"] == 2
    assert detector["localized_and_classified_count"] == 1
    assert detector["end_to_end_recall"] == pytest.approx(0.5)
    assert detector["end_to_end_precision"] == pytest.approx(1 / 3)
    assert detector["macro_f1"] == pytest.approx(1 / 3)
    assert missing["classification_accuracy_on_localized"] == 1.0
    assert missing["end_to_end_recall"] == 1.0
    assert missing["macro_f1"] == pytest.approx(((2 / 3) + 1.0) / 2)


def test_end_to_end_summary_applies_confidence_threshold():
    images = [
        {
            "source_image_id": "source",
            "ground_truth": [{"class_id": 0, "bbox": [0, 0, 10, 10]}],
            "detections": [
                {
                    "bbox": [0, 0, 10, 10],
                    "detector_confidence": 0.7,
                    "detector_label_id": 0,
                }
            ],
        }
    ]

    result = summarize_end_to_end(
        images,
        class_names=["a"],
        confidence_threshold=0.8,
        match_iou=0.5,
        prediction_source="detector",
    )

    assert result["detection_count"] == 0
    assert result["localized_count"] == 0
    assert result["end_to_end_recall"] == 0.0


def test_paired_source_bootstrap_reports_candidate_minus_reference():
    images = [
        {
            "source_image_id": "source",
            "ground_truth": [{"class_id": 1, "bbox": [0, 0, 10, 10]}],
            "detections": [
                {
                    "bbox": [0, 0, 10, 10],
                    "detector_confidence": 0.9,
                    "detector_label_id": 0,
                    "classifications": {
                        "missing": {"label_id": 1},
                        "rule-v1": {"label_id": 1},
                    },
                }
            ],
        }
    ]

    result = paired_source_bootstrap(
        images,
        class_names=["a", "b"],
        confidence_threshold=0.5,
        match_iou=0.5,
        reference_source="detector",
        candidate_source="missing",
        repeats=10,
        seed=7,
    )

    delta = result["differences"]["end_to_end_recall"]
    assert delta["point"] == 1.0
    assert delta["lower"] == 1.0
    assert delta["upper"] == 1.0


def test_protocol_hash_verification_rejects_file_drift(tmp_path):
    source = tmp_path / "input.json"
    source.write_bytes(b"locked")
    digest = hashlib.sha256(b"locked").hexdigest()
    protocol = {
        "inputs": {
            "data": {
                "path": "input.json",
                "sha256": digest,
            },
            "pending_checkpoint": {
                "path": "best.pt",
                "status": "pending_training",
            },
        }
    }

    checks = verify_protocol_file_hashes(protocol, project_root=tmp_path)
    assert checks[0]["sha256"] == digest

    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_protocol_file_hashes(protocol, project_root=tmp_path)


def test_result_path_accepts_ultralytics_synthetic_list_index(tmp_path):
    original = tmp_path / "original.jpg"
    original.write_bytes(b"image")

    assert _result_path_matches(
        tmp_path / "original.jpg",
        original_path=original,
        result_index=0,
    )
    assert _result_path_matches(
        Path("image0.jpg"),
        original_path=original,
        result_index=0,
    )
    assert not _result_path_matches(
        Path("image1.jpg"),
        original_path=original,
        result_index=0,
    )
