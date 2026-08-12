import numpy as np
import pytest

from evaluate_scene_strata import (
    assign_size_stratum,
    bbox_overlap_flags,
    build_contributions,
    summarize_contributions,
)


def test_size_strata_use_frozen_half_open_intervals():
    assert (
        assign_size_stratum(
            0.1,
            lower_quartile=0.2,
            upper_quartile=0.8,
        )
        == "relative_small"
    )
    assert (
        assign_size_stratum(
            0.2,
            lower_quartile=0.2,
            upper_quartile=0.8,
        )
        == "relative_medium"
    )
    assert (
        assign_size_stratum(
            0.8,
            lower_quartile=0.2,
            upper_quartile=0.8,
        )
        == "relative_large"
    )


def test_bbox_overlap_proxy_requires_positive_intersection_area():
    flags = bbox_overlap_flags(
        [
            [0.0, 0.0, 0.4, 0.4],
            [0.3, 0.3, 0.6, 0.6],
            [0.6, 0.0, 0.8, 0.2],
        ]
    )
    assert flags == [True, True, False]
    assert bbox_overlap_flags(
        [[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]]
    ) == [False, False]


def test_scene_strata_contributions_share_localization_across_sources():
    images = [
        {
            "source_image_id": "a",
            "ground_truth": [
                {"class_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0]}
            ],
            "detections": [
                {
                    "bbox": [0.0, 0.0, 10.0, 10.0],
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
    strata = {
        "a": [
            {
                "class_id": 1,
                "size_stratum": "relative_small",
                "density_stratum": "single_object",
                "overlap_stratum": "bbox_isolated",
            }
        ]
    }

    source_ids, contributions = build_contributions(
        images,
        strata,
        confidence_thresholds=[0.25],
        match_iou=0.5,
    )
    rows = summarize_contributions(
        source_ids,
        contributions,
        repeats=20,
        confidence=0.95,
        seed=7,
    )
    by_source = {
        row["prediction_source"]: row
        for row in rows
        if row["dimension"] == "relative_bbox_area"
    }

    assert by_source["detector"]["localization_recall"] == 1.0
    assert by_source["detector"]["end_to_end_recall"] == 0.0
    assert by_source["missing"]["end_to_end_recall"] == 1.0
    assert by_source["rule-v1"]["end_to_end_recall"] == 1.0
    assert np.isfinite(
        by_source["missing"]["end_to_end_recall_ci_lower"]
    )


def test_scene_strata_reject_invalid_bootstrap_repeats():
    with pytest.raises(ValueError, match="repeats"):
        summarize_contributions(
            ["a"],
            {(0.25, "detector", "scene_density", "single_object"): np.ones(
                (1, 3),
                dtype=np.int64,
            )},
            repeats=0,
            confidence=0.95,
            seed=1,
        )
