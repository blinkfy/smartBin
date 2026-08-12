from __future__ import annotations

import numpy as np
import pytest

from evaluate_standardized_instance_strata import (
    assign_coco_visible_scale,
    assign_overlap_proxy,
    box_iou_matrix,
    greedy_class_aware_matches,
    mask_iou_matrix,
    summarize_contributions,
)


def test_coco_visible_scale_uses_half_open_standard_thresholds() -> None:
    assert assign_coco_visible_scale(0) == "small"
    assert assign_coco_visible_scale(1023) == "small"
    assert assign_coco_visible_scale(1024) == "medium"
    assert assign_coco_visible_scale(9215) == "medium"
    assert assign_coco_visible_scale(9216) == "large"


def test_overlap_proxy_separates_isolated_low_and_high() -> None:
    assert assign_overlap_proxy(0.0) == "isolated"
    assert assign_overlap_proxy(0.01) == "overlap_low"
    assert assign_overlap_proxy(0.249) == "overlap_low"
    assert assign_overlap_proxy(0.25) == "overlap_high"


def test_box_and_mask_iou_matrices_are_target_by_prediction() -> None:
    boxes = box_iou_matrix(
        [[0, 0, 10, 10]],
        [[0, 0, 10, 10], [10, 10, 20, 20]],
    )
    assert boxes.shape == (1, 2)
    assert boxes.tolist() == [[1.0, 0.0]]

    target = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)
    masks = mask_iou_matrix([target], [prediction])
    assert masks.shape == (1, 1)
    assert masks[0, 0] == pytest.approx(1.0 / 3.0)


def test_greedy_matching_is_class_aware_and_one_to_one() -> None:
    iou = np.array([[0.9, 0.8], [0.85, 0.7]], dtype=float)

    matched = greedy_class_aware_matches(
        iou,
        target_classes=[1, 1],
        prediction_classes=[1, 1],
        threshold=0.5,
    )
    wrong_class = greedy_class_aware_matches(
        iou,
        target_classes=[1, 1],
        prediction_classes=[2, 2],
        threshold=0.5,
    )

    assert matched.tolist() == [True, True]
    assert wrong_class.tolist() == [False, False]


def test_summary_reports_cluster_bootstrap_recall() -> None:
    contributions = {
        ("coco_visible_mask_area", "small", "mask"): np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    }
    rng = np.random.default_rng(7)
    weights = rng.multinomial(
        2,
        np.array([0.5, 0.5]),
        size=100,
    )

    rows, samples = summarize_contributions(
        "model",
        contributions,
        bootstrap_weights=weights,
        confidence=0.95,
    )

    assert len(rows) == 1
    assert rows[0]["support"] == 2
    assert rows[0]["recall50"] == 1.0
    assert rows[0]["recall75"] == 0.5
    assert 0.0 <= rows[0]["mean_recall50_95"] <= 1.0
    assert samples[
        ("coco_visible_mask_area", "small", "mask")
    ]["recall50"].shape == (100,)
