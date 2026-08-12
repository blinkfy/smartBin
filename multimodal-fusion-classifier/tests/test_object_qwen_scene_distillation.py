from __future__ import annotations

import numpy as np

from audit_object_qwen_scene_distillation_feasibility import audit
from build_object_qwen_scene_teacher import aggregate_object_grids


def test_direct_bbox_aggregate_is_explicitly_prohibited(tmp_path) -> None:
    result = audit(
        tmp_path / "OBJECT_QWEN_SCENE_DISTILLATION_FEASIBILITY.json"
    )

    assert result["valid"] is True
    assert result["decision"]["direct_bbox_aggregate_as_detector_input"] == (
        "prohibited"
    )
    assert result["decision"]["final_sidecar_inference_reads"] == [
        "full_rgb_image"
    ]
    assert result["claim_boundary"]["development_evidence_only"] is True


def test_full_image_bbox_preserves_object_local_grid() -> None:
    grid = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0

    teacher, coverage = aggregate_object_grids(
        [{"bbox": [0, 0, 480, 480], "normalized_grid": grid.tolist()}],
        image_width=480,
        image_height=480,
    )

    np.testing.assert_allclose(teacher, grid, rtol=0, atol=1e-7)
    np.testing.assert_allclose(coverage, np.ones((4, 4)), rtol=0, atol=1e-7)


def test_small_bbox_is_spatial_and_coverage_attenuated() -> None:
    teacher, coverage = aggregate_object_grids(
        [
            {
                "bbox": [0, 0, 120, 120],
                "normalized_grid": np.ones((4, 4)).tolist(),
            }
        ],
        image_width=480,
        image_height=480,
    )

    assert teacher[0, 0] == 1.0
    assert coverage[0, 0] == 1.0
    assert np.count_nonzero(teacher) == 1
    assert np.count_nonzero(coverage) == 1
