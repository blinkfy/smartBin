from __future__ import annotations

import numpy as np

from prepare_joint_multimodal_seg import rasterize_sensor_target


def test_rasterize_full_image_constant_grid() -> None:
    signal, coverage = rasterize_sensor_target(
        [([0.0, 0.0, 100.0, 100.0], np.ones((4, 4), dtype=np.float32))],
        image_width=100,
        image_height=100,
    )

    np.testing.assert_allclose(signal, np.ones((4, 4)), atol=1e-7)
    np.testing.assert_allclose(coverage, np.ones((4, 4)), atol=1e-7)


def test_rasterize_quarter_cell_is_spatial_and_bounded() -> None:
    signal, coverage = rasterize_sensor_target(
        [([0.0, 0.0, 25.0, 25.0], np.full((4, 4), 0.8, dtype=np.float32))],
        image_width=100,
        image_height=100,
    )

    assert signal[0, 0] == np.float32(0.8)
    assert coverage[0, 0] == np.float32(1.0)
    assert np.count_nonzero(signal) == 1
    assert np.count_nonzero(coverage) == 1
    assert 0.0 <= float(signal.min()) <= float(signal.max()) <= 1.0


def test_rasterize_partial_coverage_scales_signal() -> None:
    signal, coverage = rasterize_sensor_target(
        [([0.0, 0.0, 12.5, 25.0], np.ones((4, 4), dtype=np.float32))],
        image_width=100,
        image_height=100,
    )

    assert coverage[0, 0] == np.float32(0.5)
    assert signal[0, 0] == np.float32(0.5)
