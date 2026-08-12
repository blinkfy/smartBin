from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from mmfc.joint_seg_external import (
    load_scene_sensor_index,
    pack_scene_sensor_channel,
    split_scene_sensor_channel,
)


def test_sensor_transport_channel_round_trips_4x4_grid() -> None:
    images = torch.randint(
        0, 256, (2, 3, 32, 48), dtype=torch.uint8
    )
    grids = torch.linspace(0.0, 1.0, 32).reshape(2, 1, 4, 4)

    packed = pack_scene_sensor_channel(images, grids)
    rgb, recovered = split_scene_sensor_channel(packed.float() / 255.0)

    assert packed.shape == (2, 4, 32, 48)
    torch.testing.assert_close(rgb, images.float() / 255.0)
    torch.testing.assert_close(
        recovered,
        (grids * 255.0).round() / 255.0,
        atol=1e-6,
        rtol=0.0,
    )


def test_sensor_transport_allows_rgb_fallback() -> None:
    rgb = torch.rand(1, 3, 16, 16)
    recovered_rgb, sensor = split_scene_sensor_channel(rgb)

    torch.testing.assert_close(recovered_rgb, rgb)
    assert sensor is None


def test_scene_sensor_index_rejects_annotation_conditioned_record(
    tmp_path,
) -> None:
    path = tmp_path / "observations.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_image_id": "source-a",
                "normalized_grid": [[0.2] * 4] * 4,
                "source": {
                    "observation_scope": "full_image",
                    "full_image_only": True,
                    "annotation_inputs": ["polygon"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid scene sensor record"):
        load_scene_sensor_index(path)
