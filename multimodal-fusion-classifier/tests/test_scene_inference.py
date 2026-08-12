from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from mmfc.scene_inference import (
    crop_scene_objects,
    detection_sample_id,
    extract_ultralytics_detections,
    load_detection_sidecar,
    prepare_scene_sensors,
)
from predict import parse_args as parse_single_prediction_args


def test_extract_ultralytics_detections_preserves_optional_masks() -> None:
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[1.0, 2.0, 8.0, 9.0]]),
            conf=torch.tensor([0.91]),
            cls=torch.tensor([3.0]),
        ),
        names={3: "object"},
        masks=SimpleNamespace(
            xy=[np.asarray([[1.0, 2.0], [8.0, 2.0], [8.0, 9.0]])]
        ),
    )
    detections = extract_ultralytics_detections(result, include_masks=True)
    assert detections[0]["bbox"] == [1.0, 2.0, 8.0, 9.0]
    assert detections[0]["detector_label"] == "object"
    assert detections[0]["detector_confidence"] == pytest.approx(0.91)
    assert len(detections[0]["mask_polygon"]) == 3


def test_sidecar_is_reordered_by_detection_index_and_validated() -> None:
    zero = [[0.0] * 4 for _ in range(4)]
    one = [[1.0] * 4 for _ in range(4)]
    grids = load_detection_sidecar(
        {
            "detections": [
                {"detection_index": 1, "normalized_grid": one},
                {"detection_index": 0, "normalized_grid": zero},
            ]
        },
        2,
    )
    assert grids == [zero, one]
    with pytest.raises(ValueError, match="count"):
        load_detection_sidecar([zero], 2)
    with pytest.raises(ValueError, match="class or target"):
        load_detection_sidecar(
            [{"normalized_grid": zero, "target": "forbidden"}],
            1,
        )


def test_scene_sensor_sources_and_crops_are_batched() -> None:
    image = Image.new("RGB", (20, 12), (120, 80, 40))
    detections = [
        {"bbox": [0.0, 0.0, 8.0, 10.0]},
        {"bbox": [10.0, 2.0, 20.0, 12.0]},
    ]
    missing, unavailable, provenance = prepare_scene_sensors(
        image,
        detections,
        source="missing",
        image_identity="scene.jpg",
    )
    assert missing.shape == (2, 1, 4, 4)
    assert torch.count_nonzero(missing) == 0
    assert unavailable.tolist() == [[0.0], [0.0]]
    assert provenance[0]["generator"] == "missing"

    simulated, available, provenance = prepare_scene_sensors(
        image,
        detections,
        source="rule-v1",
        image_identity="scene.jpg",
    )
    assert simulated.shape == (2, 1, 4, 4)
    assert torch.all((simulated >= 0.0) & (simulated <= 1.0))
    assert available.tolist() == [[1.0], [1.0]]
    assert provenance[0]["generator"] == "rule-v1"

    crops = crop_scene_objects(
        image,
        detections,
        image_size=16,
        preprocess={"enabled": False},
    )
    assert crops.shape == (2, 3, 16, 16)
    assert torch.all((crops >= 0.0) & (crops <= 1.0))


def test_detection_sample_id_is_stable_and_box_sensitive() -> None:
    first = detection_sample_id("scene.jpg", 0, [1, 2, 3, 4])
    assert first == detection_sample_id("scene.jpg", 0, [1, 2, 3, 4])
    assert first != detection_sample_id("scene.jpg", 0, [1, 2, 3, 5])


def test_single_prediction_defaults_to_locked_v3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--image",
            "sample.jpg",
            "--bbox",
            "0",
            "0",
            "10",
            "10",
            "--sensor-missing",
        ],
    )
    args = parse_single_prediction_args()
    assert args.checkpoint.as_posix().endswith(
        "checkpoints/ablations/full_cross_modal_quality/best.pt"
    )
    assert args.config.as_posix().endswith(
        "configs/ablations/full_cross_modal_quality.json"
    )
