from __future__ import annotations

import argparse
import json

from train_external_sensor_joint_seg import _protocol


def test_external_sensor_protocol_separates_rgb_and_sensor_inputs(
    tmp_path,
) -> None:
    names = (
        "dataset.yaml",
        "base.pt",
        "observations.jsonl",
        "VALIDATION_REPORT.json",
        "BUILD_REPORT.json",
        "text.npy",
    )
    paths = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"x")
    paths.update(
        {
            "data": tmp_path / "dataset.yaml",
            "base_weights": tmp_path / "base.pt",
            "sensor_observations": tmp_path / "observations.jsonl",
            "sensor_validation": tmp_path / "VALIDATION_REPORT.json",
            "sensor_build": tmp_path / "BUILD_REPORT.json",
            "text_prototypes": tmp_path / "text.npy",
        }
    )
    paths["sensor_build"].write_text(
        json.dumps(
            {
                "mode": "rule",
                "model": None,
                "source_image_count": 5990,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        name="smoke",
        seed=42,
        semantic_dim=64,
        sensor_hidden_dim=256,
        text_temperature=0.1,
        sensor_reconstruction_weight=0.25,
        sensor_calibration_weight=0.05,
    )

    protocol = _protocol(args, paths, {"batch": 32})

    assert protocol["architecture"]["image_backbone_inputs"] == ["RGB"]
    assert protocol["leakage_control"]["metric_forward_inputs"] == [
        "RGB",
        "full_image_4x4_soft_sensor",
    ]
    source = protocol["sensor_source"]
    assert source["annotation_files_read"] is False
    assert source["bbox_inputs"] is False
    assert source["polygon_inputs"] is False
