from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import pytest

import build_scene_sensor_sidecars
from build_scene_sensor_sidecars import build
from mmfc.scene_sensors import (
    QWEN_SCENE_PROMPT,
    QwenSceneSensorEnhancer,
    scene_record_usable,
    simulate_scene_rule_sensor,
)
from mmfc.sensors import contains_leakage_key
from validate_scene_sensor_sidecars import validate


def _image() -> Image.Image:
    image = Image.new("RGB", (16, 12), (30, 90, 150))
    return image


def test_scene_rule_sensor_is_full_image_and_annotation_free() -> None:
    first = simulate_scene_rule_sensor(_image(), "scene-a", seed=7)
    second = simulate_scene_rule_sensor(_image(), "scene-a", seed=7)

    assert first == second
    assert first["source"]["full_image_only"] is True
    assert first["source"]["annotation_inputs"] == []
    assert not contains_leakage_key(first)
    assert scene_record_usable(
        first, mode="rule", qwen_model="qwen3.5-flash"
    )


def test_qwen_scene_request_uses_whole_image_and_no_target_prompt() -> None:
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "properties": {
                                    "weight_g": 10,
                                    "temperature_c": 22,
                                    "moisture_pct": 30,
                                    "odor_index": 0.2,
                                    "conductivity_us_cm": 100,
                                },
                                "normalized_grid": [[0.4] * 4] * 4,
                            }
                        )
                    }
                }
            ]
        }

    rule = simulate_scene_rule_sensor(_image(), "scene-a")
    result = QwenSceneSensorEnhancer(
        api_key="fake", transport=transport, retries=0
    ).enhance(_image(), rule)

    assert result["source"]["enhancement_status"] == "success"
    assert result["source"]["generator"] == "qwen3.5-flash"
    content = calls[0]["messages"][0]["content"]
    assert content[0]["text"] == QWEN_SCENE_PROMPT
    assert content[1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    prompt = content[0]["text"]
    assert "目标框" in prompt and "不要" in prompt


def _args(images_root: Path, output: Path, **overrides):
    values = {
        "images_root": images_root,
        "output": output,
        "mode": "rule",
        "seed": 7,
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://example.invalid/v1",
        "model": "qwen3.5-flash",
        "retries": 0,
        "timeout": 1.0,
        "workers": 2,
        "checkpoint_every": 1,
        "progress_every": 10,
        "resume": False,
        "limit": None,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_rule_builder_and_validator_use_only_split_images(
    tmp_path: Path,
) -> None:
    images_root = tmp_path / "images"
    for index, split in enumerate(("train", "val", "test")):
        split_root = images_root / split
        split_root.mkdir(parents=True)
        Image.new(
            "RGB", (12, 10), (20 + index, 50, 80)
        ).save(split_root / f"source-{index}.png")
    output = tmp_path / "output"

    report = build(_args(images_root, output))
    checked = validate(output)

    assert report["valid"] is True
    assert report["source_image_count"] == 3
    assert report["source_contract"]["annotation_files_read"] is False
    assert checked["valid"] is True
    assert checked["split_counts"] == {"train": 1, "val": 1, "test": 1}


def test_qwen_builder_without_key_fails_before_writing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    images_root = tmp_path / "images"
    for split in ("train", "val", "test"):
        split_root = images_root / split
        split_root.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(split_root / f"{split}.png")
    output = tmp_path / "output"
    monkeypatch.setattr(
        build_scene_sensor_sidecars, "qwen_api_key", lambda _: None
    )

    with pytest.raises(RuntimeError, match="requires API credentials"):
        build(_args(images_root, output, mode="qwen"))
    assert not output.exists()
