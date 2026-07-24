from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from build_dataset import resume_record_usable
from mmfc.sensors import (
    QWEN_PROMPT,
    QwenSensorEnhancer,
    contains_leakage_key,
    simulate_rule_sensor,
)
from validate_dataset import validate_generated


NUMERIC_LEAKAGE_KEYS = (
    "label_id",
    "class_id",
    "target_id",
    "category_id",
    "label_index",
    "class_index",
    "target_index",
    "category_index",
)


def _image() -> Image.Image:
    image = Image.new("RGB", (32, 32))
    pixels = image.load()
    for y in range(32):
        for x in range(32):
            pixels[x, y] = (x * 7, y * 7, (x + y) * 3)
    return image


def test_rule_simulator_is_deterministic_and_label_independent() -> None:
    # The simulator API has no label argument; identical image/bbox/seed inputs
    # therefore remain unchanged when an external manifest label changes.
    first = simulate_rule_sensor(_image(), [2, 3, 27, 29], "opaque-id", seed=9)
    second = simulate_rule_sensor(_image(), [2, 3, 27, 29], "opaque-id", seed=9)
    assert first == second
    assert "label" not in json.dumps(first).casefold()
    assert first["grid_shape"] == [4, 4]
    assert all(
        0 <= value <= 1 for row in first["normalized_grid"] for value in row
    )


def test_qwen_uses_image_only_json_request_and_clamps_values() -> None:
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append((url, headers, payload, timeout))
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "properties": {
                                    "weight_g": 99999,
                                    "temperature_c": 25,
                                    "moisture_pct": -4,
                                    "odor_index": 0.2,
                                    "conductivity_us_cm": 300,
                                },
                                "normalized_grid": [[1.5] * 4] * 4,
                            }
                        )
                    }
                }
            ]
        }

    rule = simulate_rule_sensor(_image(), [2, 3, 27, 29], "sample")
    result = QwenSensorEnhancer(
        api_key="test-key", transport=transport, retries=0
    ).enhance(_image(), [2, 3, 27, 29], rule)

    assert result["source"]["generator"] == "qwen3-vl-flash"
    assert result["properties"]["weight_g"] == 5000.0
    assert result["properties"]["moisture_pct"] == 0.0
    assert result["normalized_grid"] == [[1.0] * 4] * 4
    payload = calls[0][2]
    assert payload["response_format"] == {"type": "json_object"}
    content = payload["messages"][0]["content"]
    assert content[0]["text"] == QWEN_PROMPT
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "target" not in json.dumps(payload).casefold()


def test_qwen_without_key_never_calls_api_and_invalid_json_falls_back() -> None:
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": "not-json"}}]}

    rule = simulate_rule_sensor(_image(), [2, 3, 27, 29], "sample")
    no_key = QwenSensorEnhancer(api_key=None, transport=transport).enhance(
        _image(), [2, 3, 27, 29], rule
    )
    assert calls == []
    assert no_key["source"]["enhancement_status"] == "fallback_no_api_key"

    invalid = QwenSensorEnhancer(
        api_key="test-key", transport=transport, retries=1
    ).enhance(_image(), [2, 3, 27, 29], rule)
    assert len(calls) == 2
    assert invalid["source"]["enhancement_status"] == "fallback_error"
    assert invalid["properties"] == rule["properties"]


def test_resume_is_mode_aware_and_retries_qwen_fallbacks() -> None:
    rule = simulate_rule_sensor(_image(), [2, 3, 27, 29], "sample")
    assert resume_record_usable(
        rule, sensor_mode="rule", qwen_model="qwen3-vl-flash"
    )
    assert not resume_record_usable(
        rule, sensor_mode="qwen", qwen_model="qwen3-vl-flash"
    )

    qwen_success = {
        **rule,
        "source": {
            **rule["source"],
            "generator": "custom-qwen-model",
            "enhancement_status": "success",
        },
    }
    assert resume_record_usable(
        qwen_success, sensor_mode="qwen", qwen_model="custom-qwen-model"
    )
    assert not resume_record_usable(
        qwen_success, sensor_mode="qwen", qwen_model="qwen3-vl-flash"
    )

    for status in ("fallback_no_api_key", "fallback_error"):
        fallback = {
            **rule,
            "source": {
                **rule["source"],
                "generator": "rule-v1",
                "enhancement_status": status,
            },
        }
        assert not resume_record_usable(
            fallback, sensor_mode="qwen", qwen_model="qwen3-vl-flash"
        )


def test_resume_rejects_malformed_sensor_records() -> None:
    rule = simulate_rule_sensor(_image(), [2, 3, 27, 29], "sample")
    malformed = {**rule, "normalized_grid": [[float("nan")] * 4] * 4}
    assert not resume_record_usable(
        malformed, sensor_mode="rule", qwen_model="qwen3-vl-flash"
    )


@pytest.mark.parametrize(
    "leakage_key",
    [
        "label",
        "labels",
        "class",
        "classes",
        "target",
        "targets",
        "target_label",
        "class_name",
        "category",
        "category_id",
        "label_id",
        "class_id",
        "target_id",
        "label_index",
        "class_index",
        "target_index",
        "category_index",
    ],
)
def test_resume_rejects_recursive_case_insensitive_leakage_in_all_modes(
    leakage_key,
) -> None:
    rule = simulate_rule_sensor(_image(), [2, 3, 27, 29], "sample")
    qwen = {
        **rule,
        "source": {
            **rule["source"],
            "generator": "qwen3-vl-flash",
            "enhancement_status": "success",
        },
    }
    assert resume_record_usable(
        rule, sensor_mode="rule", qwen_model="qwen3-vl-flash"
    )
    assert resume_record_usable(
        qwen, sensor_mode="qwen", qwen_model="qwen3-vl-flash"
    )
    assert not contains_leakage_key(rule)
    assert not contains_leakage_key(qwen)

    nested_leak = [{"metadata": {leakage_key.upper(): 2}}]
    leaked_rule = {**rule, "legacy": nested_leak}
    leaked_qwen = {**qwen, "legacy": nested_leak}
    assert contains_leakage_key(leaked_rule)
    assert not resume_record_usable(
        leaked_rule, sensor_mode="rule", qwen_model="qwen3-vl-flash"
    )
    assert not resume_record_usable(
        leaked_qwen, sensor_mode="qwen", qwen_model="qwen3-vl-flash"
    )


def _write_validator_fixture(
    tmp_path: Path, sensor_record: dict
) -> tuple[Path, Path]:
    output = tmp_path / "generated"
    output.mkdir()
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), (40, 80, 120)).save(image_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"classes": [{"name": "clean"}]}),
        encoding="utf-8",
    )
    sample = {
        "schema_version": 1,
        "sample_id": "validator-sample",
        "source_image_id": "validator-image",
        "image_path": str(image_path),
        "image_width": 8,
        "image_height": 8,
        "bbox": [0, 0, 8, 8],
        "label": "clean",
        "label_id": 0,
        "split": "train",
    }
    (output / "train.jsonl").write_text(
        json.dumps(sample) + "\n", encoding="utf-8"
    )
    for split in ("val", "test"):
        (output / f"{split}.jsonl").write_text("", encoding="utf-8")
    (output / "sensors.jsonl").write_text(
        json.dumps(sensor_record) + "\n", encoding="utf-8"
    )
    (output / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "labels": ["clean"],
                "label_to_index": {"clean": 0},
                "samples": {
                    "validator-sample": {
                        "label_index": 0,
                        "split": "train",
                        "sensor_index": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (output / "stats.json").write_text("{}", encoding="utf-8")
    return output, config_path


@pytest.mark.parametrize("leakage_key", NUMERIC_LEAKAGE_KEYS)
def test_validator_reuses_recursive_detector_for_numeric_aliases(
    tmp_path: Path, leakage_key: str
) -> None:
    clean = simulate_rule_sensor(
        _image(), [2, 3, 27, 29], "validator-sample"
    )
    output, config_path = _write_validator_fixture(tmp_path, clean)
    clean_report = validate_generated(
        output, project_root=tmp_path, config_path=config_path
    )
    assert clean_report["valid"], clean_report["errors"]

    leaked = {
        **clean,
        "legacy": [{"metadata": {leakage_key.upper(): 2}}],
    }
    (output / "sensors.jsonl").write_text(
        json.dumps(leaked) + "\n", encoding="utf-8"
    )
    leaked_report = validate_generated(
        output, project_root=tmp_path, config_path=config_path
    )
    assert not leaked_report["valid"]
    assert any(
        "传感器记录包含目标标签/类别字段" in error
        for error in leaked_report["errors"]
    )
