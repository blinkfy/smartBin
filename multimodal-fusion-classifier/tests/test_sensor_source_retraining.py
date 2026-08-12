import hashlib
import json

import prepare_sensor_source_retraining
from prepare_sensor_source_retraining import (
    _config_text,
    _text_sha256,
    prepare_config,
    resolve_rule_v1_config,
    validate_sensor_alignment,
)


def test_rule_v1_config_changes_only_source_output_and_experiment():
    base = {
        "model": {"feature_dim": 512},
        "loss": {"ce_weight": 1.0},
        "data": {
            "train_manifest": "train.jsonl",
            "val_manifest": "val.jsonl",
            "test_manifest": "test.jsonl",
            "sensor_manifest": "qwen.jsonl",
        },
        "training": {
            "seed": 42,
            "epochs": 100,
            "checkpoint_dir": "qwen-checkpoint",
        },
        "experiment": {"name": "qwen"},
    }

    candidate = resolve_rule_v1_config(base)

    assert candidate["model"] == base["model"]
    assert candidate["loss"] == base["loss"]
    assert candidate["training"]["seed"] == 42
    assert candidate["training"]["epochs"] == 100
    assert candidate["data"]["train_manifest"] == "train.jsonl"
    assert candidate["data"]["sensor_manifest"] == (
        "data/generated/sensors.jsonl"
    )
    assert "source-transfer" in candidate["training"]["checkpoint_dir"]
    assert candidate["experiment"]["candidate_sensor_source"] == "rule-v1"
    assert base["data"]["sensor_manifest"] == "qwen.jsonl"


def test_sensor_alignment_requires_equal_complete_disjoint_ids(
    tmp_path,
):
    qwen = tmp_path / "qwen.jsonl"
    rule = tmp_path / "rule.jsonl"
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"
    qwen.write_text(
        '{"sample_id":"a"}\n{"sample_id":"b"}\n{"sample_id":"c"}\n',
        encoding="utf-8",
    )
    rule.write_text(
        '{"sample_id":"c"}\n{"sample_id":"a"}\n{"sample_id":"b"}\n',
        encoding="utf-8",
    )
    train.write_text('{"sample_id":"a"}\n', encoding="utf-8")
    val.write_text('{"sample_id":"b"}\n', encoding="utf-8")
    test.write_text('{"sample_id":"c"}\n', encoding="utf-8")

    result = validate_sensor_alignment(
        qwen_sensor_path=qwen,
        rule_sensor_path=rule,
        split_paths={"train": train, "val": val, "test": test},
    )

    assert result["common_sample_count"] == 3
    assert result["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert result["all_records_covered"] is True


def test_prepare_config_normalizes_equivalent_crlf_bytes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        prepare_sensor_source_retraining,
        "PROJECT_ROOT",
        tmp_path,
    )
    base = {
        "model": {"feature_dim": 512},
        "loss": {"ce_weight": 1.0},
        "data": {
            "train_manifest": "train.jsonl",
            "val_manifest": "val.jsonl",
            "test_manifest": "test.jsonl",
            "sensor_manifest": "qwen.jsonl",
        },
        "training": {
            "seed": 42,
            "epochs": 100,
            "checkpoint_dir": "qwen-checkpoint",
        },
        "experiment": {"name": "qwen"},
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    for name, sample_id in (
        ("qwen.jsonl", "a"),
        ("rule.jsonl", "a"),
        ("train.jsonl", "a"),
    ):
        (tmp_path / name).write_text(
            json.dumps({"sample_id": sample_id}) + "\n",
            encoding="utf-8",
        )
    (tmp_path / "val.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "test.jsonl").write_text("", encoding="utf-8")
    resolved_text = _config_text(resolve_rule_v1_config(base))
    expected_alignment = {
        "qwen_sensor_count": 1,
        "rule_v1_sensor_count": 1,
        "common_sample_count": 1,
        "split_counts": {"train": 1, "val": 0, "test": 0},
        "sample_id_sets_equal": True,
        "split_ids_disjoint": True,
        "all_records_covered": True,
    }
    protocol = {
        "status": "frozen_before_rule_v1_retraining",
        "base_config": {"path": "base.json"},
        "resolved_config_sha256": _text_sha256(resolved_text),
        "sensor_alignment": {
            "paths": {
                "qwen": "qwen.jsonl",
                "rule-v1": "rule.jsonl",
                "splits": {
                    "train": "train.jsonl",
                    "val": "val.jsonl",
                    "test": "test.jsonl",
                },
            },
            "expected": expected_alignment,
        },
        "input_hashes": [
            {
                "path": "base.json",
                "sha256": hashlib.sha256(
                    base_path.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    output_path = tmp_path / "resolved.json"
    output_path.write_bytes(
        resolved_text.replace("\n", "\r\n").encode("utf-8")
    )

    result = prepare_config(
        protocol_path=protocol_path,
        output_path=output_path,
    )

    assert result["sensor_alignment"] == expected_alignment
    assert output_path.read_bytes() == resolved_text.encode("utf-8")
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == (
        protocol["resolved_config_sha256"]
    )
