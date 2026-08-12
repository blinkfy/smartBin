from __future__ import annotations

from run_external_sensor_qwen_formal import (
    build_command_plan,
    validate_protocol,
)


def test_formal_runner_plan_uses_all_frozen_seeds() -> None:
    protocol = {
        "protocol": "full_image_qwen_external_sensor_joint_seg_v2",
        "status": "preregistered_awaiting_api_credentials_and_data",
        "data_generation": {
            "mode": "qwen",
            "expected_source_image_count": 5990,
            "api_key_environment_variable_name": "DASHSCOPE_API_KEY",
            "api_key_value_persisted": False,
            "output": "data/scene-qwen-sensor-sidecars",
            "workers": 2,
            "retries": 5,
            "timeout_seconds": 90,
            "minimum_qwen_coverage": 0.99,
            "required_valid_record_fraction": 1.0,
        },
        "training": {
            "seeds": [42, 43, 44],
            "epochs": 20,
            "patience": 5,
            "batch_size": 32,
            "image_size": 640,
            "workers": 4,
            "freeze_layers": 24,
            "learning_rate": 0.0001,
            "weight_decay": 0.0005,
        },
        "fusion": {
            "semantic_dim": 64,
            "sensor_hidden_dim": 256,
            "text_temperature": 0.1,
            "sensor_reconstruction_weight": 0.25,
            "sensor_calibration_weight": 0.05,
        },
        "evaluation": {
            "required_conditions": [
                "external Qwen full-image sidecar",
                "same-checkpoint RGB predicted-sensor fallback",
                "same-checkpoint shuffled-sidecar negative control",
            ]
        },
        "source_bindings": {"placeholder": {}},
    }
    plan = build_command_plan(protocol, python_executable="python", device="0")

    assert [run["seed"] for run in plan["runs"]] == [42, 43, 44]
    assert all("--workers" in run["train"] for run in plan["runs"])
    assert all("4" in run["train"] for run in plan["runs"])
    assert all(
        "evaluate_external_sensor_conditions.py" in run["condition"]
        for run in plan["runs"]
    )
    assert plan["aggregate"][-1].endswith("QWEN_FORMAL_PROTOCOL.json")

    errors = validate_protocol(protocol)
    assert errors == ["source binding is missing: placeholder"]
