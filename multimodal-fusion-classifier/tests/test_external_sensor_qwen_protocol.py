from __future__ import annotations

import json

from prepare_external_sensor_qwen_protocol import build_protocol


def test_qwen_external_protocol_forbids_annotation_projected_grid(
    tmp_path,
) -> None:
    output = tmp_path / "QWEN_FORMAL_PROTOCOL.json"
    protocol = build_protocol(output)

    assert protocol["status"] == (
        "preregistered_awaiting_api_credentials_and_data"
    )
    assert protocol["training"]["seeds"] == [42, 43, 44]
    assert protocol["training"]["workers"] == 4
    assert protocol["training"]["batch_size"] == 32
    assert protocol["data_generation"]["minimum_qwen_coverage"] == 0.99
    assert protocol["current_gate"]["rule_smoke_is_formal_evidence"] is False
    forbidden = "\n".join(protocol["forbidden_inputs"])
    assert "annotation-projected grids" in forbidden
    serialized = json.dumps(protocol)
    assert "sk-" not in serialized
    assert (
        protocol["data_generation"]["api_key_value_persisted"] is False
    )
    assert "condition_evaluator" in protocol["source_bindings"]
    assert "protocol_builder" in protocol["source_bindings"]
    assert "worker_equivalence_report" in protocol["source_bindings"]
    assert "formal_runner" in protocol["source_bindings"]
    assert "formal_aggregator" in protocol["source_bindings"]
    assert protocol["fusion"]["sensor_reconstruction_weight"] == 0.25
    assert "evaluate_external_sensor_conditions.py" in (
        protocol["execution_commands"]["shuffled_control_template"]
    )
