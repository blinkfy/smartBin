from __future__ import annotations

from prepare_object_qwen_scene_distillation_protocol import build_protocol


def test_distillation_protocol_freezes_leakage_and_test_gates(tmp_path) -> None:
    protocol = build_protocol(tmp_path / "PROTOCOL.json")

    assert protocol["valid"] is True
    assert protocol["state"] == "frozen_before_training"
    assert protocol["formal_paper_metric"] is False
    assert protocol["data_contract"]["direct_teacher_to_joint_head"] is False
    assert protocol["selection"]["test_used_for_selection"] is False
    assert protocol["data_contract"]["test_read_time"] == (
        "only_after_best_checkpoint_is_frozen"
    )
    assert protocol["runtime"]["batch_size"] == 256
    assert protocol["runtime"]["workers"] == 4
