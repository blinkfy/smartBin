from __future__ import annotations

from prepare_object_qwen_scene_calibration_recovery import build


def test_calibration_recovery_preserves_raw_failure_and_test_gate(
    tmp_path,
) -> None:
    result = build(tmp_path / "CALIBRATION_RECOVERY_PROTOCOL.json")

    assert result["valid"] is True
    assert result["original_failure_preserved"] is True
    assert result["calibration"]["fit_split"] == "val"
    assert result["calibration"]["test_read_during_fit"] is False
    assert result["calibration"]["parameter_file_frozen_before_test"] is True
    assert result["formal_paper_metric"] is False
