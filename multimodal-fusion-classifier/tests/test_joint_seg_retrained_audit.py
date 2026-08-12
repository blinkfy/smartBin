from __future__ import annotations

import pytest

from audit_joint_seg_retrained_ablations import _summary


def test_retrained_ablation_summary_uses_sample_standard_deviation() -> None:
    result = _summary([0.6, 0.7, 0.8])

    assert result["mean"] == pytest.approx(0.7)
    assert result["sample_std"] == pytest.approx(0.1)
    assert result["min"] == pytest.approx(0.6)
    assert result["max"] == pytest.approx(0.8)


def test_retrained_ablation_summary_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="empty metric"):
        _summary([])
