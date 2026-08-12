from __future__ import annotations

import pytest
import torch

from evaluate_reliable_external_conditions import summarize_gate


def test_gate_summary_reports_population_statistics() -> None:
    summary = summarize_gate(torch.tensor([[0.2], [0.4], [0.6]]))

    assert abs(summary["mean"] - 0.4) < 1e-7
    assert summary["minimum"] == pytest.approx(0.2)
    assert summary["maximum"] == pytest.approx(0.6)
    assert summary["population_standard_deviation"] > 0.0
