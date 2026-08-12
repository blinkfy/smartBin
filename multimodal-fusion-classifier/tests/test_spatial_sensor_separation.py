from __future__ import annotations

import pytest
import torch

from diagnose_spatial_sensor_separation import binary_auc


def test_binary_auc_handles_perfect_and_tied_separation() -> None:
    negative = torch.tensor([0.0, 0.1, 0.2])
    positive = torch.tensor([0.8, 0.9, 1.0])

    assert binary_auc(negative, positive) == pytest.approx(1.0)
    assert binary_auc(
        torch.ones(3), torch.ones(3)
    ) == pytest.approx(0.5)
