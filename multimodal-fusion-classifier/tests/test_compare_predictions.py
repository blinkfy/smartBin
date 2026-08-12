from __future__ import annotations

import pytest

from compare_predictions import exact_mcnemar_p_value


def test_exact_mcnemar_p_value_handles_balanced_and_empty_pairs() -> None:
    assert exact_mcnemar_p_value(0, 0) == 1.0
    assert exact_mcnemar_p_value(1, 1) == 1.0
    assert exact_mcnemar_p_value(0, 5) == pytest.approx(0.0625)
    with pytest.raises(ValueError, match="non-negative"):
        exact_mcnemar_p_value(-1, 2)
