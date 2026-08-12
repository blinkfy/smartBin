from __future__ import annotations

import torch

from audit_external_sensor_worker_equivalence import (
    compare_state_dicts,
    normalized_train_args,
)


def test_worker_equivalence_helpers_require_exact_tensors() -> None:
    baseline = {
        "weight": torch.tensor([[1.0, 2.0]]),
        "count": torch.tensor(3),
    }
    candidate = {name: value.clone() for name, value in baseline.items()}

    assert compare_state_dicts(baseline, candidate)["exact"] is True

    candidate["weight"][0, 1] = 2.5
    comparison = compare_state_dicts(baseline, candidate)
    assert comparison["exact"] is False
    assert comparison["different_tensor_count"] == 1


def test_worker_normalization_changes_only_runtime_fields() -> None:
    baseline = {"name": "baseline", "workers": 0, "batch": 32, "seed": 42}
    candidate = {"name": "candidate", "workers": 4, "batch": 32, "seed": 42}

    assert normalized_train_args(baseline) == normalized_train_args(candidate)
