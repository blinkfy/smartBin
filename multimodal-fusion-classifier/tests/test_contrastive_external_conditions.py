from __future__ import annotations

import pytest

from evaluate_contrastive_external_conditions import (
    PROTOCOL_NAME,
    validate_run_bindings,
)


def test_contrastive_condition_binding_validation() -> None:
    protocol = {
        "protocol": PROTOCOL_NAME,
        "formal_paper_metric": False,
    }
    evaluation = {
        "valid": True,
        "formal_paper_metric": False,
        "checkpoint": {"sha256": "checkpoint"},
        "sensor_observations": {"sha256": "observations"},
    }

    validate_run_bindings(
        protocol,
        evaluation,
        checkpoint_sha256="checkpoint",
        observations_sha256="observations",
    )

    with pytest.raises(ValueError, match="bindings"):
        validate_run_bindings(
            protocol,
            evaluation,
            checkpoint_sha256="different",
            observations_sha256="observations",
        )
