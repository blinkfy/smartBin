from __future__ import annotations

from audit_external_sensor_joint_seg import metric_deltas


def test_external_sensor_metric_deltas_are_same_checkpoint_diagnostics() -> None:
    keys = (
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95(M)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
    )
    fallback = {key: 0.8 for key in keys}
    external = {key: 0.81 for key in keys}

    deltas = metric_deltas(external, fallback)

    assert set(deltas) == set(keys)
    assert all(abs(value - 0.01) < 1e-12 for value in deltas.values())
