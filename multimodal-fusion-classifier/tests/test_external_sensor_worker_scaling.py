from audit_external_sensor_worker_scaling_4_to_8 import (
    select_worker_count,
)


def test_worker_scaling_keeps_faster_exact_candidate() -> None:
    assert select_worker_count(30.0, 40.0, exact=True) == 4
    assert select_worker_count(30.0, 20.0, exact=True) == 8


def test_worker_scaling_rejects_non_equivalent_candidate() -> None:
    assert select_worker_count(30.0, 20.0, exact=False) == 4
