import pytest

from assess_class_balance_candidate import assess_candidate


def _metrics(accuracy: float, macro_f1: float) -> dict[str, float]:
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def test_class_balance_candidate_passes_locked_continuation_rule():
    report = assess_candidate(
        reference_val=_metrics(0.90, 0.82),
        candidate_val=_metrics(0.89, 0.83),
        reference_test=_metrics(0.88, 0.81),
        candidate_test=_metrics(0.871, 0.815),
        reference_name="v3",
        candidate_name="balanced",
    )

    assert report["passed"] is True
    assert report["decision"] == "extend_training_seeds"
    assert all(item["passed"] for item in report["criteria"].values())


@pytest.mark.parametrize(
    ("candidate_val_f1", "candidate_test_accuracy", "candidate_test_f1"),
    [
        (0.819, 0.88, 0.82),
        (0.83, 0.869, 0.82),
        (0.83, 0.88, 0.809),
    ],
)
def test_class_balance_candidate_stops_when_any_locked_condition_fails(
    candidate_val_f1: float,
    candidate_test_accuracy: float,
    candidate_test_f1: float,
):
    report = assess_candidate(
        reference_val=_metrics(0.90, 0.82),
        candidate_val=_metrics(0.90, candidate_val_f1),
        reference_test=_metrics(0.88, 0.81),
        candidate_test=_metrics(candidate_test_accuracy, candidate_test_f1),
        reference_name="v3",
        candidate_name="balanced",
    )

    assert report["passed"] is False
    assert report["decision"] == "stop_negative_result"


def test_class_balance_accuracy_boundary_is_inclusive():
    report = assess_candidate(
        reference_val=_metrics(0.90, 0.82),
        candidate_val=_metrics(0.90, 0.82),
        reference_test=_metrics(0.88, 0.81),
        candidate_test=_metrics(0.87, 0.81),
        reference_name="v3",
        candidate_name="balanced",
    )

    assert report["passed"] is True


def test_class_balance_rejects_invalid_accuracy_drop_limit():
    with pytest.raises(ValueError, match="max_test_accuracy_drop"):
        assess_candidate(
            reference_val=_metrics(0.90, 0.82),
            candidate_val=_metrics(0.90, 0.82),
            reference_test=_metrics(0.88, 0.81),
            candidate_test=_metrics(0.88, 0.81),
            reference_name="v3",
            candidate_name="balanced",
            max_test_accuracy_drop=-0.01,
        )
