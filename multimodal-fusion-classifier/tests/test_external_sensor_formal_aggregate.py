from __future__ import annotations

from aggregate_external_sensor_formal import summarize_conditions


def test_formal_condition_summary_uses_paired_seed_differences() -> None:
    metric_names = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
    )
    rows = []
    for seed_index in range(3):
        matched = {name: 0.9 + seed_index * 0.01 for name in metric_names}
        fallback = {name: value - 0.02 for name, value in matched.items()}
        shuffled = {name: value - 0.03 for name, value in matched.items()}
        rows.append(
            {
                "metrics": {
                    "external_matched": matched,
                    "rgb_fallback": fallback,
                    "external_shuffled": shuffled,
                }
            }
        )

    summary = summarize_conditions(rows)

    fallback_delta = summary["paired_differences"][
        "matched_minus_fallback"
    ]["metrics/mAP50-95(M)"]
    shuffled_delta = summary["paired_differences"][
        "matched_minus_shuffled"
    ]["metrics/mAP50-95(M)"]
    assert abs(fallback_delta["mean"] - 0.02) < 1e-12
    assert abs(shuffled_delta["mean"] - 0.03) < 1e-12
    assert fallback_delta["count"] == 3
