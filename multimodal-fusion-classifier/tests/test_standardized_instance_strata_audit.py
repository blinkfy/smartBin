from __future__ import annotations

from audit_standardized_instance_strata import _paired_lookup


def test_paired_lookup_uses_full_strata_identity() -> None:
    rows = [
        {
            "candidate": "v5",
            "dimension": "scale",
            "stratum": "small",
            "metric": "mask",
            "score": "mean_recall50_95",
            "difference": 0.1,
        }
    ]

    lookup = _paired_lookup(rows)

    assert lookup[
        ("v5", "scale", "small", "mask", "mean_recall50_95")
    ]["difference"] == 0.1
