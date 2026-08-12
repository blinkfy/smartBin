from __future__ import annotations

from audit_spatial_reliability_extension import (
    select_development_candidate,
)


def _report(clean_box, clean_mask, shuffled_box, shuffled_mask):
    return {
        "conditions": {
            "clean_matched": {
                "metrics": {
                    "metrics/mAP50-95(B)": clean_box,
                    "metrics/mAP50-95(M)": clean_mask,
                }
            },
            "shuffled": {
                "metrics": {
                    "metrics/mAP50-95(B)": shuffled_box,
                    "metrics/mAP50-95(M)": shuffled_mask,
                }
            },
        }
    }


def test_selection_rejects_overcalibrated_candidate() -> None:
    primary = _report(0.97, 0.95, 0.969, 0.949)
    calibrated = _report(0.96, 0.94, 0.961, 0.941)

    selection = select_development_candidate(primary, calibrated)

    assert selection["selected"] == "spatial_v5_primary"
    assert selection["calibrated_rejected"] is True
