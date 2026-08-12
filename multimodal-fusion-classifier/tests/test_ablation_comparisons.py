from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmfc.ablations import (
    holm_adjust_p_values,
    summarize_ablation_comparisons,
    write_ablation_comparison_artifacts,
)


def _comparison(
    candidate: str,
    *,
    accuracy_delta: float,
    macro_f1_delta: float,
    p_value: float,
) -> dict:
    return {
        "reference": {"name": "full"},
        "candidate": {"name": candidate},
        "outcome_changes": {
            "candidate_only_correct": 5,
            "reference_only_correct": 9,
        },
        "mcnemar_exact": {"p_value": p_value},
        "paired_bootstrap": {
            "differences": {
                "accuracy": {
                    "point": accuracy_delta,
                    "lower": accuracy_delta - 0.01,
                    "upper": accuracy_delta + 0.01,
                },
                "macro_f1": {
                    "point": macro_f1_delta,
                    "lower": macro_f1_delta - 0.02,
                    "upper": macro_f1_delta + 0.02,
                },
            }
        },
    }


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust_p_values([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_ablation_comparison_artifacts_include_intervals_and_holm(
    tmp_path: Path,
) -> None:
    comparisons = {
        "vision_only": _comparison(
            "vision_only",
            accuracy_delta=-0.1,
            macro_f1_delta=-0.15,
            p_value=0.01,
        ),
        "vision_text": _comparison(
            "vision_text",
            accuracy_delta=-0.05,
            macro_f1_delta=-0.07,
            p_value=0.2,
        ),
    }
    rows = summarize_ablation_comparisons(comparisons)
    assert rows[0]["accuracy_delta_pp"] == pytest.approx(-10.0)
    assert rows[0]["macro_f1_ci_lower_pp"] == pytest.approx(-17.0)
    assert rows[0]["mcnemar_holm_p_value"] == pytest.approx(0.02)

    artifacts = write_ablation_comparison_artifacts(
        comparisons,
        tmp_path,
    )
    assert all(Path(path).is_file() for path in artifacts.values())
    document = json.loads(
        (tmp_path / "ablation_pairwise_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["multiple_comparison_correction"] == "Holm"
    markdown = (tmp_path / "ABLATION_STATISTICS.md").read_text(
        encoding="utf-8"
    )
    assert "source_image_id" in markdown
    assert "训练种子方差" in markdown
