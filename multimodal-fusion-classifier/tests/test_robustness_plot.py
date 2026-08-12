from __future__ import annotations

import csv

import pytest

from plot_robustness_comparison import REQUIRED_COLUMNS, load_rows


def test_load_robustness_comparison_allows_missing_rank_values(tmp_path):
    path = tmp_path / "comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                name: (
                    "clean"
                    if name == "scenario"
                    else ""
                    if name in {"v2b_spearman", "v3_spearman"}
                    else "0.5"
                )
                for name in REQUIRED_COLUMNS
            }
        )

    rows = load_rows(path)

    assert rows[0]["scenario"] == "clean"
    assert rows[0]["v2b_macro_f1"] == pytest.approx(0.5)
    assert rows[0]["v2b_spearman"] != rows[0]["v2b_spearman"]


def test_load_robustness_comparison_rejects_missing_columns(tmp_path):
    path = tmp_path / "comparison.csv"
    path.write_text("scenario,v3_macro_f1\nclean,0.8\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        load_rows(path)
