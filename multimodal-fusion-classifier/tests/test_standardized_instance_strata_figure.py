from __future__ import annotations

from plot_standardized_instance_strata import figure_spec, parse_args


def test_standardized_strata_figure_preserves_evidence_boundary() -> None:
    spec = figure_spec()

    assert spec["metric"] == "mask_mean_recall50_95"
    assert spec["formal_paper_metric"] is False
    assert spec["bbox_overlap_is_human_occlusion"] is False
    assert spec["y_limits"][0] <= 0.55
    assert spec["models"] == [
        "rgb_yolo",
        "v5_rgb_fallback",
        "v6_rgb_fallback",
    ]


def test_standardized_strata_figure_defaults_are_additive(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["plot"])

    args = parse_args()

    assert args.output.name == "standardized_instance_strata.png"
    assert args.caption.name == "FIGURE_CAPTION.md"
