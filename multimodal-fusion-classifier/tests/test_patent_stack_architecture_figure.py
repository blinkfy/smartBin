from __future__ import annotations

from plot_patent_stack_architecture import architecture_spec, parse_args


def test_architecture_spec_preserves_claim_boundaries() -> None:
    spec = architecture_spec()

    assert spec["head"] == "native YOLO11s-seg Segment head"
    assert spec["boundaries"]["rgb_d_input"] is False
    assert spec["boundaries"]["hardware_sensor"] is False
    assert spec["boundaries"]["ground_truth_class_in_text_query"] is False
    assert "8-head 3-token attention" in spec["fusion_path"]


def test_architecture_figure_defaults_are_additive(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["plot"])

    args = parse_args()

    assert args.output.name == "patent_stack_architecture.png"
    assert args.caption.name == "PATENT_STACK_FIGURE_CAPTION.md"
