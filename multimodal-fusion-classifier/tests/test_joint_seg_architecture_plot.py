from __future__ import annotations

from plot_joint_seg_architecture import draw_joint_architecture


def test_joint_architecture_plot_writes_png_pdf_and_report(tmp_path) -> None:
    output = tmp_path / "joint.png"

    report = draw_joint_architecture(output)

    assert report["valid"] is True
    assert output.stat().st_size > 10_000
    assert output.with_suffix(".pdf").stat().st_size > 1_000
    assert output.with_suffix(".json").is_file()
    assert report["claim_boundaries"]["metric_forward_inputs"] == ["RGB"]
