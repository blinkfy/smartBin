from __future__ import annotations

from plot_method_architecture import draw_method_architecture


def test_method_architecture_writes_png_and_pdf(tmp_path):
    output = tmp_path / "architecture.png"

    report = draw_method_architecture(output)

    assert output.stat().st_size > 10_000
    assert output.with_suffix(".pdf").stat().st_size > 5_000
    assert report["scope"] == "implemented object-level classifier"
    assert report["query_paths"] == {
        "predictor_context": "t_v = Query(v, E)",
        "fusion_context": "t = Query(v + s_eff, E)",
    }
