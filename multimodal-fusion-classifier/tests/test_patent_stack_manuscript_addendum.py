from __future__ import annotations

from build_patent_stack_manuscript_addendum import build_addendum


def test_patent_stack_addendum_keeps_v6_development_only(tmp_path) -> None:
    output = tmp_path / "PATENT_STACK_MANUSCRIPT_ADDENDUM.md"

    report = build_addendum(output)
    markdown = output.read_text(encoding="utf-8")

    assert report["valid"] is True
    assert report["formal_paper_metric"] is False
    assert report["v6_result"]["formal_performance_evidence"] is False
    assert report["comparison"]["v6_promoted_for_task_performance"] is False
    assert "结构成立并不等于已经获得任务性能增益" in markdown
    assert "不是同步实体阵列" in markdown
