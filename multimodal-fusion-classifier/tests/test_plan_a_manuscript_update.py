from __future__ import annotations

import json

import pytest

from apply_plan_a_confirmatory_manuscript_update import (
    BEGIN_MARKER,
    CONCLUSION_ANCHOR,
    DEFAULT_MAIN,
    END_MARKER,
    INSERTION_ANCHOR,
    NEW_LIMITATION,
    OLD_LIMITATION,
    apply_update,
    revised_tex,
)


def _source() -> str:
    return (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        + OLD_LIMITATION
        + "\n"
        + INSERTION_ANCHOR
        + "\n诊断正文。\n"
        + CONCLUSION_ANCHOR
        + "\n\\end{document}\n"
    )


def test_canonical_manuscript_is_ready_or_already_revised_once():
    source = DEFAULT_MAIN.read_text(encoding="utf-8")

    if BEGIN_MARKER in source or END_MARKER in source:
        assert source.count(BEGIN_MARKER) == 1
        assert source.count(END_MARKER) == 1
        assert OLD_LIMITATION not in source
        assert r"\subsection{组件分析、分布外退化与方法比较}" in source
        assert "等11种软件条件" in source
        assert "QMF" in source
        assert "TMC" in source
        return

    revised = revised_tex(
        source,
        r"\subsection{方案A预检夹具}" "\n仅验证插入契约。",
    )
    assert revised.count(BEGIN_MARKER) == 1
    assert revised.count(END_MARKER) == 1
    assert OLD_LIMITATION not in revised
    assert NEW_LIMITATION in revised


def test_revised_tex_inserts_fragment_and_retires_pending_limitations():
    addendum = "\\subsection{组件级归因、未见退化与直接基线}\n结果。"

    revised = revised_tex(_source(), addendum)

    assert revised.count(BEGIN_MARKER) == 1
    assert revised.count(END_MARKER) == 1
    assert revised.index("组件级归因") < revised.index("潜表示不确定性诊断")
    assert OLD_LIMITATION not in revised
    assert NEW_LIMITATION in revised
    assert "11种未见软件条件" in revised
    assert "TMC仅保留seed42单种子参考" in revised
    assert revised.count("\\begin{document}") == 1


def test_revised_tex_refuses_repeat_or_standalone_document():
    source = _source()
    fragment = "\\subsection{结果}\n正文。"

    revised = revised_tex(source, fragment)
    with pytest.raises(ValueError, match="already exist"):
        revised_tex(revised, fragment)
    with pytest.raises(ValueError, match="fragment"):
        revised_tex(source, "\\documentclass{article}")


def test_apply_update_binds_backup_and_revised_source(tmp_path):
    main_path = tmp_path / "main.tex"
    addendum_path = tmp_path / "addendum.tex"
    outcome_path = tmp_path / "OUTCOME.json"
    audit_path = tmp_path / "audit.json"
    backup_dir = tmp_path / "backup"
    main_path.write_text(_source(), encoding="utf-8")
    addendum_path.write_text(
        "\\subsection{组件级归因、未见退化与直接基线}\n结果。",
        encoding="utf-8",
    )
    outcome_path.write_text(
        json.dumps(
            {
                "state": "complete",
                "valid": True,
                "scope": (
                    "plan_a_post_queue_confirmatory_manuscript_evidence"
                ),
            }
        ),
        encoding="utf-8",
    )

    report = apply_update(
        main_path,
        addendum_path,
        outcome_path,
        audit_path,
        backup_dir,
    )

    assert report["valid"] is True
    assert audit_path.is_file()
    backups = list(backup_dir.glob("main.*.tex"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == _source()
    assert BEGIN_MARKER in main_path.read_text(encoding="utf-8")
    assert (
        report["source_bindings"]["pre_revision_main"]["sha256"]
        != report["revised_main"]["sha256"]
    )
    with pytest.raises(FileExistsError, match="revision audit"):
        apply_update(
            main_path,
            addendum_path,
            outcome_path,
            audit_path,
            backup_dir,
        )
