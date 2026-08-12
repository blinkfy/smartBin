from pathlib import Path

from audit_manuscript_stable_sections import (
    DEFAULT_SOURCES,
    REQUIRED_BOUNDARIES,
    REQUIRED_HEADINGS,
    REQUIRED_METHOD_CONTRACTS,
    audit_manuscript_stable_sections,
)


def _sources(tmp_path: Path, *, invalid: bool = False) -> dict[str, Path]:
    citations = [f"cite{index}" for index in range(10)]
    manuscript = "\n".join(
        [
            "# stable",
            *REQUIRED_HEADINGS,
            *REQUIRED_BOUNDARIES,
            *REQUIRED_METHOD_CONTRACTS,
            "[" + "; ".join(f"@{key}" for key in citations) + "]",
            "Accuracy 90%" if invalid else "dynamic results are omitted",
            "[@missing]" if invalid else "",
        ]
    )
    bibliography = "\n".join(
        f"@article{{{key}, title={{Title {key}}}}}" for key in citations
    )
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "manuscript": "manuscript.md",
            "bibliography": "references.bib",
            "method_spec": "method.md",
            "dataset_card": "dataset.md",
            "statistical_protocol": "statistics.md",
            "patent_alignment": "patent.md",
        }.items()
    }
    paths["manuscript"].write_text(manuscript, encoding="utf-8")
    paths["bibliography"].write_text(bibliography, encoding="utf-8")
    for name in (
        "method_spec",
        "dataset_card",
        "statistical_protocol",
        "patent_alignment",
    ):
        paths[name].write_text(name, encoding="utf-8")
    return paths


def test_stable_manuscript_audit_accepts_boundaries_and_citations(tmp_path):
    report = audit_manuscript_stable_sections(sources=_sources(tmp_path))

    assert report["valid"] is True
    assert report["summary"]["citation_count"] == 10
    assert report["summary"]["percentage_literal_count"] == 0
    assert len(report["source_bindings"]) == 6


def test_stable_manuscript_audit_rejects_dynamic_number_and_unknown_cite(
    tmp_path,
):
    report = audit_manuscript_stable_sections(
        sources=_sources(tmp_path, invalid=True)
    )

    assert report["valid"] is False
    assert report["missing_citations"] == ["missing"]
    assert report["percentage_literals"] == ["90%"]


def test_default_sources_lock_governance_and_training_integrity_evidence():
    assert {
        "release_checklist",
        "image_recovery_outcome",
        "source_repair_transition",
        "formal_interruption_addendum",
        "formal_interruption_outcome",
        "yolo_dependency_incident",
        "yolo_dependency_incident_addendum",
        "scene_strata_reporting_incident",
        "scene_strata_reporting_addendum",
        "scene_strata_reporting_outcome",
        "v3_config",
    }.issubset(DEFAULT_SOURCES)
    assert {
        "未完成独立隐私审查",
        "写入来源和精确时刻未知",
        "不等同于专利实施例的医疗废物类别体系",
    }.issubset(REQUIRED_BOUNDARIES)


def test_stable_manuscript_audit_rejects_missing_method_contract(tmp_path):
    sources = _sources(tmp_path)
    manuscript = sources["manuscript"]
    text = manuscript.read_text(encoding="utf-8")
    text = text.replace(REQUIRED_METHOD_CONTRACTS[0], "")
    manuscript.write_text(text, encoding="utf-8")

    report = audit_manuscript_stable_sections(sources=sources)

    assert report["valid"] is False
    assert report["missing_method_contracts"] == [
        REQUIRED_METHOD_CONTRACTS[0]
    ]
