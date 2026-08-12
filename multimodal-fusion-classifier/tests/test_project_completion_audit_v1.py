from __future__ import annotations

import hashlib
import json

from build_project_completion_audit_v1 import (
    QWEN_PAPER_GATES,
    build_completion_audit,
)
from build_evidence_archive import PLAN_A_TRAINING_RUN_DIRS


def test_completion_audit_separates_qwen_paper_and_external_gates(tmp_path):
    output = tmp_path / "PROJECT_COMPLETION_AUDIT_V1.json"

    report = build_completion_audit(output)

    assert report["valid"] is True
    assert report["state"] == "complete"
    assert report["qwen_paper_goal_ready"] is all(
        report["qwen_paper_gates"][name] is True
        for name in QWEN_PAPER_GATES
    )
    assert report["blocking_count"] == len(report["blockers"])
    assert {
        item["gate"] for item in report["blockers"]
    } == {
        name
        for name in QWEN_PAPER_GATES
        if report["qwen_paper_gates"][name] is not True
    }
    assert "complete_patent_replication_ready" in report[
        "independent_claim_gates"
    ]
    assert "external_deployment_ready" in report[
        "independent_claim_gates"
    ]
    assert "paper_readiness_bindings_current" in report["source_integrity"]
    assert "full_regression_bindings_current" in report[
        "source_integrity"
    ]
    if report["qwen_paper_gates"]["full_regression_ready"]:
        assert report["source_integrity"][
            "full_regression_bindings_current"
        ] is True
    assert report["source_integrity"][
        "plan_a_training_metadata_file_count"
    ] == 2 * len(PLAN_A_TRAINING_RUN_DIRS)
    if report["qwen_paper_gates"]["internal_evidence_ready"]:
        assert report["source_integrity"][
            "paper_readiness_bindings_current"
        ] is True
    if report["qwen_paper_gates"]["patent_method_alignment_ready"]:
        assert report["source_integrity"][
            "patent_matrix_bindings_current"
        ] is True
    if report["qwen_paper_gates"]["plan_a_visual_qa_complete"]:
        assert report["source_integrity"][
            "plan_a_visual_qa_bindings_current"
        ] is True
    assert output.is_file()
    assert output.with_suffix(".md").is_file()


def test_completion_audit_binds_current_sources(tmp_path):
    report = build_completion_audit(tmp_path / "audit.json")

    for binding in report["source_bindings"].values():
        path = binding["path"]
        source = (
            __import__("pathlib").Path(__file__).resolve().parents[1] / path
        )
        assert source.is_file()
        assert source.stat().st_size == binding["size_bytes"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == binding[
            "sha256"
        ]


def test_completion_audit_markdown_preserves_chinese(tmp_path):
    output = tmp_path / "audit.json"
    build_completion_audit(output)
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert markdown.startswith("# 项目总完成审计 v1")
    assert "Qwen方案A论文目标完成" in markdown
    assert persisted["scope"].endswith("qwen_paper_option_a")
    assert not any(
        marker in markdown for marker in ("ä¸", "åˆ", "ï¼", "Â", "\ufffd")
    )
