from __future__ import annotations

import json

from build_full_regression_audit import build_full_regression_audit


FORMAL_OUTPUT = """\
................................................................ [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\\test_plan_a_visual_qa.py:8: could not import 'fitz': No module named 'fitz'
SKIPPED [1] tests\\test_plan_a_automated_paper_build.py:36: could not import 'docx': No module named 'docx'
528 passed, 2 skipped in 301.37s (0:05:01)
"""


def _prepare(tmp_path, formal_output=FORMAL_OUTPUT):
    formal_stdout = tmp_path / "formal.stdout.log"
    formal_stderr = tmp_path / "formal.stderr.log"
    document_stdout = tmp_path / "document.stdout.log"
    document_stderr = tmp_path / "document.stderr.log"
    formal_stdout.write_text(formal_output, encoding="utf-8")
    formal_stderr.write_text("", encoding="utf-8")
    document_stdout.write_text(
        "....... [100%]\n7 passed in 0.62s\n",
        encoding="utf-8",
    )
    document_stderr.write_text("", encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    builder = tmp_path / "build_full_regression_audit.py"
    builder.write_text("builder", encoding="utf-8")
    tests = []
    for name in (
        "test_plan_a_visual_qa.py",
        "test_plan_a_automated_paper_build.py",
    ):
        path = tmp_path / "tests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        tests.append(path)
    return (
        formal_stdout,
        formal_stderr,
        document_stdout,
        document_stderr,
        builder,
        tuple(tests),
    )


def test_full_regression_audit_combines_environment_coverage(tmp_path):
    formal, stderr, document, document_stderr, builder, tests = _prepare(
        tmp_path
    )
    output = tmp_path / "FULL_REGRESSION_AUDIT.json"

    report = build_full_regression_audit(
        project_root=tmp_path,
        formal_stdout_path=formal,
        formal_stderr_path=stderr,
        document_stdout_path=document,
        document_stderr_path=document_stderr,
        output_path=output,
        document_test_paths=tests,
        all_test_paths=tests,
        source_inventory_paths=(builder,),
        builder_path=builder,
        expected_formal_passed=528,
    )

    assert report["valid"] is True
    assert report["combined"]["unique_tests_passed"] == 535
    assert report["combined"]["unresolved_skips"] == 0
    assert report["formal_environment"]["source_file_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is True
    assert "535 passed" in output.with_suffix(".md").read_text(
        encoding="utf-8"
    )


def test_full_regression_audit_rejects_unexplained_count(tmp_path):
    formal, stderr, document, document_stderr, builder, tests = _prepare(
        tmp_path,
        FORMAL_OUTPUT.replace("528 passed", "527 passed"),
    )

    report = build_full_regression_audit(
        project_root=tmp_path,
        formal_stdout_path=formal,
        formal_stderr_path=stderr,
        document_stdout_path=document,
        document_stderr_path=document_stderr,
        output_path=tmp_path / "audit.json",
        document_test_paths=tests,
        all_test_paths=tests,
        source_inventory_paths=(builder,),
        builder_path=builder,
        expected_formal_passed=528,
    )

    assert report["valid"] is False
    assert report["combined"]["all_collected_tests_passed"] is False
