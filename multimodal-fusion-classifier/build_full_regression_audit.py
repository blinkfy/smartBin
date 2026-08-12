"""Audit the split-environment full regression after Plan A release tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
TRAINING_ROOT = REPORT_ROOT / "training"
DEFAULT_FORMAL_STDOUT = (
    TRAINING_ROOT
    / "full_regression_after_mixed_seed_amendment.stdout.log"
)
DEFAULT_FORMAL_STDERR = (
    TRAINING_ROOT
    / "full_regression_after_mixed_seed_amendment.stderr.log"
)
DEFAULT_DOCUMENT_STDOUT = (
    TRAINING_ROOT
    / "document_regression_after_mixed_seed_amendment.stdout.log"
)
DEFAULT_DOCUMENT_STDERR = (
    TRAINING_ROOT
    / "document_regression_after_mixed_seed_amendment.stderr.log"
)
DEFAULT_OUTPUT = REPORT_ROOT / "FULL_REGRESSION_AUDIT.json"
DOCUMENT_TESTS = (
    PROJECT_ROOT / "tests" / "test_plan_a_visual_qa.py",
    PROJECT_ROOT / "tests" / "test_plan_a_automated_paper_build.py",
)
EXPECTED_FORMAL_PASSED = 537
EXPECTED_FORMAL_SKIPPED_MODULES = 2
EXPECTED_DOCUMENT_PASSED = 7
SOURCE_INVENTORY_PATTERNS = (
    "*.py",
    "mmfc/*.py",
    "runtime_recovery/*.py",
    "paper_rewriting_output/final_paper/*.py",
    "configs/**/*.json",
    "schemas/*.json",
    "reports/qwen-material/training/*.ps1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit full pytest coverage across formal and document environments"
    )
    parser.add_argument(
        "--formal-stdout",
        type=Path,
        default=DEFAULT_FORMAL_STDOUT,
    )
    parser.add_argument(
        "--formal-stderr",
        type=Path,
        default=DEFAULT_FORMAL_STDERR,
    )
    parser.add_argument(
        "--document-stdout",
        type=Path,
        default=DEFAULT_DOCUMENT_STDOUT,
    )
    parser.add_argument(
        "--document-stderr",
        type=Path,
        default=DEFAULT_DOCUMENT_STDERR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path, *, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (
        project_root / path
    ).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, *, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(project_root).as_posix()
    except ValueError:
        display = resolved.as_posix()
    return {
        "path": display,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _pytest_summary(text: str) -> tuple[int, int, float]:
    match = re.search(
        r"(?P<passed>\d+) passed"
        r"(?:, (?P<skipped>\d+) skipped)?"
        r" in (?P<seconds>\d+(?:\.\d+)?)s",
        text,
    )
    if not match:
        raise ValueError("pytest summary is missing or malformed")
    return (
        int(match.group("passed")),
        int(match.group("skipped") or 0),
        float(match.group("seconds")),
    )


def build_full_regression_audit(
    *,
    project_root: Path = PROJECT_ROOT,
    formal_stdout_path: Path = DEFAULT_FORMAL_STDOUT,
    formal_stderr_path: Path = DEFAULT_FORMAL_STDERR,
    document_stdout_path: Path = DEFAULT_DOCUMENT_STDOUT,
    document_stderr_path: Path = DEFAULT_DOCUMENT_STDERR,
    output_path: Path = DEFAULT_OUTPUT,
    document_test_paths: tuple[Path, ...] = DOCUMENT_TESTS,
    all_test_paths: tuple[Path, ...] | None = None,
    source_inventory_paths: tuple[Path, ...] | None = None,
    builder_path: Path = Path(__file__),
    expected_formal_passed: int = EXPECTED_FORMAL_PASSED,
    expected_document_passed: int = EXPECTED_DOCUMENT_PASSED,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    paths = {
        "formal_stdout": _resolve(
            formal_stdout_path,
            project_root=project_root,
        ),
        "formal_stderr": _resolve(
            formal_stderr_path,
            project_root=project_root,
        ),
        "document_stdout": _resolve(
            document_stdout_path,
            project_root=project_root,
        ),
        "document_stderr": _resolve(
            document_stderr_path,
            project_root=project_root,
        ),
        "pytest_config": project_root / "pytest.ini",
        "builder_source": _resolve(
            builder_path,
            project_root=project_root,
        ),
    }
    for index, path in enumerate(document_test_paths):
        paths[f"document_test_{index + 1}"] = _resolve(
            path,
            project_root=project_root,
        )
    resolved_test_paths = (
        tuple(
            sorted(
                (project_root / "tests").glob("test_*.py"),
                key=lambda path: path.as_posix(),
            )
        )
        if all_test_paths is None
        else tuple(
            _resolve(path, project_root=project_root)
            for path in all_test_paths
        )
    )
    if not resolved_test_paths:
        raise FileNotFoundError("full-regression test inventory is empty")
    for index, path in enumerate(resolved_test_paths):
        paths[f"test_inventory_{index + 1:03d}"] = path
    resolved_source_paths = (
        tuple(
            sorted(
                {
                    path.resolve()
                    for pattern in SOURCE_INVENTORY_PATTERNS
                    for path in project_root.glob(pattern)
                    if path.is_file()
                },
                key=lambda path: path.as_posix(),
            )
        )
        if source_inventory_paths is None
        else tuple(
            _resolve(path, project_root=project_root)
            for path in source_inventory_paths
        )
    )
    if not resolved_source_paths:
        raise FileNotFoundError("full-regression source inventory is empty")
    for index, path in enumerate(resolved_source_paths):
        paths[f"source_inventory_{index + 1:03d}"] = path
    output_path = _resolve(output_path, project_root=project_root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "full-regression audit inputs missing: " + ", ".join(missing)
        )

    formal_stdout = paths["formal_stdout"].read_text(
        encoding="utf-8-sig",
        errors="replace",
    )
    formal_stderr = paths["formal_stderr"].read_text(
        encoding="utf-8-sig",
        errors="replace",
    )
    document_stdout = paths["document_stdout"].read_text(
        encoding="utf-8-sig",
        errors="replace",
    )
    document_stderr = paths["document_stderr"].read_text(
        encoding="utf-8-sig",
        errors="replace",
    )
    formal_passed, formal_skipped, formal_seconds = _pytest_summary(
        formal_stdout
    )
    document_passed, document_skipped, document_seconds = _pytest_summary(
        document_stdout
    )
    errors = []
    if (
        formal_passed != expected_formal_passed
        or formal_skipped != EXPECTED_FORMAL_SKIPPED_MODULES
    ):
        errors.append(
            "formal regression summary mismatch: "
            f"{formal_passed} passed, {formal_skipped} skipped"
        )
    if formal_stderr.strip():
        errors.append("formal regression stderr is not empty")
    for marker in (
        "test_plan_a_visual_qa.py",
        "No module named 'fitz'",
        "test_plan_a_automated_paper_build.py",
        "No module named 'docx'",
    ):
        if marker not in formal_stdout:
            errors.append(f"formal skip provenance missing: {marker}")
    if document_passed != expected_document_passed or document_skipped != 0:
        errors.append(
            "document regression summary mismatch: "
            f"{document_passed} passed, {document_skipped} skipped"
        )
    if document_stderr.strip():
        errors.append("document regression stderr is not empty")

    report = {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "full_project_regression_with_document_dependency_supplement",
        "formal_environment": {
            "python": "E:/Anaconda/envs/yolov11/python.exe",
            "command": "python -m pytest -q",
            "process_priority": "Normal",
            "processor_affinity": "all available logical processors",
            "passed": formal_passed,
            "skipped_modules": formal_skipped,
            "elapsed_seconds": formal_seconds,
            "test_file_count": len(resolved_test_paths),
            "source_file_count": len(resolved_source_paths),
        },
        "document_environment": {
            "python": "E:/Anaconda/python.exe",
            "command": (
                "python -m pytest -q tests/test_plan_a_visual_qa.py "
                "tests/test_plan_a_automated_paper_build.py"
            ),
            "passed": document_passed,
            "skipped": document_skipped,
            "elapsed_seconds": document_seconds,
        },
        "combined": {
            "unique_tests_passed": formal_passed + document_passed,
            "unresolved_skips": 0,
            "unresolved_failures": 0 if not errors else len(errors),
            "all_collected_tests_passed": not errors,
        },
        "claim_boundary": (
            "This audit proves the current automated test suite passed across "
            "the formal training and document environments. It does not prove "
            "hardware, RGB-D, external-blind, or complete-patent validation."
        ),
        "source_bindings": {
            name: _binding(path, project_root=project_root)
            for name, path in paths.items()
        },
        "errors": errors,
    }
    _atomic_text(
        output_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# 全量回归审计",
        "",
        f"- 状态：{'PASS' if report['valid'] else 'FAIL'}。",
        (
            f"- 正式训练环境：{formal_passed} passed，"
            f"{formal_skipped} 个文档依赖模块跳过。"
        ),
        (
            f"- 文档环境补跑：{document_passed} passed，"
            f"{document_skipped} skipped。"
        ),
        (
            "- 合并后唯一测试："
            f"{report['combined']['unique_tests_passed']} passed，"
            "0 unresolved skips。"
        ),
        "",
        "该审计不代表实体硬件、RGB-D、外部盲测或完整专利复现。",
        "",
    ]
    if errors:
        lines.extend(["## 错误", "", *[f"- {error}" for error in errors], ""])
    _atomic_text(output_path.with_suffix(".md"), "\n".join(lines))
    return report


def main() -> None:
    args = parse_args()
    report = build_full_regression_audit(
        project_root=PROJECT_ROOT,
        formal_stdout_path=args.formal_stdout,
        formal_stderr_path=args.formal_stderr,
        document_stdout_path=args.document_stdout,
        document_stderr_path=args.document_stderr,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                **report["combined"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["valid"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
