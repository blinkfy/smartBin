"""Apply the audited Plan A confirmatory addendum to the canonical TeX source."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAIN = (
    PROJECT_ROOT / "paper_rewriting_output" / "final_paper" / "main.tex"
)
DEFAULT_ADDENDUM = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "plan-a-confirmatory-outcome"
    / "MANUSCRIPT_ADDENDUM.tex"
)
DEFAULT_OUTCOME = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "plan-a-confirmatory-outcome"
    / "OUTCOME.json"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "paper_rewriting_output"
    / "plan_a_confirmatory_revision_audit.json"
)
DEFAULT_BACKUP_DIR = (
    PROJECT_ROOT
    / "paper_rewriting_output"
    / "revision_snapshots"
    / "pre_plan_a_confirmatory"
)
INSERTION_ANCHOR = r"\subsection{潜表示不确定性诊断}"
BEGIN_MARKER = "% BEGIN AUTO PLAN_A_CONFIRMATORY"
END_MARKER = "% END AUTO PLAN_A_CONFIRMATORY"
OLD_LIMITATION = """独立外部数据；同步物理阵列、时间配准和物理量校准也未完成。第五，软件退化与
训练增强同族，训练范围外强度、未见故障以及availability错误尚未验证。第六，
主要匹配变体联合移除整个可靠性子系统，历史组件消融又只有单训练种子；当前证据
不能完成组件级归因，也未直接实现最近邻鲁棒融合方法作为统一训练基线。原专利方向中的"""
NEW_LIMITATION = """独立外部数据；同步物理阵列、时间配准和物理量校准也未完成。第五，新增的
范围外强度、未见故障和availability错误均为确定性软件干预，仍不能替代实体
传感器故障或跨设备验证。第六，两个辅助监督组件和QMF同输入适配按三个匹配
种子报告；TMC仅保留seed42单种子参考，不计算样本标准差或声称重复性。直接
基线仍复用内部固定测试集，也不是原作者数据、模态编码器和训练流水线的逐字
复现。原专利方向中的"""
CONCLUSION_ANCHOR = """上述证据限定于19类内部固定基准及图像条件sidecar。潜空间不确定性能够排序误差，"""
CONCLUSION_REPLACEMENT = """追加的组件独立重训、11种未见软件条件以及QMF/TMC同输入直接基线进一步
补足了组件归因、训练分布外故障响应和近期方法对照；QMF保留三种子结果，TMC
按透明范围修订仅作单种子参考，部分seed43与未启动seed44均不进入统计。这不
改变内部测试复用、软件注入和非硬件观测的证据边界。

上述证据限定于19类内部固定基准及图像条件sidecar。潜空间不确定性能够排序误差，"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the audited Plan A confirmatory TeX revision"
    )
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--addendum", type=Path, default=DEFAULT_ADDENDUM)
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = resolved.as_posix()
    return {
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def revised_tex(source: str, addendum: str) -> str:
    """Return a one-time, anchor-checked manuscript revision."""

    if BEGIN_MARKER in source or END_MARKER in source:
        raise ValueError("Plan A confirmatory markers already exist")
    if source.count(INSERTION_ANCHOR) != 1:
        raise ValueError("confirmatory insertion anchor identity mismatch")
    if source.count(OLD_LIMITATION) != 1:
        raise ValueError("old limitation paragraph identity mismatch")
    if source.count(CONCLUSION_ANCHOR) != 1:
        raise ValueError("conclusion anchor identity mismatch")
    if r"\documentclass" in addendum or r"\begin{document}" in addendum:
        raise ValueError("addendum must be a TeX fragment")
    fragment = (
        BEGIN_MARKER
        + "\n"
        + addendum.strip()
        + "\n"
        + END_MARKER
        + "\n\n"
    )
    revised = source.replace(
        INSERTION_ANCHOR,
        fragment + INSERTION_ANCHOR,
        1,
    )
    revised = revised.replace(OLD_LIMITATION, NEW_LIMITATION, 1)
    revised = revised.replace(
        CONCLUSION_ANCHOR,
        CONCLUSION_REPLACEMENT,
        1,
    )
    return revised


def apply_update(
    main_path: Path = DEFAULT_MAIN,
    addendum_path: Path = DEFAULT_ADDENDUM,
    outcome_path: Path = DEFAULT_OUTCOME,
    audit_path: Path = DEFAULT_AUDIT,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
) -> dict[str, Any]:
    main_path = _resolve(main_path)
    addendum_path = _resolve(addendum_path)
    outcome_path = _resolve(outcome_path)
    audit_path = _resolve(audit_path)
    backup_dir = _resolve(backup_dir)
    required = {
        "main": main_path,
        "addendum": addendum_path,
        "outcome": outcome_path,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "manuscript update inputs missing: " + ", ".join(missing)
        )
    if audit_path.exists():
        raise FileExistsError("refusing to overwrite manuscript revision audit")
    outcome = json.loads(outcome_path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(outcome, dict)
        or outcome.get("valid") is not True
        or outcome.get("state") != "complete"
        or outcome.get("scope")
        != "plan_a_post_queue_confirmatory_manuscript_evidence"
    ):
        raise ValueError("Plan A confirmatory outcome is not valid")

    original = main_path.read_text(encoding="utf-8")
    addendum = addendum_path.read_text(encoding="utf-8")
    revised = revised_tex(original, addendum)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"main.{timestamp}.tex"
    if backup_path.exists():
        raise FileExistsError(f"backup path already exists: {backup_path}")
    _atomic_text(backup_path, original)
    original_binding = _binding(backup_path)
    _atomic_text(main_path, revised)
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "plan_a_confirmatory_canonical_tex_revision",
        "markers": {
            "begin": BEGIN_MARKER,
            "end": END_MARKER,
            "count_in_revised_source": (
                revised.count(BEGIN_MARKER) + revised.count(END_MARKER)
            ),
        },
        "changes": [
            "inserted audited confirmatory TeX fragment before uncertainty diagnostics",
            "replaced obsolete pending-experiment limitation text",
            "added a boundary-preserving confirmatory conclusion paragraph",
        ],
        "source_bindings": {
            "pre_revision_main": original_binding,
            "confirmatory_addendum": _binding(addendum_path),
            "confirmatory_outcome": _binding(outcome_path),
            "updater_source": _binding(Path(__file__)),
        },
        "revised_main": _binding(main_path),
        "errors": [],
    }
    _atomic_text(
        audit_path,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> None:
    args = parse_args()
    report = apply_update(
        args.main,
        args.addendum,
        args.outcome,
        args.audit,
        args.backup_dir,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "revised_main": report["revised_main"]["path"],
                "backup": report["source_bindings"]["pre_revision_main"][
                    "path"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
