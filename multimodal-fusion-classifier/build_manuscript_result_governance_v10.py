"""Record the frozen QMF/TMC extension without rewriting v9 history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V10.json"
INPUTS = {
    "governance_v9": REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V9.json",
    "recent_baseline_protocol": (
        REPORT_ROOT / "recent-robust-baselines" / "PROTOCOL.json"
    ),
}
STATUS_PATH = (
    REPORT_ROOT / "training" / "recent_robust_baseline_status.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the option-A manuscript governance v10 addendum"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def validate_sources(
    governance: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    if (
        governance.get("valid") is not True
        or governance.get("version") != 9
        or governance.get("decision", {}).get("confirmed_option") != "A"
        or protocol.get("valid") is not True
        or protocol.get("state") != "frozen_before_baseline_training"
        or protocol.get("methods") != ["qmf", "tmc"]
        or protocol.get("training_seeds") != [42, 43, 44]
        or protocol.get("run_count") != 6
    ):
        raise ValueError("governance v9 and baseline protocol disagree")


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite governance v10")
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("governance inputs missing: " + ", ".join(missing))
    governance = _read_json(INPUTS["governance_v9"])
    protocol = _read_json(INPUTS["recent_baseline_protocol"])
    validate_sources(governance, protocol)
    status = (
        _read_json(STATUS_PATH)
        if STATUS_PATH.is_file()
        else {"state": "not_started", "stage": "not_started"}
    )
    if status.get("state") not in {"waiting", "running", "complete"}:
        raise ValueError("recent baseline queue is not in an admissible state")

    remaining_gates = [
        dict(value) for value in governance.get("remaining_gates", [])
    ]
    for gate in remaining_gates:
        if gate.get("gate") == "direct_recent_robust_multimodal_baselines":
            gate["state"] = "implemented_preregistered_and_queued"

    report = {
        "schema_version": 1,
        "version": 10,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": governance["decision"],
        "paper_artifact_state": governance["paper_artifact_state"],
        "component_extension": governance["component_extension"],
        "unseen_condition_extension": governance[
            "unseen_condition_extension"
        ],
        "recent_baseline_extension": {
            "state": "implemented_preregistered_and_queued",
            "result_available": False,
            "methods": protocol["methods"],
            "training_seeds": protocol["training_seeds"],
            "run_count": protocol["run_count"],
            "conditions": list(protocol["conditions"]),
            "execution_dependency": protocol["execution_dependency"],
            "runtime_observation": {
                "state": status.get("state"),
                "stage": status.get("stage"),
                "updated_at": status.get("updated_at"),
                "mutable_status_path": STATUS_PATH.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
            },
            "claim_boundary": protocol["adaptation_contract"],
        },
        "remaining_gates": remaining_gates,
        "supplementary_pending": governance["supplementary_pending"],
        "unchanged_exclusions": governance["unchanged_exclusions"],
        "source_bindings": {
            name: _binding(path) for name, path in INPUTS.items()
        },
        "errors": [],
    }
    _atomic_write(output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(
        markdown,
        "\n".join(
            [
                "# 论文结果治理 v10：近期直接基线预注册附录",
                "",
                "- 方案A、论文题目和既有排除项保持不变。",
                "- QMF与TMC同输入适配已实现、测试、冻结并自动排队。",
                "- 两种方法各使用42/43/44三种子和100轮独立训练。",
                "- 干净测试及11种冻结传感器条件均须完整报告。",
                "- 当前尚无正式基线结果，不得提前写入优越性结论。",
                "",
                "## 表述边界",
                "",
                "这是把原方法融合与损失公式适配到视觉、视觉-候选文本原型和",
                "传感器三分支的同输入重实现，不是原作者数据集与编码器的逐字节",
                "复现；候选文本也不是独立采集的逐样本文本。",
                "",
                "## 仍需证据",
                "",
                "1. 完成组件消融、v3未见条件评估与六次直接基线并通过冻结审计。",
                "2. 获得独立外部盲测批次。",
                "3. 若声称完整专利复现，则补实体传感器与RGB-D。",
                "",
            ]
        ),
    )
    return report


def main() -> None:
    report = build(parse_args().output)
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "version": report["version"],
                "recent_baseline_state": report[
                    "recent_baseline_extension"
                ]["state"],
                "result_available": report[
                    "recent_baseline_extension"
                ]["result_available"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
