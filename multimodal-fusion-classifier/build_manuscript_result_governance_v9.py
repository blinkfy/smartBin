"""Record the frozen unseen-condition extension without rewriting v8 history."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V9.json"
INPUTS = {
    "governance_v8": REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V8.json",
    "unseen_condition_protocol": (
        REPORT_ROOT / "unseen-sensor-conditions-v1" / "PROTOCOL.json"
    ),
    "unseen_condition_runner": (
        REPORT_ROOT
        / "training"
        / "run_unseen_sensor_conditions_after_components.ps1"
    ),
}
STATUS_PATH = REPORT_ROOT / "training" / "unseen_sensor_conditions_status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the option-A manuscript governance v9 addendum"
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


def _validate_source_state(
    governance: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    if (
        governance.get("valid") is not True
        or governance.get("version") != 8
        or governance.get("decision", {}).get("confirmed_option") != "A"
        or protocol.get("valid") is not True
        or protocol.get("state") != "frozen_before_unseen_condition_evaluation"
        or len(protocol.get("conditions", {})) != 11
        or protocol.get("model_seeds") != [42, 43, 44]
    ):
        raise ValueError("governance v8 and unseen-condition protocol disagree")


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite governance v9")
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("governance inputs missing: " + ", ".join(missing))

    governance = _read_json(INPUTS["governance_v8"])
    protocol = _read_json(INPUTS["unseen_condition_protocol"])
    _validate_source_state(governance, protocol)

    runtime_status = (
        _read_json(STATUS_PATH)
        if STATUS_PATH.is_file()
        else {"state": "not_started", "stage": "not_started"}
    )
    if runtime_status.get("state") not in {"waiting", "running", "complete"}:
        raise ValueError("unseen-condition queue is not in an admissible state")

    remaining_gates = [
        dict(gate) for gate in governance.get("remaining_gates", [])
    ]
    for gate in remaining_gates:
        if gate.get("gate") == "unseen_degradation_and_availability_errors":
            gate["state"] = "preregistered_and_queued"

    report = {
        "schema_version": 1,
        "version": 9,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": governance["decision"],
        "paper_artifact_state": governance["paper_artifact_state"],
        "component_extension": governance["new_confirmatory_extension"],
        "unseen_condition_extension": {
            "state": "preregistered_and_queued",
            "result_available": False,
            "model_seeds": protocol["model_seeds"],
            "condition_count": len(protocol["conditions"]),
            "condition_names": protocol["conditions"],
            "execution_dependency": "v3_component_multiseed_ablation",
            "runtime_observation": {
                "state": runtime_status.get("state"),
                "stage": runtime_status.get("stage"),
                "updated_at": runtime_status.get("updated_at"),
                "mutable_status_path": STATUS_PATH.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
            },
            "claim_boundary": (
                "the frozen protocol and queued runner are ready, but no unseen-"
                "condition performance may be reported before OUTCOME audit"
            ),
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
                "# 论文结果治理 v9：未见传感器条件预注册附录",
                "",
                "- 方案A、论文题目与v8主张边界保持不变。",
                "- 11种未见退化及availability误报/漏报条件已经冻结。",
                "- 完整v3种子42/43/44将在组件消融审计通过后自动评估。",
                "- 当前仅完成预注册与排队，尚无可写入结果表的性能结论。",
                "",
                "## 执行边界",
                "",
                "评估不再训练模型；所有扰动确定性生成，并采用源图簇配对",
                "bootstrap。上游失败时接力脚本关闭，不生成正式结果。",
                "",
                "## 仍需证据",
                "",
                "1. 完成组件消融与未见条件评估并通过各自冻结审计。",
                "2. 近期鲁棒多模态方法的同协议直接基线。",
                "3. 新外部盲测批次。",
                "4. 若声称完整专利复现，则需实体传感器与RGB-D。",
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
                "unseen_condition_state": report[
                    "unseen_condition_extension"
                ]["state"],
                "result_available": report[
                    "unseen_condition_extension"
                ]["result_available"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
