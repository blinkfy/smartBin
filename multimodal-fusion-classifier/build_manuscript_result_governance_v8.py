"""Record option-A confirmation and the post-draft experiment state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
DEFAULT_OUTPUT = REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V8.json"
INPUTS = {
    "governance_v7": REPORT_ROOT / "MANUSCRIPT_RESULT_GOVERNANCE_V7.json",
    "motivation_packet": REPORT_ROOT / "MOTIVATION_DECISION_PACKET.json",
    "component_protocol": (
        REPORT_ROOT
        / "confirmatory-ablations"
        / "v3-loss-components"
        / "PROTOCOL.json"
    ),
    "paper_source": (
        PROJECT_ROOT / "paper_rewriting_output" / "final_paper" / "main.tex"
    ),
    "paper_progress": PROJECT_ROOT / "paper_rewriting_output" / "progress.md",
}
EXPECTED_TITLE = "面向图像条件软传感器退化的可靠多模态垃圾分类方法"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the option-A manuscript governance v8 addendum"
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


def _extract_title(tex: str) -> str:
    match = re.search(r"\\title\{\\textbf\{([^{}]+)\}\}", tex)
    if not match:
        raise ValueError("paper title could not be extracted")
    return match.group(1).strip()


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _resolve(output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError("refusing to overwrite governance v8")
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("governance inputs missing: " + ", ".join(missing))
    v7 = _read_json(INPUTS["governance_v7"])
    motivation = _read_json(INPUTS["motivation_packet"])
    protocol = _read_json(INPUTS["component_protocol"])
    title = _extract_title(INPUTS["paper_source"].read_text(encoding="utf-8"))
    progress = INPUTS["paper_progress"].read_text(encoding="utf-8")
    if (
        v7.get("valid") is not True
        or v7.get("version") != 7
        or v7.get("paper_scope", {}).get("recommended_option") != "A"
        or v7.get("paper_scope", {}).get("user_confirmed") is not False
        or motivation.get("recommended_option") != "A"
        or protocol.get("state") != "frozen_before_component_training"
        or protocol.get("valid") is not True
        or title != EXPECTED_TITLE
        or "Next stage: Complete" not in progress
    ):
        raise ValueError("option-A confirmation sources are inconsistent")

    report = {
        "schema_version": 1,
        "version": 8,
        "state": "complete",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "confirmed_option": "A",
            "confirmation_source": (
                "user statement in the active Codex thread: 我选择方案A"
            ),
            "confirmation_is_external_to_file_archive": True,
            "paper_title": title,
            "title_locked_to_option_a": True,
            "primary_motivation": (
                "reliable multimodal classification under degraded "
                "image-conditioned soft-sensor observations"
            ),
        },
        "paper_artifact_state": {
            "audited_draft_complete": True,
            "paper_spine_final_gate_passed": True,
            "external_submission_ready": False,
            "editorial_position": "major_revision_before_external_submission",
        },
        "new_confirmatory_extension": {
            "name": "v3_auxiliary_loss_component_ablation",
            "state": "preregistered_and_running",
            "variants": [
                "v3_no_grid_reconstruction",
                "v3_no_ranking",
            ],
            "training_seeds": [42, 43, 44],
            "run_count": 6,
            "intended_manuscript_role": (
                "component-level evidence for the v3 grid-reconstruction and "
                "pairwise-ranking supervision terms"
            ),
            "result_inclusion_rule": (
                "include null, favorable, and adverse results only after the "
                "frozen outcome audit passes"
            ),
        },
        "remaining_gates": [
            {
                "priority": 1,
                "gate": "v3_component_multiseed_ablation",
                "owner": "local_compute",
                "state": "running",
            },
            {
                "priority": 2,
                "gate": "unseen_degradation_and_availability_errors",
                "owner": "local_experiment",
                "state": "not_yet_preregistered",
            },
            {
                "priority": 3,
                "gate": "direct_recent_robust_multimodal_baselines",
                "owner": "method_implementation",
                "state": "not_yet_implemented",
            },
            {
                "priority": 4,
                "gate": "external_blind_validation",
                "owner": "new_data_collection",
                "state": "awaiting_external_bundle",
            },
            {
                "priority": 5,
                "gate": "physical_sensor_or_rgbd_patent_completion",
                "owner": "hardware_and_data",
                "state": "not_implemented",
            },
        ],
        "supplementary_pending": {
            "formal_direct_full_image_qwen_v6": (
                "preregistered but awaiting API credentials and data; "
                "supplementary rather than required for option-A core claims"
            )
        },
        "unchanged_exclusions": v7["mandatory_exclusions"],
        "source_bindings": {
            name: _binding(path) for name, path in INPUTS.items()
        },
        "errors": [],
    }
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        markdown,
        "\n".join(
            [
                "# 论文结果治理 v8：方案A确认附录",
                "",
                "- 用户已确认方案A。",
                f"- 锁定题目：{title}。",
                "- PaperSpine完整稿与最终门禁已通过。",
                "- 外部投稿状态：Major Revision，尚未宣告投稿完成。",
                "",
                "## 当前运行",
                "",
                (
                    "v3的网格重建监督与可靠性排序监督正在进行"
                    "42/43/44三种子独立消融。所有正、负和零结果均须在冻结审计"
                    "通过后纳入。"
                ),
                "",
                "## 仍需证据",
                "",
                "1. 未见退化和availability误报/漏报。",
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
                "confirmed_option": report["decision"]["confirmed_option"],
                "component_extension_state": report[
                    "new_confirmatory_extension"
                ]["state"],
                "external_submission_ready": report[
                    "paper_artifact_state"
                ]["external_submission_ready"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
