"""Build an additive, hash-bound evidence index for the joint-head extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_ROOT = PROJECT_ROOT / "reports" / "qwen-material"
JOINT_ROOT = REPORTS_ROOT / "joint-multimodal-seg"
DEFAULT_OUTPUT = REPORTS_ROOT / "JOINT_HEAD_EXTENSION_INDEX.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the audited joint-head extension index"
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


def _validate_protocol_sources(protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    bindings = protocol.get("source_bindings")
    if not isinstance(bindings, Mapping):
        return ["retrained-ablation protocol source bindings are missing"]
    for relative, expected_hash in bindings.items():
        path = _resolve(Path(str(relative)))
        if not path.is_file():
            errors.append(f"protocol source is missing: {relative}")
        elif _sha256(path) != expected_hash:
            errors.append(f"protocol source binding drift: {relative}")
    return errors


def _validate_document_bindings(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    bindings = document.get("source_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        return ["document source bindings are missing"]
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            errors.append(f"malformed document binding: {name}")
            continue
        path = _resolve(Path(str(binding.get("path", ""))))
        if not path.is_file():
            errors.append(f"document source is missing: {name}")
            continue
        if (
            path.stat().st_size != binding.get("size_bytes")
            or _sha256(path) != binding.get("sha256")
        ):
            errors.append(f"document source binding drift: {name}")
    return errors


def build_index(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = _resolve(output_path)
    paths = {
        "builder_source": Path(__file__),
        "base_submission_index": REPORTS_ROOT / "SUBMISSION_INDEX.json",
        "base_readiness_audit": REPORTS_ROOT / "PAPER_READINESS_AUDIT.json",
        "base_archive": (
            REPORTS_ROOT / "submission-evidence" / "paper-evidence.zip"
        ),
        "base_archive_manifest": (
            REPORTS_ROOT / "submission-evidence" / "archive-manifest.json"
        ),
        "multiseed_summary": JOINT_ROOT / "MULTISEED_SUMMARY.json",
        "multiseed_protocol": JOINT_ROOT / "MULTISEED_PROTOCOL.json",
        "joint_readiness": JOINT_ROOT / "PAPER_READINESS.json",
        "joint_method_spec_json": JOINT_ROOT / "JOINT_METHOD_SPEC.json",
        "joint_method_spec_md": JOINT_ROOT / "JOINT_METHOD_SPEC.md",
        "architecture_json": JOINT_ROOT / "joint_head_architecture.json",
        "architecture_png": JOINT_ROOT / "joint_head_architecture.png",
        "architecture_pdf": JOINT_ROOT / "joint_head_architecture.pdf",
        "performance_benchmark_json": JOINT_ROOT / "PERFORMANCE_BENCHMARK.json",
        "performance_benchmark_md": JOINT_ROOT / "PERFORMANCE_BENCHMARK.md",
        "deployment_profile_json": JOINT_ROOT / "DEPLOYMENT_PROFILE.json",
        "deployment_profile_md": JOINT_ROOT / "DEPLOYMENT_PROFILE.md",
        "ablation_protocol": (
            JOINT_ROOT / "retrained-ablations" / "PROTOCOL.json"
        ),
        "ablation_status": JOINT_ROOT / "retrained-ablations" / "STATUS.json",
        "ablation_outcome": JOINT_ROOT / "retrained-ablations" / "OUTCOME.json",
        "ablation_report": JOINT_ROOT / "retrained-ablations" / "REPORT.md",
        "manuscript_results_json": (
            REPORTS_ROOT / "MANUSCRIPT_RESULTS_SECTIONS.json"
        ),
        "manuscript_results_md": REPORTS_ROOT / "MANUSCRIPT_RESULTS_SECTIONS.md",
        "stable_manuscript_audit": (
            REPORTS_ROOT / "MANUSCRIPT_STABLE_SECTIONS_AUDIT.json"
        ),
        "stable_manuscript_md": REPORTS_ROOT / "MANUSCRIPT_STABLE_SECTIONS.md",
        "patent_alignment": REPORTS_ROOT / "PATENT_ALIGNMENT.md",
        "decision_brief": REPORTS_ROOT / "MANUSCRIPT_DECISION_BRIEF.md",
        "figure_captions": REPORTS_ROOT / "FIGURE_CAPTIONS.md",
        "motivation_options": (
            REPORTS_ROOT / "MOTIVATION_OPTIONS_AFTER_JOINT_EVIDENCE.md"
        ),
        "readme": PROJECT_ROOT / "README.md",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "joint-head extension inputs are missing: " + ", ".join(missing)
        )

    base_index = _read_json(paths["base_submission_index"])
    base_readiness = _read_json(paths["base_readiness_audit"])
    base_archive_manifest = _read_json(paths["base_archive_manifest"])
    summary = _read_json(paths["multiseed_summary"])
    readiness = _read_json(paths["joint_readiness"])
    protocol = _read_json(paths["ablation_protocol"])
    status = _read_json(paths["ablation_status"])
    ablation = _read_json(paths["ablation_outcome"])
    method = _read_json(paths["joint_method_spec_json"])
    architecture = _read_json(paths["architecture_json"])
    performance = _read_json(paths["performance_benchmark_json"])
    deployment = _read_json(paths["deployment_profile_json"])
    manuscript = _read_json(paths["manuscript_results_json"])
    stable_audit = _read_json(paths["stable_manuscript_audit"])

    errors: list[str] = []
    base_binding = base_index.get("artifacts", {}).get(
        "PAPER_READINESS_AUDIT.json", {}
    )
    if (
        base_index.get("state") != "complete"
        or base_index.get("gates", {}).get("internal_paper_evidence_ready")
        is not True
        or base_readiness.get("state") != "complete"
        or base_readiness.get("valid") is not True
        or base_binding.get("sha256") != _sha256(paths["base_readiness_audit"])
        or base_archive_manifest.get("archive_sha256")
        != _sha256(paths["base_archive"])
        or base_archive_manifest.get("checkpoints_included") is not False
        or base_archive_manifest.get("raw_data_included") is not False
    ):
        errors.append("locked base evidence chain is not valid")
    if (
        summary.get("valid") is not True
        or summary.get("run_count") != 3
        or summary.get("seeds") != [42, 43, 44]
        or readiness.get("valid") is not True
        or readiness.get("run_counts", {}).get(
            "independently_retrained_branch_controls"
        )
        != 6
    ):
        errors.append("joint-head multiseed readiness evidence is not valid")
    if (
        protocol.get("status") != "frozen_before_formal_ablation_training"
        or status.get("state") != "complete"
        or status.get("run_count") != 6
        or ablation.get("valid") is not True
        or ablation.get("formal_run_count") != 6
        or ablation.get("independent_retraining") is not True
        or ablation.get("same_checkpoint_masking") is not False
    ):
        errors.append("independent branch-ablation evidence is not valid")
    errors.extend(_validate_protocol_sources(protocol))
    errors.extend(_validate_document_bindings(readiness))
    errors.extend(_validate_document_bindings(manuscript))
    if (
        method.get("valid") is not True
        or architecture.get("valid") is not True
        or performance.get("selected_policy", {}).get("batch_size") != 32
        or performance.get("selected_policy", {}).get("freeze_layers") != 24
        or performance.get("comparisons", {}).get(
            "adapter_batch64_vs_batch32_percent_slower", 0.0
        )
        <= 0.0
        or deployment.get("valid") is not True
        or deployment.get("interpretation", {}).get(
            "single_sample_latency_claim_supported"
        )
        is not False
        or stable_audit.get("valid") is not True
    ):
        errors.append("method, figure, performance, or manuscript audit is invalid")
    for name in ("model_source", "training_source"):
        source = method.get(name, {})
        source_path = _resolve(Path(str(source.get("path", ""))))
        if (
            not source_path.is_file()
            or _sha256(source_path) != source.get("sha256")
        ):
            errors.append(f"joint method source binding drift: {name}")

    text_box = ablation["aggregate"]["text_only"]["metrics"][
        "metrics/mAP50-95(B)"
    ]
    text_mask = ablation["aggregate"]["text_only"]["metrics"][
        "metrics/mAP50-95(M)"
    ]
    sensor_box = ablation["aggregate"]["sensor_only"]["metrics"][
        "metrics/mAP50-95(B)"
    ]
    sensor_mask = ablation["aggregate"]["sensor_only"]["metrics"][
        "metrics/mAP50-95(M)"
    ]
    sensor_mask_seed_deltas = ablation["aggregate"]["sensor_only"][
        "paired_full_minus_mode"
    ]["metrics/mAP50-95(M)"]["full_minus_mode_by_seed"]
    if not (
        text_box["mean_minus_rgb_baseline"] > 0.0
        and text_mask["mean_minus_rgb_baseline"] > 0.0
        and sensor_box["mean_minus_rgb_baseline"] > 0.0
        and sensor_mask["mean_minus_rgb_baseline"] > 0.0
        and any(value > 0.0 for value in sensor_mask_seed_deltas)
        and any(value < 0.0 for value in sensor_mask_seed_deltas)
    ):
        errors.append("joint-head interpretation invariants do not match evidence")
    if errors:
        raise ValueError("; ".join(errors))

    full_box = summary["aggregate"]["metrics/mAP50-95(B)"]
    full_mask = summary["aggregate"]["metrics/mAP50-95(M)"]
    index = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "additive_joint_head_extension_to_locked_base_evidence",
        "base_evidence_preserved": True,
        "base_paper_readiness_gate": {
            "internal_paper_evidence_ready": True,
            "full_patent_reproduction_ready": False,
            "note": (
                "The immutable base audit remains hash-bound. New joint-head "
                "capability is established by the additive readiness document."
            ),
        },
        "formal_run_counts": {
            "full_joint_head": 3,
            "text_only": 3,
            "sensor_only": 3,
            "independently_retrained_branch_controls": 6,
        },
        "primary_metrics": {
            "rgb_box_map50_95": text_box["rgb_baseline"],
            "rgb_mask_map50_95": text_mask["rgb_baseline"],
            "text_only_box_map50_95_mean": text_box["mean"],
            "text_only_mask_map50_95_mean": text_mask["mean"],
            "sensor_only_box_map50_95_mean": sensor_box["mean"],
            "sensor_only_mask_map50_95_mean": sensor_mask["mean"],
            "full_box_map50_95_mean": full_box["mean"],
            "full_box_map50_95_sample_std": full_box["sample_std"],
            "full_mask_map50_95_mean": full_mask["mean"],
            "full_mask_map50_95_sample_std": full_mask["sample_std"],
        },
        "audited_interpretation": {
            "both_single_branches_improve_over_rgb_mean": True,
            "full_minus_text_box_percentage_points": (
                text_box["full_mean_minus_mode_mean_percentage_points"]
            ),
            "full_minus_text_mask_percentage_points": (
                text_mask["full_mean_minus_mode_mean_percentage_points"]
            ),
            "full_minus_sensor_box_percentage_points": (
                sensor_box["full_mean_minus_mode_mean_percentage_points"]
            ),
            "full_minus_sensor_mask_percentage_points": (
                sensor_mask["full_mean_minus_mode_mean_percentage_points"]
            ),
            "full_minus_sensor_mask_has_mixed_seed_signs": True,
            "strong_synergy_claim_supported": False,
            "statistical_significance_claim_supported": False,
        },
        "claim_boundaries": {
            "formal_metric_forward_inputs": ["RGB"],
            "qwen_rule_targets_are_training_only": True,
            "external_blind_validation": False,
            "hardware_sensor_measurement": False,
            "rgb_d_input": False,
            "full_patent_reproduction": False,
            "latency_is_amortized_validator_speed": True,
            "single_sample_latency_claim": False,
        },
        "source_bindings": {
            name: _binding(path) for name, path in paths.items()
        },
    }
    _atomic_write(
        output_path,
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# 联合多模态分割头扩展证据索引",
        "",
        "- 状态：完成且审计通过",
        "- 基础证据链：保持原文件与原哈希，不覆盖",
        "- 完整联合头：3 个独立训练种子",
        "- 分支消融：6 个独立重训 run，不是同检查点屏蔽",
        "",
        "| 模型 | Box mAP50-95 | Mask mAP50-95 |",
        "|---|---:|---:|",
        f"| RGB | {text_box['rgb_baseline']:.4f} | {text_mask['rgb_baseline']:.4f} |",
        f"| 仅文本 | {text_box['mean']:.4f} | {text_mask['mean']:.4f} |",
        f"| 仅预测软传感器 | {sensor_box['mean']:.4f} | {sensor_mask['mean']:.4f} |",
        (
            f"| 完整联合头 | {full_box['mean']:.4f}±{full_box['sample_std']:.4f} | "
            f"{full_mask['mean']:.4f}±{full_mask['sample_std']:.4f} |"
        ),
        "",
        (
            "两个单分支的 mAP50-95 均值均高于 RGB 基线。完整头相对仅软传感器"
            "的 mask 均值差仅约 +0.007 个百分点，且三种子差异符号不一致；"
            "该证据不支持强协同或统计显著性主张。"
        ),
        "",
        (
            "正式指标前向只接收 RGB。Qwen/rule-v1 仅提供训练期图像条件软目标；"
            "本扩展不包含 RGB-D、同步硬件传感器或外部盲测。"
        ),
        "",
    ]
    _atomic_write(output_path.with_suffix(".md"), "\n".join(markdown))
    return index


def main() -> None:
    args = parse_args()
    index = build_index(args.output)
    print(
        json.dumps(
            {
                "state": index["state"],
                "valid": index["valid"],
                "output": _resolve(args.output).as_posix(),
                "source_binding_count": len(index["source_bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
