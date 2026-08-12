"""Freeze the stratified-evaluation protocol amendment."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "original_protocol": (
        "reports/qwen-material/yolo-seg/"
        "STRATIFIED_EVALUATION_PROTOCOL.json"
    ),
    "integrated_protocol": (
        "reports/qwen-material/yolo-seg/"
        "INTEGRATED_EVALUATION_PROTOCOL.json"
    ),
    "dependency_protocol_addendum": (
        "reports/qwen-material/yolo-seg/"
        "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
    ),
    "integrated_report": (
        "reports/qwen-material/yolo-seg/integrated/evaluation.json"
    ),
    "original_strata_source": (
        "reports/qwen-material/training/interrupted/"
        "scene_path_identity_20260728/"
        "evaluate_scene_strata.pre_addendum.py"
    ),
    "revised_strata_source": "evaluate_scene_strata.py",
    "original_integrated_source": (
        "reports/qwen-material/training/interrupted/"
        "scene_path_identity_20260728/"
        "evaluate_scene_pipeline.pre_path_fix.py"
    ),
    "revised_integrated_source": "evaluate_scene_pipeline.py",
    "final_evidence_queue_source": (
        "reports/qwen-material/training/run_final_evidence_audit.ps1"
    ),
    "recovery_queue_source": (
        "reports/qwen-material/training/"
        "run_scene_path_recovery_continuation.ps1"
    ),
    "dataset_export": "data/yolo-seg-unique-linked/export_report.json",
    "builder_source": "build_scene_strata_protocol_addendum.py",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "yolo-seg"
    / "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json"
)
DEFAULT_STRATIFIED_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "yolo-seg"
    / "integrated"
    / "stratified"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the scene-strata protocol addendum"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _protocol_hash(protocol: Mapping[str, Any], relative_path: str) -> str:
    for entry in protocol.get("input_hashes", []):
        if Path(str(entry["path"])).as_posix() == relative_path:
            return str(entry["sha256"]).lower()
    return ""


def build_scene_strata_addendum(
    *,
    files: Mapping[str, Path],
    stratified_output: Path,
) -> dict[str, Any]:
    missing_keys = sorted(set(DEFAULT_FILES) - set(files))
    if missing_keys:
        raise FileNotFoundError(
            "scene-strata addendum source map is incomplete: "
            + ", ".join(missing_keys)
        )
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "scene-strata addendum inputs are missing: "
            + ", ".join(sorted(missing))
        )

    protocol = _read_json(resolved["original_protocol"])
    integrated_protocol = _read_json(resolved["integrated_protocol"])
    dependency = _read_json(resolved["dependency_protocol_addendum"])
    integrated = _read_json(resolved["integrated_report"])
    revised_source = resolved["revised_strata_source"].read_text(
        encoding="utf-8-sig"
    )
    final_queue = resolved["final_evidence_queue_source"].read_text(
        encoding="utf-8-sig"
    )
    hashes = {
        name: _sha256(path)
        for name, path in resolved.items()
    }
    errors: list[str] = []

    if (
        protocol.get("status") != "frozen_before_detector_training"
        or protocol.get("protocol")
        != "preregistered_scene_stratified_analysis"
    ):
        errors.append("original stratified protocol is not frozen")
    if (
        _protocol_hash(protocol, "evaluate_scene_strata.py")
        != hashes["original_strata_source"]
        or _protocol_hash(protocol, "evaluate_scene_pipeline.py")
        != hashes["original_integrated_source"]
    ):
        errors.append("original stratified protocol source hashes mismatch")
    if (
        dependency.get("status")
        != "frozen_after_path_identity_failure_before_recovery_rerun"
        or dependency.get("valid") is not True
        or dependency.get("original_protocol", {}).get("sha256")
        != hashes["integrated_protocol"]
    ):
        errors.append("integrated protocol addendum is not valid")
    dependency_fix = dependency.get("implementation_fix", {})
    if (
        dependency_fix.get("original_evaluation_source", {}).get("sha256")
        != hashes["original_integrated_source"]
        or dependency_fix.get("revised_evaluation_source", {}).get("sha256")
        != hashes["revised_integrated_source"]
    ):
        errors.append("integrated source transition is not bound")
    integrated_binding = (
        integrated.get("frozen_protocol", {}).get("addendum", {})
    )
    if (
        integrated.get("protocol") != "fixed_source_grouped_scene_test"
        or integrated_binding.get("sha256")
        != hashes["dependency_protocol_addendum"]
        or integrated_binding.get("evaluation_source_sha256")
        != hashes["revised_integrated_source"]
    ):
        errors.append("integrated report does not bind the dependency addendum")
    if hashes["original_strata_source"] == hashes["revised_strata_source"]:
        errors.append("revised stratified source did not change")
    required_tokens = (
        "STRATIFIED_EVALUATION_PROTOCOL_ADDENDUM.json",
        "_validate_protocol_addenda",
        '"evaluate_scene_pipeline.py": integrated_source_hash',
        '"evaluate_scene_strata.py": stratified_source_hash',
        '"stratified_protocol_addendum"',
    )
    if any(token not in revised_source for token in required_tokens):
        errors.append("revised stratified source lacks the authorized bindings")

    evaluation = protocol.get("evaluation", {})
    bootstrap = protocol.get("bootstrap", {})
    strata = protocol.get("strata", {})
    expected_row_count = (
        len(evaluation.get("confidence_thresholds", []))
        * len(evaluation.get("prediction_sources", []))
        * sum(
            len(definition.get("expected_support", {}))
            for definition in strata.values()
        )
    )
    if (
        evaluation.get("confidence_thresholds") != [0.25, 0.8]
        or float(evaluation.get("match_iou", -1)) != 0.5
        or evaluation.get("prediction_sources")
        != ["detector", "missing", "rule-v1"]
        or int(bootstrap.get("repeats", -1)) != 2000
        or int(bootstrap.get("seed", -1)) != 20260727
        or expected_row_count != 42
    ):
        errors.append("stratified analytical invariants changed")
    if (
        "evaluate_scene_strata.py" not in final_queue
        or "scene-strata evaluation failed" not in final_queue
    ):
        errors.append("final evidence queue does not bind stratified evaluation")
    output_dir = _resolve(stratified_output)
    if output_dir.exists() and any(output_dir.iterdir()):
        errors.append("stratified output directory is not empty before run")

    return {
        "schema_version": 1,
        "status": (
            "frozen_after_integrated_path_addendum_before_stratified_run"
            if not errors
            else "invalid"
        ),
        "valid": not errors,
        "frozen_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "original_protocol": _binding(resolved["original_protocol"]),
        "dependency_protocol_addendum": _binding(
            resolved["dependency_protocol_addendum"]
        ),
        "implementation_fix": {
            "allowed_changes": [
                "consume_integrated_path_identity_protocol_addendum",
                "replace_two_frozen_source_hashes_with_bound_revisions",
            ],
            "original_strata_source": _binding(
                resolved["original_strata_source"]
            ),
            "revised_strata_source": _binding(
                resolved["revised_strata_source"]
            ),
            "original_integrated_source": _binding(
                resolved["original_integrated_source"]
            ),
            "revised_integrated_source": _binding(
                resolved["revised_integrated_source"]
            ),
        },
        "invariants": {
            "dataset_export_unchanged": True,
            "integrated_predictions_unchanged": True,
            "strata_definitions_unchanged": True,
            "confidence_thresholds": [0.25, 0.8],
            "match_iou": 0.5,
            "prediction_sources": ["detector", "missing", "rule-v1"],
            "bootstrap_repeats": 2000,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 20260727,
            "expected_row_count": 42,
            "metric_definitions_unchanged": True,
        },
        "source_bindings": {
            name: _binding(path)
            for name, path in sorted(resolved.items())
        },
        "errors": errors,
        "boundaries": [
            (
                "The amendment only authorizes the stratified evaluator to "
                "consume the earlier integrated path-identity amendment."
            ),
            (
                "Dataset, integrated predictions, strata, thresholds, "
                "matching, metrics, bootstrap settings, and output schema "
                "remain unchanged."
            ),
            (
                "The amendment was frozen before the stratified output "
                "directory contained any result."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    fix = report["implementation_fix"]
    return "\n".join(
        [
            "# 整图分层评估协议补充",
            "",
            f"冻结时间：`{report['frozen_at']}`。",
            "",
            (
                "整图推理因 Ultralytics 路径标识兼容修订而产生了已审计的源码"
                "哈希变化；本补充件仅允许分层评估器读取该先行补充协议，并用"
                "绑定后的两份修订源码哈希完成输入校验。"
            ),
            (
                "分层评估器修订前后哈希分别为 "
                f"`{fix['original_strata_source']['sha256']}` 与 "
                f"`{fix['revised_strata_source']['sha256']}`。"
            ),
            (
                "数据、整图预测、分层定义、置信度阈值、匹配规则、指标、"
                "bootstrap 及 42 行预期输出均保持冻结。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError(f"scene-strata addendum is immutable: {output}")
    report = build_scene_strata_addendum(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        },
        stratified_output=DEFAULT_STRATIFIED_OUTPUT,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "valid": report["valid"],
                "source_binding_count": len(report["source_bindings"]),
                "original_source_sha256": report["implementation_fix"][
                    "original_strata_source"
                ]["sha256"],
                "revised_source_sha256": report["implementation_fix"][
                    "revised_strata_source"
                ]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
