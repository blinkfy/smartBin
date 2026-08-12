"""Freeze the protocol amendment for Ultralytics synthetic list paths."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
INCIDENT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "training"
    / "interrupted"
    / "scene_path_identity_20260728"
)
DEFAULT_FILES = {
    "original_protocol": (
        "reports/qwen-material/yolo-seg/"
        "INTEGRATED_EVALUATION_PROTOCOL.json"
    ),
    "original_evaluation_source": (
        "reports/qwen-material/training/interrupted/"
        "scene_path_identity_20260728/"
        "evaluate_scene_pipeline.pre_path_fix.py"
    ),
    "revised_evaluation_source": "evaluate_scene_pipeline.py",
    "recovery_queue_source": (
        "reports/qwen-material/training/"
        "run_scene_path_recovery_continuation.ps1"
    ),
    "frozen_queue_source": (
        "reports/qwen-material/training/run_yolo_seg_pipeline.ps1"
    ),
    "failed_integrated_stdout": (
        "reports/qwen-material/training/interrupted/"
        "scene_path_identity_20260728/"
        "yolo_seg.integrated.failed.stdout.log"
    ),
    "failed_integrated_stderr": (
        "reports/qwen-material/training/interrupted/"
        "scene_path_identity_20260728/"
        "yolo_seg.integrated.failed.stderr.log"
    ),
    "failed_yolo_status": (
        "reports/qwen-material/training/interrupted/"
        "scene_path_identity_20260728/yolo_seg.failed.status.json"
    ),
    "failed_continuation_status": (
        "reports/qwen-material/training/interrupted/"
        "scene_path_identity_20260728/"
        "source_repair_continuation.failed.status.json"
    ),
    "checkpoint_audit": (
        "reports/qwen-material/yolo-seg/CHECKPOINT_AUDIT.json"
    ),
    "detector_evaluation": (
        "reports/qwen-material/yolo-seg/evaluation.json"
    ),
    "requirements_repro": "requirements-repro.txt",
    "environment": "reports/qwen-material/ENVIRONMENT.md",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "yolo-seg"
    / "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the integrated path-identity protocol addendum"
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


def build_scene_path_addendum(
    *,
    files: Mapping[str, Path],
    integrated_output: Path,
) -> dict[str, Any]:
    missing_keys = sorted(set(DEFAULT_FILES) - set(files))
    if missing_keys:
        raise FileNotFoundError(
            "scene path addendum source map is incomplete: "
            + ", ".join(missing_keys)
        )
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "scene path addendum inputs are missing: "
            + ", ".join(sorted(missing))
        )
    output_dir = _resolve(integrated_output)
    protocol = _read_json(resolved["original_protocol"])
    failed_yolo = _read_json(resolved["failed_yolo_status"])
    failed_continuation = _read_json(
        resolved["failed_continuation_status"]
    )
    checkpoint = _read_json(resolved["checkpoint_audit"])
    detector = _read_json(resolved["detector_evaluation"])
    stderr = resolved["failed_integrated_stderr"].read_text(
        encoding="utf-8-sig"
    )
    revised = resolved["revised_evaluation_source"].read_text(
        encoding="utf-8-sig"
    )
    recovery = resolved["recovery_queue_source"].read_text(
        encoding="utf-8-sig"
    )
    original_hash = _sha256(resolved["original_evaluation_source"])
    revised_hash = _sha256(resolved["revised_evaluation_source"])
    errors: list[str] = []
    if protocol.get("status") != "frozen_before_detector_training":
        errors.append("original integrated protocol is not frozen")
    inputs = protocol.get("inputs", {})
    if (
        inputs.get("evaluation_source", {}).get("sha256") != original_hash
        or inputs.get("queue_source", {}).get("sha256")
        != _sha256(resolved["frozen_queue_source"])
    ):
        errors.append("original protocol source hashes do not match")
    if original_hash == revised_hash:
        errors.append("revised evaluation source did not change")
    if (
        "_result_path_matches" not in revised
        or "image{result_index}.jpg" not in revised
        or "validate_protocol_addendum" not in revised
    ):
        errors.append("revised source lacks the authorized path-index fix")
    if (
        "evaluate_scene_pipeline.py" not in recovery
        or "--confidence-thresholds" not in recovery
        or '"0.25", "0.8"' not in recovery
        or '"--bootstrap-repeats", "2000"' not in recovery
    ):
        errors.append("recovery queue does not preserve integrated arguments")
    if (
        "detector result order/path differs from test manifest" not in stderr
        or "image0.jpg" not in stderr
    ):
        errors.append("archived integrated traceback mismatch")
    if (
        failed_yolo.get("state") != "failed"
        or "integrated scene evaluation failed" not in str(
            failed_yolo.get("message", "")
        )
        or failed_continuation.get("state") != "failed"
        or "yolo_seg_pipeline failed" not in str(
            failed_continuation.get("message", "")
        )
    ):
        errors.append("archived failed statuses do not bind the incident")
    if (
        checkpoint.get("completed_epochs") != 100
        or detector.get("protocol") != "held_out_source_grouped_test"
        or detector.get("dataset", {}).get("test_source_images") != 570
        or detector.get("dataset", {}).get("test_objects") != 825
    ):
        errors.append("completed detector evidence is not valid")
    if output_dir.exists() and any(output_dir.iterdir()):
        errors.append("integrated output directory is not empty before rerun")

    return {
        "schema_version": 1,
        "status": (
            "frozen_after_path_identity_failure_before_recovery_rerun"
            if not errors
            else "invalid"
        ),
        "valid": not errors,
        "frozen_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "original_protocol": _binding(resolved["original_protocol"]),
        "incident": {
            "stage": "integrated_scene_evaluation",
            "exception": "ValueError",
            "observed_result_path": "image0.jpg",
            "cause": (
                "ultralytics_8_4_70_list_autocast_drops_pil_filename"
            ),
            "detector_training_complete_before_incident": True,
            "integrated_output_created": False,
        },
        "implementation_fix": {
            "allowed_change": (
                "accept_ultralytics_synthetic_list_index_for_order_validation"
            ),
            "original_evaluation_source": _binding(
                resolved["original_evaluation_source"]
            ),
            "revised_evaluation_source": _binding(
                resolved["revised_evaluation_source"]
            ),
            "recovery_queue_source": _binding(
                resolved["recovery_queue_source"]
            ),
            "frozen_queue_source_unchanged": _binding(
                resolved["frozen_queue_source"]
            ),
        },
        "invariants": {
            "dataset_export_unchanged": True,
            "detector_checkpoint_unchanged": True,
            "classifier_checkpoint_unchanged": True,
            "confidence_thresholds": [0.25, 0.8],
            "nms_iou": 0.5,
            "match_iou": 0.5,
            "bootstrap_repeats": 2000,
            "bootstrap_seed": 20260727,
            "metric_definitions_unchanged": True,
        },
        "source_bindings": {
            name: _binding(path)
            for name, path in sorted(resolved.items())
        },
        "errors": errors,
        "boundaries": [
            (
                "The amendment changes only result-path identity validation "
                "for Ultralytics synthetic image{index}.jpg names."
            ),
            (
                "Detector training, checkpoints, predictions, matching, "
                "thresholds, metrics, bootstrap settings, and MMFC inference "
                "remain unchanged."
            ),
            (
                "The failed integrated attempt produced no report directory "
                "and contributes no metric to the formal result."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    fix = report["implementation_fix"]
    return "\n".join(
        [
            "# 整图评估路径标识协议补充",
            "",
            f"冻结时间：`{report['frozen_at']}`。",
            "",
            (
                "Ultralytics 8.4.70 将路径列表转为 PIL 图像时丢失 filename，"
                "首个结果因而使用合成名 `image0.jpg`，旧路径等值校验在产生任何"
                "整图报告前退出。"
            ),
            (
                "唯一授权修订是按 `image{index}.jpg` 的合成索引校验列表顺序；"
                "数据、两类检查点、阈值、匹配、指标和 bootstrap 均保持冻结。"
            ),
            (
                "修订前后评估源码哈希分别为 "
                f"`{fix['original_evaluation_source']['sha256']}` 与 "
                f"`{fix['revised_evaluation_source']['sha256']}`。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError(f"scene path addendum is immutable: {output}")
    report = build_scene_path_addendum(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        },
        integrated_output=(
            PROJECT_ROOT
            / "reports"
            / "qwen-material"
            / "yolo-seg"
            / "integrated"
        ),
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
                    "original_evaluation_source"
                ]["sha256"],
                "revised_source_sha256": report["implementation_fix"][
                    "revised_evaluation_source"
                ]["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
