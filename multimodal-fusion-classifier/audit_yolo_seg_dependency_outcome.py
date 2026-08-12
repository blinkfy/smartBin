"""Audit the clean YOLO-seg restart after the dependency interruption."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "incident": "reports/qwen-material/YOLO_SEG_DEPENDENCY_INCIDENT.json",
    "incident_addendum": (
        "reports/qwen-material/YOLO_SEG_DEPENDENCY_INCIDENT_ADDENDUM.json"
    ),
    "yolo_status": "reports/qwen-material/training/yolo_seg.status.json",
    "run_manifest": "reports/qwen-material/yolo-seg/RUN_MANIFEST.json",
    "integrated_protocol": (
        "reports/qwen-material/yolo-seg/"
        "INTEGRATED_EVALUATION_PROTOCOL.json"
    ),
    "export_report": "data/yolo-seg-unique-linked/export_report.json",
    "args": (
        "checkpoints/yolo-seg/yolo11s-seg-qwen-split/args.yaml"
    ),
    "results": (
        "checkpoints/yolo-seg/yolo11s-seg-qwen-split/results.csv"
    ),
    "best_checkpoint": (
        "checkpoints/yolo-seg/yolo11s-seg-qwen-split/weights/best.pt"
    ),
    "last_checkpoint": (
        "checkpoints/yolo-seg/yolo11s-seg-qwen-split/weights/last.pt"
    ),
    "checkpoint_audit": (
        "reports/qwen-material/yolo-seg/CHECKPOINT_AUDIT.json"
    ),
    "detector_evaluation": (
        "reports/qwen-material/yolo-seg/evaluation.json"
    ),
    "integrated_evaluation": (
        "reports/qwen-material/yolo-seg/integrated/evaluation.json"
    ),
    "integrated_protocol_addendum": (
        "reports/qwen-material/yolo-seg/"
        "INTEGRATED_EVALUATION_PROTOCOL_ADDENDUM.json"
    ),
    "training_stdout": (
        "reports/qwen-material/training/yolo_seg.stdout.log"
    ),
    "training_stderr": (
        "reports/qwen-material/training/yolo_seg.stderr.log"
    ),
    "integrated_stdout": (
        "reports/qwen-material/training/"
        "yolo_seg.integrated.recovery.stdout.log"
    ),
    "integrated_stderr": (
        "reports/qwen-material/training/"
        "yolo_seg.integrated.recovery.stderr.log"
    ),
    "path_identity_failed_stderr": (
        "reports/qwen-material/training/interrupted/"
        "scene_path_identity_20260728/"
        "yolo_seg.integrated.failed.stderr.log"
    ),
    "requirements_repro": "requirements-repro.txt",
    "environment": "reports/qwen-material/ENVIRONMENT.md",
    "queue_source": (
        "reports/qwen-material/training/run_yolo_seg_pipeline.ps1"
    ),
    "current_integrity": (
        "reports/qwen-material/SOURCE_IMAGE_INTEGRITY_CURRENT.json"
    ),
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "YOLO_SEG_DEPENDENCY_OUTCOME.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the clean 100-epoch YOLO-seg dependency restart"
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


def build_yolo_dependency_outcome(
    *,
    files: Mapping[str, Path],
) -> dict[str, Any]:
    resolved = {name: _resolve(path) for name, path in files.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "YOLO-seg dependency outcome inputs are missing: "
            + ", ".join(sorted(missing))
        )
    incident = _read_json(resolved["incident"])
    addendum = _read_json(resolved["incident_addendum"])
    status = _read_json(resolved["yolo_status"])
    manifest = _read_json(resolved["run_manifest"])
    protocol = _read_json(resolved["integrated_protocol"])
    export = _read_json(resolved["export_report"])
    checkpoint = _read_json(resolved["checkpoint_audit"])
    detector = _read_json(resolved["detector_evaluation"])
    integrated = _read_json(resolved["integrated_evaluation"])
    integrated_addendum = _read_json(
        resolved["integrated_protocol_addendum"]
    )
    integrity = _read_json(resolved["current_integrity"])
    with resolved["results"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(float(row["epoch"])) for row in rows]
    args_text = resolved["args"].read_text(encoding="utf-8-sig")
    training_stderr = resolved["training_stderr"].read_text(
        encoding="utf-8-sig"
    )
    integrated_stderr = resolved["integrated_stderr"].read_text(
        encoding="utf-8-sig"
    )
    failed_integrated_stderr = resolved[
        "path_identity_failed_stderr"
    ].read_text(encoding="utf-8-sig")
    requirements = resolved["requirements_repro"].read_text(
        encoding="utf-8-sig"
    )
    best_sha = _sha256(resolved["best_checkpoint"])
    last_sha = _sha256(resolved["last_checkpoint"])
    errors: list[str] = []
    if incident.get("valid") is not True or addendum.get("valid") is not True:
        errors.append("dependency incident or immutable addendum is invalid")
    if (
        incident.get("failure", {}).get("completed_epochs") != 1
        or incident.get("failure", {}).get("checkpoint_saved") is not False
        or incident.get("remediation", {}).get("decision")
        != "clean_restart_same_frozen_protocol_from_epoch_1"
    ):
        errors.append("dependency incident restart decision mismatch")
    if status.get("state") != "complete" or status.get("stage") != "all":
        errors.append("YOLO-seg pipeline status is not complete")
    training = manifest.get("training", {})
    if (
        manifest.get("protocol")
        != "independent_yolo11_segmentation_training"
        or training.get("epochs") != 100
        or training.get("batch") != 16
        or training.get("imgsz") != 640
        or training.get("seed") != 42
    ):
        errors.append("frozen YOLO-seg training manifest mismatch")
    queue_binding = protocol.get("inputs", {}).get("queue_source", {})
    if queue_binding.get("sha256") != _sha256(resolved["queue_source"]):
        errors.append("pre-registered YOLO-seg queue source hash drift")
    if epochs != list(range(1, 101)):
        errors.append("YOLO-seg results are not contiguous from epoch 1 to 100")
    for contract in (
        "epochs: 100",
        "batch: 16",
        "imgsz: 640",
        "seed: 42",
        "resume: false",
    ):
        if contract not in args_text:
            errors.append(f"Ultralytics args contract is missing: {contract}")
    if training_stderr.strip():
        errors.append("clean YOLO-seg training stderr is not empty")
    if "Traceback (most recent call last)" in integrated_stderr:
        errors.append("integrated scene stderr contains a traceback")
    if (
        integrated_addendum.get("status")
        != "frozen_after_path_identity_failure_before_recovery_rerun"
        or integrated_addendum.get("valid") is not True
        or "detector result order/path differs from test manifest"
        not in failed_integrated_stderr
        or "image0.jpg" not in failed_integrated_stderr
    ):
        errors.append("integrated path-identity recovery is not bound")
    if (
        checkpoint.get("completed_epochs") != 100
        or checkpoint.get("best", {}).get("sha256") != best_sha
        or checkpoint.get("last", {}).get("sha256") != last_sha
    ):
        errors.append("YOLO-seg checkpoint audit does not bind best/last")
    detector_dataset = detector.get("dataset", {})
    if (
        detector.get("protocol") != "held_out_source_grouped_test"
        or detector_dataset.get("test_source_images") != 570
        or detector_dataset.get("test_objects") != 825
        or detector_dataset.get("class_count") != 19
        or detector.get("checkpoint", {}).get("sha256") != best_sha
    ):
        errors.append("YOLO-seg fixed-test evaluation contract mismatch")
    integrated_dataset = integrated.get("dataset", {})
    if (
        integrated.get("protocol") != "fixed_source_grouped_scene_test"
        or integrated_dataset.get("source_image_count") != 570
        or integrated_dataset.get("ground_truth_object_count") != 825
        or integrated.get("detector", {}).get("sha256") != best_sha
        or len(integrated.get("scenarios", {})) != 2
        or integrated.get("frozen_protocol", {})
        .get("addendum", {})
        .get("sha256")
        != _sha256(resolved["integrated_protocol_addendum"])
    ):
        errors.append("integrated scene evaluation contract mismatch")
    if (
        export.get("valid") is not True
        or export.get("source_image_count") != 5990
        or export.get("object_count") != 8646
        or export.get("copied_image_count") != 0
    ):
        errors.append("YOLO-seg export report mismatch")
    if (
        integrity.get("valid") is not True
        or integrity.get("expected_recovery_state") != "repaired"
    ):
        errors.append("current repaired source-image integrity is invalid")
    for requirement in (
        "polars==1.36.1",
        "nvidia-ml-py==13.610.43",
        "ultralytics-thop==2.0.18",
    ):
        if requirement not in requirements:
            errors.append(f"repaired runtime lock is missing: {requirement}")

    detector_metrics = detector.get("metrics", {})
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "yolo_seg_clean_restart_after_dependency_interruption",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "restart": {
            "failed_run_completed_epochs": incident.get("failure", {}).get(
                "completed_epochs"
            ),
            "failed_run_checkpoint_saved": incident.get("failure", {}).get(
                "checkpoint_saved"
            ),
            "formal_completed_epochs": len(epochs),
            "formal_epoch_sequence": [epochs[0], epochs[-1]] if epochs else [],
            "resume": False,
            "same_frozen_protocol": True,
            "best_checkpoint_sha256": best_sha,
            "last_checkpoint_sha256": last_sha,
        },
        "detector_test": {
            "source_image_count": detector_dataset.get("test_source_images"),
            "object_count": detector_dataset.get("test_objects"),
            "box_map50": detector_metrics.get("metrics/mAP50(B)"),
            "box_map50_95": detector_metrics.get("metrics/mAP50-95(B)"),
            "mask_map50": detector_metrics.get("metrics/mAP50(M)"),
            "mask_map50_95": detector_metrics.get("metrics/mAP50-95(M)"),
        },
        "integrated_scene": {
            "source_image_count": integrated_dataset.get(
                "source_image_count"
            ),
            "ground_truth_object_count": integrated_dataset.get(
                "ground_truth_object_count"
            ),
            "scenario_count": len(integrated.get("scenarios", {})),
        },
        "source_bindings": {
            name: _binding(path)
            for name, path in sorted(resolved.items())
        },
        "errors": errors,
        "boundaries": [
            (
                "The failed epoch-1 run saved no checkpoint and contributes "
                "no model state or metric to the formal result."
            ),
            (
                "The formal detector result comes from a clean contiguous "
                "1-to-100 epoch run under the pre-registered protocol."
            ),
            (
                "The detector is RGB-only and independent of MMFC; integrated "
                "scene evaluation is a two-stage pipeline, not fused-head "
                "patent reproduction."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    restart = report["restart"]
    detector = report["detector_test"]
    return "\n".join(
        [
            "# YOLO-seg 依赖中断结案审计",
            "",
            f"生成时间：`{report['generated_at']}`。",
            "",
            (
                "失败运行完成 1 轮但未保存权重；正式结果来自 `resume=false` 的"
                f"连续 {restart['formal_completed_epochs']} 轮干净重启。"
            ),
            (
                f"固定 test 含 {detector['source_image_count']} 张源图和 "
                f"{detector['object_count']} 个唯一对象；检测、分割和整图数值只"
                "在严格论文表生成后进入结果正文。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError(f"YOLO dependency outcome is immutable: {output}")
    report = build_yolo_dependency_outcome(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        }
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
                "formal_completed_epochs": report["restart"][
                    "formal_completed_epochs"
                ],
                "source_binding_count": len(report["source_bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
