"""Capture the failed YOLO-seg epoch-1 run caused by a missing dependency."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, requires, version
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = (
    PROJECT_ROOT
    / "checkpoints"
    / "interrupted"
    / "yolo_seg_missing_polars_epoch1_20260728"
)
DEFAULT_FILES = {
    "formal_status": (
        "reports/qwen-material/training/formal_ablations.status.json"
    ),
    "formal_reporting_status": (
        "reports/qwen-material/training/"
        "formal_ablations.reporting_status.json"
    ),
    "yolo_status": "reports/qwen-material/training/yolo_seg.status.json",
    "continuation_status": (
        "reports/qwen-material/training/"
        "source_repair_continuation.status.json"
    ),
    "yolo_stdout": "reports/qwen-material/training/yolo_seg.stdout.log",
    "yolo_stderr": "reports/qwen-material/training/yolo_seg.stderr.log",
    "continuation_yolo_stdout": (
        "reports/qwen-material/training/"
        "source_repair_continuation.yolo_seg_pipeline.stdout.log"
    ),
    "continuation_yolo_stderr": (
        "reports/qwen-material/training/"
        "source_repair_continuation.yolo_seg_pipeline.stderr.log"
    ),
    "run_manifest": "reports/qwen-material/yolo-seg/RUN_MANIFEST.json",
    "integrated_protocol": (
        "reports/qwen-material/yolo-seg/"
        "INTEGRATED_EVALUATION_PROTOCOL.json"
    ),
    "stratified_protocol": (
        "reports/qwen-material/yolo-seg/"
        "STRATIFIED_EVALUATION_PROTOCOL.json"
    ),
    "export_report": "data/yolo-seg-unique-linked/export_report.json",
    "requirements": "requirements.txt",
    "requirements_repro": "requirements-repro.txt",
    "environment": "reports/qwen-material/ENVIRONMENT.md",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "YOLO_SEG_DEPENDENCY_INCIDENT.json"
)
REQUIRED_PACKAGES = {
    "ultralytics": "8.4.70",
    "polars": "1.36.1",
    "nvidia-ml-py": "13.610.43",
    "ultralytics-thop": "2.0.18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the missing-Polars YOLO-seg interruption"
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
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


def runtime_dependency_state() -> tuple[dict[str, str | None], list[str]]:
    versions: dict[str, str | None] = {}
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    try:
        ultralytics_requirements = requires("ultralytics") or []
    except PackageNotFoundError:
        ultralytics_requirements = []
    relevant = sorted(
        requirement
        for requirement in ultralytics_requirements
        if any(
            package in requirement.lower()
            for package in ("polars", "nvidia-ml-py", "ultralytics-thop")
        )
    )
    return versions, relevant


def build_dependency_incident(
    *,
    files: Mapping[str, Path],
    archive_dir: Path,
    dependency_versions: Mapping[str, str | None],
    ultralytics_requirements: list[str],
) -> dict[str, Any]:
    resolved = {name: _resolve(path) for name, path in files.items()}
    archive = _resolve(archive_dir)
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "YOLO-seg incident inputs are missing: "
            + ", ".join(sorted(missing))
        )
    if not archive.is_dir():
        raise FileNotFoundError(f"YOLO-seg incident archive is missing: {archive}")

    formal_status = _read_json(resolved["formal_status"])
    reporting_status = _read_json(resolved["formal_reporting_status"])
    yolo_status = _read_json(resolved["yolo_status"])
    continuation_status = _read_json(resolved["continuation_status"])
    stderr = resolved["yolo_stderr"].read_text(encoding="utf-8-sig")
    archived_files = sorted(path for path in archive.rglob("*") if path.is_file())
    results_path = archive / "results.csv"
    args_path = archive / "args.yaml"
    rows: list[dict[str, str]] = []
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    weight_files = [
        path
        for path in archived_files
        if path.suffix.lower() == ".pt" or "weights" in path.parts
    ]
    fresh_run_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "yolo-seg"
        / "yolo11s-seg-qwen-split"
    )

    errors: list[str] = []
    if formal_status.get("state") != "complete":
        errors.append("formal ablation queue was not complete before YOLO-seg")
    if reporting_status.get("state") != "complete":
        errors.append("formal ablation reporting was not complete before YOLO-seg")
    if yolo_status.get("state") != "failed":
        errors.append("YOLO-seg status does not record the failed run")
    if (
        continuation_status.get("state") != "failed"
        or "yolo_seg_pipeline" not in str(continuation_status.get("message", ""))
    ):
        errors.append("outer continuation does not bind the YOLO-seg failure")
    if "ModuleNotFoundError: No module named 'polars'" not in stderr:
        errors.append("traceback does not identify the missing Polars dependency")
    if "read_results_csv" not in stderr or "save_model" not in stderr:
        errors.append("traceback is not from the epoch-save results reader")
    if len(rows) != 1 or rows[0].get("epoch") != "1":
        errors.append("archived results.csv is not exactly the completed epoch 1")
    if not args_path.is_file():
        errors.append("archived Ultralytics args.yaml is missing")
    if weight_files:
        errors.append("failed epoch-1 archive unexpectedly contains model weights")
    if fresh_run_dir.exists() and any(fresh_run_dir.iterdir()):
        errors.append("fresh YOLO-seg checkpoint directory is not empty")
    for package, expected in REQUIRED_PACKAGES.items():
        if dependency_versions.get(package) != expected:
            errors.append(
                f"runtime dependency {package} is "
                f"{dependency_versions.get(package)!r}, expected {expected!r}"
            )
    expected_metadata = {
        "polars>=0.20.0",
        "nvidia-ml-py>=12.0.0",
        "ultralytics-thop>=2.0.18",
    }
    if not expected_metadata.issubset(set(ultralytics_requirements)):
        errors.append("Ultralytics dependency metadata is incomplete")

    source_bindings = {
        name: _binding(path)
        for name, path in sorted(resolved.items())
    }
    archive_bindings = {
        path.relative_to(archive).as_posix(): _binding(path)
        for path in archived_files
    }
    return {
        "schema_version": 1,
        "state": "complete" if not errors else "invalid",
        "valid": not errors,
        "scope": "yolo_seg_missing_polars_epoch1_interruption",
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "failure": {
            "stage": "training_and_test",
            "completed_epochs": len(rows),
            "last_completed_epoch": (
                int(rows[-1]["epoch"]) if rows and rows[-1].get("epoch") else None
            ),
            "checkpoint_saved": bool(weight_files),
            "exception": "ModuleNotFoundError",
            "missing_module": "polars",
            "failure_point": "Ultralytics BaseTrainer.save_model/read_results_csv",
        },
        "remediation": {
            "decision": "clean_restart_same_frozen_protocol_from_epoch_1",
            "partial_run_archive": _display_path(archive),
            "archived_file_count": len(archived_files),
            "fresh_checkpoint_directory_empty": not (
                fresh_run_dir.exists() and any(fresh_run_dir.iterdir())
            ),
            "runtime_dependency_versions": dict(dependency_versions),
            "ultralytics_dependency_requirements": ultralytics_requirements,
        },
        "source_bindings": source_bindings,
        "archive_bindings": archive_bindings,
        "errors": errors,
        "boundaries": [
            (
                "The failed run completed one training/validation epoch but "
                "saved no model checkpoint, so it cannot be resumed or used "
                "as paper evidence."
            ),
            (
                "The clean restart keeps the pre-registered dataset, initial "
                "weights, seed, hyperparameters, and evaluation protocol."
            ),
            (
                "Installing declared runtime dependencies changes only the "
                "Python environment; it does not alter source images, labels, "
                "manifests, model code, or frozen experiment arguments."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    failure = report["failure"]
    remediation = report["remediation"]
    return "\n".join(
        [
            "# YOLO-seg 运行时依赖中断记录",
            "",
            f"捕获时间：`{report['captured_at']}`。",
            "",
            (
                "独立 YOLO-seg 正式训练完成第 "
                f"{failure['last_completed_epoch']} 轮训练与验证后，在 "
                "`BaseTrainer.save_model/read_results_csv` 保存路径因缺少 "
                "`polars` 退出。归档中没有模型权重，因此该次运行不进入论文结果。"
            ),
            (
                f"失败目录的 {remediation['archived_file_count']} 个文件已移至 "
                f"`{remediation['partial_run_archive']}`，正式检查点目录已清空。"
            ),
            (
                "补齐并锁定 Ultralytics 包元数据要求的 Polars、"
                "nvidia-ml-py 与 ultralytics-thop 后，按完全相同的冻结协议从"
                "第 1 轮重新训练。"
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output = _resolve(args.output)
    markdown = output.with_suffix(".md")
    if output.exists() or markdown.exists():
        raise FileExistsError(
            f"YOLO-seg dependency incident is immutable: {output}"
        )
    dependency_versions, metadata_requirements = runtime_dependency_state()
    report = build_dependency_incident(
        files={
            name: PROJECT_ROOT / relative
            for name, relative in DEFAULT_FILES.items()
        },
        archive_dir=args.archive,
        dependency_versions=dependency_versions,
        ultralytics_requirements=metadata_requirements,
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
                "completed_epochs": report["failure"]["completed_epochs"],
                "checkpoint_saved": report["failure"]["checkpoint_saved"],
                "source_binding_count": len(report["source_bindings"]),
                "archive_binding_count": len(report["archive_bindings"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
