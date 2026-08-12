"""Train and evaluate an independent YOLO11 segmentation model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping

from mmfc.training import PROJECT_ROOT, resolve_project_path
from validate_yolo_seg_export import validate_export


DEFAULT_DATA = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "dataset.yaml"
)
DEFAULT_WEIGHTS = PROJECT_ROOT.parent / "yolo11s-seg.pt"
DEFAULT_CHECKPOINT_PROJECT = PROJECT_ROOT / "checkpoints" / "yolo-seg"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "qwen-material" / "yolo-seg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "训练独立 YOLO11s-seg，并在锁定 source-image 分组测试集上评估"
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--checkpoint-project",
        type=Path,
        default=DEFAULT_CHECKPOINT_PROJECT,
    )
    parser.add_argument(
        "--name",
        default="yolo11s-seg-qwen-split",
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只验证输入并生成预训练清单，不加载或训练模型。",
    )
    return parser.parse_args()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(paths: Iterable[Path], *, root: Path) -> dict[str, Any]:
    """Hash relative names and bytes so label-tree identity is order stable."""

    digest = hashlib.sha256()
    count = 0
    size_bytes = 0
    for path in sorted((item.resolve() for item in paths), key=str):
        relative = path.relative_to(root.resolve()).as_posix()
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        file_size = path.stat().st_size
        size_bytes += file_size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        count += 1
    return {
        "file_count": count,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def _runtime_environment() -> dict[str, Any]:
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


def _resolved(path: Path) -> Path:
    return resolve_project_path(path)


def validate_training_inputs(
    *,
    data_path: Path,
    weights_path: Path,
    checkpoint_project: Path,
    run_name: str,
    report_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    image_size: int,
    workers: int,
) -> tuple[Path, dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if patience < 0:
        raise ValueError("patience must be non-negative")
    if batch_size == 0 or batch_size < -1:
        raise ValueError("batch size must be -1 or greater than zero")
    if image_size <= 0:
        raise ValueError("image size must be greater than zero")
    if workers < 0:
        raise ValueError("workers must be non-negative")
    if not run_name.strip() or Path(run_name).name != run_name:
        raise ValueError("name must be one non-empty path component")
    if not data_path.is_file():
        raise FileNotFoundError(f"missing dataset yaml: {data_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing initial weights: {weights_path}")

    dataset_dir = data_path.parent
    validation = validate_export(dataset_dir)
    if not validation["valid"]:
        raise ValueError(
            "YOLO-seg export validation failed: "
            + "; ".join(validation["errors"][:5])
        )
    if validation["link_mode"] != "symlink":
        raise ValueError("YOLO-seg training requires link_mode=symlink")
    if int(validation["copied_image_count"]) != 0:
        raise ValueError("YOLO-seg export must not contain copied images")

    run_dir = checkpoint_project / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty training directory: {run_dir}"
        )
    test_dir = report_dir / "test"
    if test_dir.exists() and any(test_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty test report directory: {test_dir}"
        )
    return dataset_dir, validation


def build_run_manifest(
    *,
    data_path: Path,
    weights_path: Path,
    checkpoint_project: Path,
    run_name: str,
    report_dir: Path,
    train_args: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_dir = data_path.parent.resolve()
    export_report = dataset_dir / "export_report.json"
    label_paths = dataset_dir.glob("labels/*/*.txt")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "independent_yolo11_segmentation_training",
        "inputs": {
            "training_source": {
                "path": (PROJECT_ROOT / "train_yolo_seg.py").as_posix(),
                "sha256": sha256_file(PROJECT_ROOT / "train_yolo_seg.py"),
            },
            "data_yaml": {
                "path": data_path.resolve().as_posix(),
                "sha256": sha256_file(data_path),
            },
            "export_report": {
                "path": export_report.resolve().as_posix(),
                "sha256": sha256_file(export_report),
            },
            "label_tree": sha256_tree(label_paths, root=dataset_dir),
            "initial_weights": {
                "path": weights_path.resolve().as_posix(),
                "size_bytes": weights_path.stat().st_size,
                "sha256": sha256_file(weights_path),
            },
        },
        "dataset_validation": dict(validation),
        "training": dict(train_args),
        "outputs": {
            "checkpoint_dir": (
                checkpoint_project.resolve() / run_name
            ).as_posix(),
            "report_dir": report_dir.resolve().as_posix(),
        },
        "environment": _runtime_environment(),
        "scope": {
            "task": "instance_segmentation",
            "image_modality": "RGB",
            "source_grouped_split": True,
            "original_images_copied": False,
            "soft_sensor_used_by_detector": False,
        },
    }


def write_or_validate_run_manifest(
    path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Create the frozen manifest once, then require exact semantic reuse."""

    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_comparable = dict(existing)
        proposed_comparable = dict(manifest)
        existing_comparable.pop("created_at_utc", None)
        proposed_comparable.pop("created_at_utc", None)
        if existing_comparable != proposed_comparable:
            raise ValueError(
                f"existing run manifest differs from current inputs: {path}"
            )
        return existing, False
    _atomic_write_text(
        path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return dict(manifest), True


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_training_summary(results_csv: Path) -> dict[str, Any]:
    if not results_csv.is_file():
        raise FileNotFoundError(f"missing Ultralytics results.csv: {results_csv}")
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Ultralytics results.csv is empty")

    def clean(row: Mapping[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, raw in row.items():
            name = key.strip()
            value = raw.strip()
            try:
                result[name] = float(value)
            except ValueError:
                result[name] = value
        return result

    cleaned = [clean(row) for row in rows]
    fitness_keys = (
        "metrics/mAP50-95(M)",
        "metrics/mAP50-95(B)",
    )

    def score(row: Mapping[str, Any]) -> tuple[float, float]:
        return tuple(float(row.get(key, float("-inf"))) for key in fitness_keys)

    best_index = max(range(len(cleaned)), key=lambda index: score(cleaned[index]))
    return {
        "completed_epochs": len(cleaned),
        "last_epoch": cleaned[-1],
        "best_recorded_epoch_index": best_index,
        "best_recorded_epoch": cleaned[best_index],
        "selection_note": (
            "Diagnostic only. Ultralytics best.pt follows its internal fitness; "
            "the recorded epoch maximizes mask mAP50-95 then box mAP50-95."
        ),
    }


def _write_markdown_report(report_dir: Path, report: Mapping[str, Any]) -> None:
    metrics = report["metrics"]
    speed = report.get("speed_ms_per_image", {})

    def pct(key: str) -> str:
        return f"{100.0 * float(metrics[key]):.2f}%"

    lines = [
        "# YOLO11s-seg 目标域固定测试结果",
        "",
        f"生成日期：{datetime.now().date().isoformat()}",
        "",
        "## 结果",
        "",
        f"- 测试源图：{report['dataset']['test_source_images']:,}；",
        f"- 测试 polygon：{report['dataset']['test_objects']:,}；",
        f"- Box Precision / Recall：{pct('metrics/precision(B)')} / "
        f"{pct('metrics/recall(B)')}；",
        f"- Box mAP50 / mAP50-95：{pct('metrics/mAP50(B)')} / "
        f"{pct('metrics/mAP50-95(B)')}；",
        f"- Mask Precision / Recall：{pct('metrics/precision(M)')} / "
        f"{pct('metrics/recall(M)')}；",
        f"- Mask mAP50 / mAP50-95：{pct('metrics/mAP50(M)')} / "
        f"{pct('metrics/mAP50-95(M)')}；",
        "",
        "## 协议",
        "",
        "- 模型：独立 YOLO11s-seg，未修改原版 YOLO 系统；",
        "- 划分：沿用对象分类实验的 source_image_id 分组；",
        "- 输入：RGB 原图符号链接，复制图片数为 0；",
        "- 测试集只用于训练完成后的固定评估；",
        "- 检测器不读取 Qwen/rule 软传感器。",
        "",
        "## 速度边界",
        "",
        (
            f"Ultralytics 批量验证记录的 inference 为 "
            f"{float(speed.get('inference', 0.0)):.3f} ms/图，postprocess 为 "
            f"{float(speed.get('postprocess', 0.0)):.3f} ms/图。该值不包含磁盘读取、"
            "外部传感器、MMFC 复分类和真实设备通信，不能写成端到端部署延迟。"
        ),
        "",
        "## 限制",
        "",
        "该实验只补充目标域 RGB box/mask 证据，不证明 RGB-D、软传感器直接接入检测头、"
        "真实硬件多模态检测或外部数据泛化。固定测试集此前已用于分类模型比较，仍不是"
        "从未访问的外部盲测集。",
        "",
    ]
    _atomic_write_text(report_dir / "REPORT.md", "\n".join(lines))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    data_path = _resolved(args.data)
    weights_path = _resolved(args.weights)
    checkpoint_project = _resolved(args.checkpoint_project)
    report_dir = _resolved(args.report_dir)
    train_args = {
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch": int(args.batch_size),
        "imgsz": int(args.image_size),
        "workers": int(args.workers),
        "device": str(args.device),
        "seed": int(args.seed),
        "deterministic": True,
        "amp": not args.no_amp,
        "optimizer": "auto",
        "cos_lr": True,
        "close_mosaic": 10,
        "cache": False,
        "pretrained": True,
    }
    dataset_dir, validation = validate_training_inputs(
        data_path=data_path,
        weights_path=weights_path,
        checkpoint_project=checkpoint_project,
        run_name=args.name,
        report_dir=report_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        image_size=args.image_size,
        workers=args.workers,
    )
    manifest = build_run_manifest(
        data_path=data_path,
        weights_path=weights_path,
        checkpoint_project=checkpoint_project,
        run_name=args.name,
        report_dir=report_dir,
        train_args=train_args,
        validation=validation,
    )
    manifest_path = report_dir / "RUN_MANIFEST.json"
    manifest, manifest_created = write_or_validate_run_manifest(
        manifest_path,
        manifest,
    )
    if args.validate_only:
        return {
            "state": "validated",
            "manifest": manifest_path.as_posix(),
            "manifest_created": manifest_created,
            "dataset": dataset_dir.as_posix(),
        }

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_path),
        project=str(checkpoint_project),
        name=args.name,
        exist_ok=False,
        plots=True,
        save=True,
        verbose=True,
        **train_args,
    )
    run_dir = checkpoint_project / args.name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("YOLO training did not create best.pt and last.pt")
    training_summary = _read_training_summary(run_dir / "results.csv")
    checkpoint_audit = {
        "schema_version": 1,
        **training_summary,
        "best": {
            "path": best_path.resolve().as_posix(),
            "size_bytes": best_path.stat().st_size,
            "sha256": sha256_file(best_path),
        },
        "last": {
            "path": last_path.resolve().as_posix(),
            "size_bytes": last_path.stat().st_size,
            "sha256": sha256_file(last_path),
        },
    }
    _atomic_write_text(
        report_dir / "CHECKPOINT_AUDIT.json",
        json.dumps(checkpoint_audit, ensure_ascii=False, indent=2) + "\n",
    )

    best_model = YOLO(str(best_path))
    validation_metrics = best_model.val(
        data=str(data_path),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        device=str(args.device),
        workers=args.workers,
        project=str(report_dir),
        name="test",
        exist_ok=False,
        plots=True,
        save_json=False,
        verbose=True,
    )
    export_report = json.loads(
        (dataset_dir / "export_report.json").read_text(encoding="utf-8")
    )
    evaluation = {
        "schema_version": 1,
        "protocol": "held_out_source_grouped_test",
        "metrics": _json_safe(validation_metrics.results_dict),
        "per_class": _json_safe(validation_metrics.summary(decimals=8)),
        "speed_ms_per_image": _json_safe(validation_metrics.speed),
        "dataset": {
            "yaml": data_path.resolve().as_posix(),
            "test_source_images": int(
                export_report["split_source_counts"]["test"]
            ),
            "test_objects": int(export_report["split_object_counts"]["test"]),
            "class_count": len(export_report["class_names"]),
            "source_grouped_split": True,
            "copied_image_count": int(export_report["copied_image_count"]),
        },
        "checkpoint": checkpoint_audit["best"],
        "runtime": _runtime_environment(),
        "output_dir": Path(validation_metrics.save_dir).resolve().as_posix(),
        "limitations": [
            "RGB target-domain segmentation only; no depth or hardware sensor input.",
            "Metrics do not include downstream MMFC classification.",
            "The fixed test split is not an external never-seen deployment set.",
        ],
    }
    _atomic_write_text(
        report_dir / "evaluation.json",
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
    )
    _write_markdown_report(report_dir, evaluation)
    return {
        "state": "complete",
        "checkpoint": checkpoint_audit["best"],
        "metrics": evaluation["metrics"],
        "report_dir": report_dir.as_posix(),
    }


def main() -> None:
    result = run_training(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
