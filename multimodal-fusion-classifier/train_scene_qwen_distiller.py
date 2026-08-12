"""Train an RGB-only student for object-Qwen-derived scene fields."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mmfc.model import PROJECT_ROOT
from mmfc.scene_distillation import (
    SceneQwenDistiller,
    SceneTeacherDataset,
    distillation_loss,
    prediction_metrics,
)


DEFAULT_TEACHER = (
    PROJECT_ROOT
    / "data"
    / "object-qwen-scene-teacher"
)
DEFAULT_IMAGES = (
    PROJECT_ROOT / "data" / "yolo-seg-unique-linked" / "images"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "checkpoints" / "scene-qwen-distiller-v1"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "object-qwen-scene-distillation"
    / "TRAINING.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RGB-only object-Qwen scene distiller"
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (
        path.resolve()
        if path.is_absolute()
        else (PROJECT_ROOT / path).resolve()
    )


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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: SceneTeacherDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(42)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": False,
        "generator": generator,
    }
    if workers:
        kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )
    return DataLoader(**kwargs)


@torch.inference_mode()
def _evaluate(
    model: SceneQwenDistiller,
    loader: DataLoader,
    *,
    device: torch.device,
    train_mean_grid: np.ndarray,
) -> tuple[float, dict[str, float]]:
    model.eval()
    predictions: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    coverages: list[np.ndarray] = []
    loss_sum = 0.0
    sample_count = 0
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True)
        coverage = batch["coverage"].to(device, non_blocking=True)
        with amp_context:
            prediction = model(images)
            loss, _ = distillation_loss(prediction, teacher, coverage)
        size = int(images.shape[0])
        loss_sum += float(loss) * size
        sample_count += size
        predictions.append(prediction.float().cpu().numpy())
        teachers.append(teacher.float().cpu().numpy())
        coverages.append(coverage.float().cpu().numpy())
    metrics = prediction_metrics(
        np.concatenate(predictions),
        np.concatenate(teachers),
        np.concatenate(coverages),
        train_mean_grid=train_mean_grid,
    )
    metrics["composite_loss"] = loss_sum / max(1, sample_count)
    return metrics["composite_loss"], metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("batch_size", "epochs", "patience", "progress_every"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.workers) < 0 or int(args.workers) > 16:
        raise ValueError("--workers must be in range 0..16")
    if int(args.image_size) < 64:
        raise ValueError("--image-size must be at least 64")
    if float(args.learning_rate) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("learning rate must be positive and decay non-negative")

    teacher_root = _resolve(args.teacher)
    images_root = _resolve(args.images_root)
    output = _resolve(args.output)
    report_path = _resolve(args.report)
    protocol_path = _resolve(args.protocol)
    teacher_paths = {
        split: teacher_root / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }
    for path in (*teacher_paths.values(), protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    runtime = protocol.get("runtime", {})
    if (
        protocol.get("valid") is not True
        or protocol.get("state") != "frozen_before_training"
        or runtime.get("seed") != int(args.seed)
        or runtime.get("image_size") != int(args.image_size)
        or runtime.get("batch_size") != int(args.batch_size)
        or runtime.get("workers") != int(args.workers)
        or runtime.get("epochs") != int(args.epochs)
        or runtime.get("learning_rate") != float(args.learning_rate)
        or runtime.get("weight_decay") != float(args.weight_decay)
        or runtime.get("patience") != int(args.patience)
    ):
        raise ValueError("training arguments differ from frozen protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _seed_everything(int(args.seed))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_dataset = SceneTeacherDataset(
        teacher_paths["train"],
        images_root,
        split="train",
        image_size=int(args.image_size),
    )
    val_dataset = SceneTeacherDataset(
        teacher_paths["val"],
        images_root,
        split="val",
        image_size=int(args.image_size),
    )
    train_loader = _loader(
        train_dataset,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        shuffle=True,
    )
    val_loader = _loader(
        val_dataset,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        shuffle=False,
    )
    train_mean_grid = np.mean(
        np.stack([record["teacher"] for record in train_dataset.records]),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    model = SceneQwenDistiller().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.epochs))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.monotonic()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for batch in train_loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
                memory_format=(
                    torch.channels_last
                    if device.type == "cuda"
                    else torch.contiguous_format
                ),
            )
            teacher_batch = batch["teacher"].to(
                device, non_blocking=True
            )
            coverage = batch["coverage"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                prediction = model(images)
                loss, _ = distillation_loss(
                    prediction, teacher_batch, coverage
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            size = int(images.shape[0])
            train_loss += float(loss.detach()) * size
            train_samples += size
        val_loss, val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            train_mean_grid=train_mean_grid,
        )
        row = {
            "epoch": epoch,
            "train_composite_loss": train_loss / max(1, train_samples),
            "val_composite_loss": val_loss,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_elementwise_pearson": val_metrics[
                "elementwise_pearson"
            ],
            "val_elementwise_spearman": val_metrics[
                "elementwise_spearman"
            ],
            "val_positive_iou_at_0_05": val_metrics[
                "positive_iou_at_0_05"
            ],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        checkpoint = {
            "schema_version": 1,
            "architecture": "SceneQwenDistiller",
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_composite_loss": val_loss,
            "val_metrics": val_metrics,
            "train_mean_grid": torch.from_numpy(train_mean_grid),
            "image_size": int(args.image_size),
            "protocol_sha256": _sha256(protocol_path),
        }
        torch.save(checkpoint, output / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale_epochs += 1
        scheduler.step()
        if (
            epoch == 1
            or epoch % int(args.progress_every) == 0
            or stale_epochs >= int(args.patience)
        ):
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": row["train_composite_loss"],
                        "val_loss": val_loss,
                        "val_mse": val_metrics["mse"],
                        "val_spearman": val_metrics[
                            "elementwise_spearman"
                        ],
                        "best_epoch": best_epoch,
                        "stale_epochs": stale_epochs,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if stale_epochs >= int(args.patience):
            break

    best_path = output / "best.pt"
    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    val_loss, val_metrics = _evaluate(
        model,
        val_loader,
        device=device,
        train_mean_grid=train_mean_grid,
    )
    test_dataset = SceneTeacherDataset(
        teacher_paths["test"],
        images_root,
        split="test",
        image_size=int(args.image_size),
    )
    test_loader = _loader(
        test_dataset,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        shuffle=False,
    )
    _, test_metrics = _evaluate(
        model,
        test_loader,
        device=device,
        train_mean_grid=train_mean_grid,
    )
    result = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "evidence_role": "object_qwen_scene_distiller_development",
        "training": {
            "seed": int(args.seed),
            "completed_epochs": len(history),
            "best_epoch": int(best["epoch"]),
            "early_stopped": len(history) < int(args.epochs),
            "train_source_image_count": len(train_dataset),
            "val_source_image_count": len(val_dataset),
            "test_source_image_count": len(test_dataset),
            "test_used_for_training_or_selection": False,
            "elapsed_seconds": time.monotonic() - started,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "validation": val_metrics,
        "test": test_metrics,
        "acceptance": {
            "validation_beats_train_mean_grid_baseline": (
                val_metrics["mse"]
                < val_metrics["train_mean_grid_baseline_mse"]
            ),
            "test_beats_train_mean_grid_baseline": (
                test_metrics["mse"]
                < test_metrics["train_mean_grid_baseline_mse"]
            ),
            "validation_positive_spearman": (
                val_metrics["elementwise_spearman"] > 0.0
            ),
            "test_positive_spearman": (
                test_metrics["elementwise_spearman"] > 0.0
            ),
        },
        "history": history,
        "bindings": {
            "protocol": _binding(protocol_path),
            "train_teacher": _binding(teacher_paths["train"]),
            "val_teacher": _binding(teacher_paths["val"]),
            "test_teacher": _binding(teacher_paths["test"]),
            "best_checkpoint": _binding(best_path),
            "last_checkpoint": _binding(output / "last.pt"),
            "trainer_source": _binding(Path(__file__)),
        },
        "claim_boundary": (
            "The student predicts an annotation-free image-conditioned "
            "soft sidecar distilled from object-crop Qwen observations; "
            "it is neither an independent physical sensor nor a direct "
            "full-image Qwen API observation."
        ),
    }
    if not all(result["acceptance"].values()):
        result["valid"] = False
        result["state"] = "failed_acceptance"
    _atomic_json(report_path, result)
    _atomic_json(output / "history.json", {"history": history})
    if not result["valid"]:
        raise ValueError("scene Qwen distiller failed frozen acceptance")
    return result


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "training": result["training"],
                "validation": result["validation"],
                "test": result["test"],
                "acceptance": result["acceptance"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
