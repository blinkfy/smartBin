"""Training loop for isolated same-input multimodal baseline adaptations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .checkpoints import load_checkpoint
from .model import PROJECT_ROOT
from .recent_baselines import (
    BRANCH_NAMES,
    RecentFusionBaseline,
    build_recent_baseline_model,
    tmc_evidential_loss,
)
from .training import (
    ManifestDataset,
    degrade_sensor_grid,
    make_scheduler,
    resolve_project_path,
    set_seed,
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }
    payload = json.loads(json.dumps(payload, ensure_ascii=False))
    training = payload.get("training")
    if isinstance(training, dict):
        training.pop("resume", None)
        training.pop("resume_checkpoint", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sample_key(record: Mapping[str, Any]) -> str:
    value = record.get("sample_id", record.get("id"))
    if value is None or not str(value).strip():
        raise ValueError("training record lacks sample_id/id")
    return str(value)


def _limited_dataset(
    dataset: Dataset,
    value: Any,
    *,
    option_name: str,
) -> Dataset:
    if value is None:
        return dataset
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"{option_name} must be greater than 0")
    return Subset(dataset, range(min(limit, len(dataset))))


def _dataset_sample_ids(dataset: Dataset) -> list[str]:
    if isinstance(dataset, Subset):
        parent = dataset.dataset
        if not isinstance(parent, ManifestDataset):
            raise TypeError("baseline training subset must wrap ManifestDataset")
        return [_sample_key(parent.records[index]) for index in dataset.indices]
    if not isinstance(dataset, ManifestDataset):
        raise TypeError("baseline training requires ManifestDataset")
    return [_sample_key(record) for record in dataset.records]


class CumulativeLossHistory:
    """Per-sample branch loss history used by the QMF ranking objective."""

    def __init__(self, sample_ids: Sequence[str]) -> None:
        values = [str(value) for value in sample_ids]
        if not values or len(set(values)) != len(values):
            raise ValueError("sample_ids must be non-empty and unique")
        self.sample_ids = tuple(values)
        self.index = {value: position for position, value in enumerate(values)}
        self.losses = torch.zeros(len(values), len(BRANCH_NAMES), dtype=torch.float64)
        self.updates = torch.zeros(len(values), dtype=torch.int64)

    def update_and_pair_targets(
        self,
        sample_ids: Sequence[str],
        branch_losses: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values = [str(value) for value in sample_ids]
        if tuple(branch_losses.shape) != (len(values), len(BRANCH_NAMES)):
            raise ValueError("branch_losses must have shape [B,3]")
        rows = []
        for value in values:
            if value not in self.index:
                raise KeyError(f"unknown training sample_id: {value}")
            rows.append(self.index[value])
        row_tensor = torch.as_tensor(rows, dtype=torch.long)
        self.losses[row_tensor] += branch_losses.detach().cpu().double()
        self.updates[row_tensor] += 1

        rolled_rows = torch.roll(row_tensor, shifts=-1, dims=0)
        first = self.losses[row_tensor]
        second = self.losses[rolled_rows]
        minimum = self.losses.amin(dim=0, keepdim=True)
        maximum = self.losses.amax(dim=0, keepdim=True)
        scale = (maximum - minimum).clamp_min(1e-12)
        first = (first - minimum) / scale
        second = (second - minimum) / scale
        target = torch.sign(first - second)
        margin = (first - second).abs()
        return (
            target.to(
                device=branch_losses.device,
                dtype=branch_losses.dtype,
            ),
            margin.to(
                device=branch_losses.device,
                dtype=branch_losses.dtype,
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_ids": list(self.sample_ids),
            "losses": self.losses.clone(),
            "updates": self.updates.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if tuple(state.get("sample_ids", [])) != self.sample_ids:
            raise ValueError("QMF history sample order changed")
        losses = torch.as_tensor(state.get("losses"), dtype=torch.float64)
        updates = torch.as_tensor(state.get("updates"), dtype=torch.int64)
        if losses.shape != self.losses.shape or updates.shape != self.updates.shape:
            raise ValueError("QMF history state shape mismatch")
        self.losses.copy_(losses)
        self.updates.copy_(updates)


def qmf_ranking_loss(
    confidence: Tensor,
    target: Tensor,
    margin: Tensor,
) -> Tensor:
    """Pairwise confidence ranking with the QMF cumulative-loss direction."""

    if confidence.shape != target.shape or confidence.shape != margin.shape:
        raise ValueError("confidence, target, and margin must share shape [B,3]")
    if confidence.ndim != 2 or confidence.shape[1] != len(BRANCH_NAMES):
        raise ValueError("QMF ranking inputs must have shape [B,3]")
    nonzero = target != 0
    if not bool(nonzero.any()):
        return confidence.sum() * 0.0
    first = confidence
    second = torch.roll(confidence, shifts=-1, dims=0)
    safe_target = torch.where(nonzero, target, torch.ones_like(target))
    adjusted_second = second + margin / safe_target
    losses = F.margin_ranking_loss(
        first,
        adjusted_second,
        -safe_target,
        margin=0.0,
        reduction="none",
    )
    return losses[nonzero].mean()


def _macro_f1(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0, dtype=np.float64)
    actual = confusion.sum(axis=1, dtype=np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(f1.mean())


def baseline_objective(
    model: RecentFusionBaseline,
    logits: Tensor,
    auxiliary: Mapping[str, Tensor],
    labels: Tensor,
    *,
    epoch: int,
    qmf_history: CumulativeLossHistory | None,
    sample_ids: Sequence[str] | None,
    qmf_ranking_weight: float,
    tmc_annealing_epochs: int,
    classification_focal_weight: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if model.method in {"concat_mlp", "gmu", "lmf"}:
        classification = F.cross_entropy(logits, labels)
        per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
        focal = (
            float(focal_alpha)
            * (1.0 - torch.exp(-per_sample_ce)).pow(float(focal_gamma))
            * per_sample_ce
        ).mean()
        total = classification + float(classification_focal_weight) * focal
        zero = logits.sum() * 0.0
        return total, {
            "joint_classification": classification,
            "focal": focal,
            "branch_classification": zero,
            "ranking": zero,
        }

    branch_logits = auxiliary["branch_logits"]
    batch_size, branches, classes = branch_logits.shape
    if branches != len(BRANCH_NAMES) or classes != model.num_classes:
        raise ValueError("unexpected baseline branch logit shape")
    if tuple(labels.shape) != (batch_size,):
        raise ValueError("labels must have shape [B]")

    if model.method == "qmf":
        branch_losses = F.cross_entropy(
            branch_logits.reshape(batch_size * branches, classes),
            labels.repeat_interleave(branches),
            reduction="none",
        ).reshape(batch_size, branches)
        joint_losses = F.cross_entropy(logits, labels, reduction="none")
        ranking = logits.sum() * 0.0
        if qmf_history is not None:
            if sample_ids is None:
                raise ValueError("QMF training requires sample_ids")
            target, margin = qmf_history.update_and_pair_targets(
                sample_ids,
                branch_losses,
            )
            ranking = qmf_ranking_loss(
                auxiliary["branch_confidence"],
                target,
                margin,
            )
        total = (
            joint_losses.mean()
            + branch_losses.mean(dim=0).sum()
            + float(qmf_ranking_weight) * ranking
        )
        return total, {
            "joint_classification": joint_losses.mean(),
            "branch_classification": branch_losses.mean(),
            "ranking": ranking,
        }

    branch_alpha = auxiliary["branch_alpha"]
    fused_alpha = auxiliary["fused_alpha"]
    branch_losses = torch.stack(
        [
            tmc_evidential_loss(
                labels,
                branch_alpha[:, index],
                epoch=epoch,
                annealing_epochs=tmc_annealing_epochs,
            )
            for index in range(branches)
        ],
        dim=1,
    )
    fused_losses = tmc_evidential_loss(
        labels,
        fused_alpha,
        epoch=epoch,
        annealing_epochs=tmc_annealing_epochs,
    )
    total = fused_losses.mean() + branch_losses.mean(dim=0).sum()
    return total, {
        "joint_evidential": fused_losses.mean(),
        "branch_evidential": branch_losses.mean(),
        "ranking": logits.sum() * 0.0,
    }


def _run_epoch(
    model: RecentFusionBaseline,
    loader: DataLoader,
    *,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    robustness_config: Mapping[str, Any],
    qmf_history: CumulativeLossHistory | None,
    qmf_ranking_weight: float,
    tmc_annealing_epochs: int,
    classification_focal_weight: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    confusion = np.zeros(
        (model.num_classes, model.num_classes),
        dtype=np.int64,
    )
    component_totals: dict[str, float] = {}
    weight_totals = torch.zeros(len(BRANCH_NAMES), dtype=torch.float64)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            clean_sensors = batch["sensor"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
                degradation = degrade_sensor_grid(
                    clean_sensors,
                    dict(robustness_config),
                )
            else:
                degradation = {
                    "degraded_grid": clean_sensors,
                    "availability": clean_sensors.new_ones(
                        labels.shape[0], 1
                    ),
                }
            logits, auxiliary = model(
                images,
                degradation["degraded_grid"],
                sensor_available=degradation["availability"],
                return_aux=True,
            )
            loss, components = baseline_objective(
                model,
                logits,
                auxiliary,
                labels,
                epoch=epoch,
                qmf_history=qmf_history if training else None,
                sample_ids=(
                    [str(value) for value in batch["sample_id"]]
                    if training
                    else None
                ),
                qmf_ranking_weight=qmf_ranking_weight,
                tmc_annealing_epochs=tmc_annealing_epochs,
                classification_focal_weight=classification_focal_weight,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("baseline objective became non-finite")
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.numel()
            predictions = logits.argmax(dim=1)
            total_loss += float(loss.detach()) * batch_size
            correct += int((predictions == labels).sum())
            count += batch_size
            encoded = labels * model.num_classes + predictions
            confusion += (
                torch.bincount(
                    encoded,
                    minlength=model.num_classes * model.num_classes,
                )
                .reshape(model.num_classes, model.num_classes)
                .detach()
                .cpu()
                .numpy()
            )
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + (
                    float(value.detach()) * batch_size
                )
            weight_totals += (
                auxiliary["modality_weights"]
                .detach()
                .double()
                .sum(dim=0)
                .cpu()
            )

    metrics = {
        "loss": total_loss / max(count, 1),
        "accuracy": correct / max(count, 1),
        "macro_f1": _macro_f1(confusion),
    }
    metrics.update(
        {
            name: value / max(count, 1)
            for name, value in component_totals.items()
        }
    )
    for index, name in enumerate(BRANCH_NAMES):
        metrics[f"{name}_weight"] = float(weight_totals[index] / max(count, 1))
    return metrics


def _resume_path(
    training: Mapping[str, Any],
    checkpoint_dir: Path,
) -> Path | None:
    explicit = training.get("resume_checkpoint")
    if explicit:
        return resolve_project_path(explicit)
    return checkpoint_dir / "last.pt" if bool(training.get("resume", False)) else None


def _restore_checkpoint(
    checkpoint_path: Path,
    *,
    expected_fingerprint: str,
    model: RecentFusionBaseline,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    qmf_history: CumulativeLossHistory | None,
    device: torch.device,
) -> tuple[int, float, list[dict[str, float | int]]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("baseline resume configuration fingerprint mismatch")
    if checkpoint.get("baseline_method") != model.method:
        raise ValueError("baseline resume method mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    generator.set_state(checkpoint["data_loader_generator_state"].cpu())
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in checkpoint["cuda_rng_state_all"]]
        )
    if qmf_history is not None:
        qmf_history.load_state_dict(checkpoint["qmf_history_state"])
    history = checkpoint.get("history", [])
    start_epoch = int(checkpoint.get("epoch", 0))
    if len(history) != start_epoch:
        raise ValueError("baseline resume history length mismatch")
    return start_epoch, float(checkpoint["best_selection_score"]), history


def train_recent_baseline_from_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    classes = config.get("classes", [])
    class_names = [
        value["name"] if isinstance(value, Mapping) else str(value)
        for value in classes
    ]
    if not class_names:
        raise ValueError("config classes must not be empty")
    training = config.get("training", {})
    data = config.get("data", {})
    baseline = config.get("recent_baseline", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    device = torch.device(
        training.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    full_train_dataset = ManifestDataset(
        data["train_manifest"],
        class_names=class_names,
        image_size=int(data.get("image_size", 480)),
        sensor_manifest=data.get("sensor_manifest"),
        preprocess=config.get("preprocess", {}),
    )
    full_val_dataset = ManifestDataset(
        data["val_manifest"],
        class_names=class_names,
        image_size=int(data.get("image_size", 480)),
        sensor_manifest=data.get("sensor_manifest"),
        preprocess=config.get("preprocess", {}),
    )
    train_dataset = _limited_dataset(
        full_train_dataset,
        training.get("max_train_samples"),
        option_name="training.max_train_samples",
    )
    val_dataset = _limited_dataset(
        full_val_dataset,
        training.get("max_val_samples"),
        option_name="training.max_val_samples",
    )
    generator = torch.Generator().manual_seed(seed)
    num_workers = int(training.get("num_workers", 0))
    loader_options = {
        "batch_size": int(training.get("batch_size", 16)),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_options.update(
            {
                "persistent_workers": True,
                "prefetch_factor": int(training.get("prefetch_factor", 2)),
            }
        )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options,
    )
    model = build_recent_baseline_model(config).to(device)
    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    epochs = int(training.get("epochs", 100))
    if epochs <= 0:
        raise ValueError("training.epochs must be positive")
    scheduler = make_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=int(training.get("warmup_epochs", 5)),
    )
    checkpoint_dir = resolve_project_path(training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _config_fingerprint(config)
    selection_metric = str(
        training.get("selection_metric", "accuracy")
    ).strip().lower()
    if selection_metric not in {"accuracy", "macro_f1"}:
        raise ValueError("selection_metric must be accuracy or macro_f1")
    qmf_history = (
        CumulativeLossHistory(_dataset_sample_ids(train_dataset))
        if model.method == "qmf"
        else None
    )
    qmf_ranking_weight = float(baseline.get("ranking_weight", 0.1))
    tmc_annealing_epochs = int(baseline.get("annealing_epochs", 10))
    classification_focal_weight = float(
        baseline.get("classification_focal_weight", 0.0)
    )
    loss_config = config.get("loss", {})
    focal_alpha = float(loss_config.get("alpha", 0.25))
    focal_gamma = float(loss_config.get("gamma", 2.0))
    if (
        qmf_ranking_weight < 0.0
        or tmc_annealing_epochs <= 0
        or classification_focal_weight < 0.0
        or not 0.0 <= focal_alpha <= 1.0
        or focal_gamma < 0.0
    ):
        raise ValueError("invalid recent baseline loss hyperparameters")

    start_epoch = 0
    best_selection_score = -math.inf
    history: list[dict[str, float | int]] = []
    resumed_from: str | None = None
    resume_path = _resume_path(training, checkpoint_dir)
    if resume_path is not None:
        start_epoch, best_selection_score, history = _restore_checkpoint(
            resume_path,
            expected_fingerprint=fingerprint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            qmf_history=qmf_history,
            device=device,
        )
        resumed_from = str(resume_path.resolve())

    robustness = config.get("robustness", {})
    for epoch_index in range(start_epoch, epochs):
        epoch = epoch_index + 1
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            epoch=epoch,
            optimizer=optimizer,
            robustness_config=robustness,
            qmf_history=qmf_history,
            qmf_ranking_weight=qmf_ranking_weight,
            tmc_annealing_epochs=tmc_annealing_epochs,
            classification_focal_weight=classification_focal_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            epoch=epoch,
            optimizer=None,
            robustness_config=robustness,
            qmf_history=None,
            qmf_ranking_weight=qmf_ranking_weight,
            tmc_annealing_epochs=tmc_annealing_epochs,
            classification_focal_weight=classification_focal_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        row.update(
            {
                f"train_{name}": float(value)
                for name, value in train_metrics.items()
            }
        )
        row.update(
            {
                f"val_{name}": float(value)
                for name, value in val_metrics.items()
            }
        )
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        selection_score = float(val_metrics[selection_metric])
        is_best = selection_score > best_selection_score
        if is_best:
            best_selection_score = selection_score
        metadata = {
            "epoch": epoch,
            "baseline_method": model.method,
            "class_names": class_names,
            "config_fingerprint": fingerprint,
            "selection_metric": selection_metric,
            "selection_score": selection_score,
            "best_selection_score": best_selection_score,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "history": history,
            "config": {
                key: value
                for key, value in config.items()
                if not str(key).startswith("_")
            },
        }
        _atomic_write_json(
            checkpoint_dir / "history.json",
            {
                **metadata,
                "train_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "resumed_from": resumed_from,
            },
        )
        last = {
            **metadata,
            "checkpoint_role": "last_resume",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "data_loader_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
            "qmf_history_state": (
                qmf_history.state_dict()
                if qmf_history is not None
                else None
            ),
        }
        _atomic_torch_save(checkpoint_dir / "last.pt", last)
        if is_best:
            _atomic_torch_save(
                checkpoint_dir / "best.pt",
                {
                    **metadata,
                    "checkpoint_role": "best_evaluation",
                    "model_state": model.state_dict(),
                },
            )

    result = {
        "valid": len(history) == epochs,
        "baseline_method": model.method,
        "seed": seed,
        "epochs": epochs,
        "completed_epochs": len(history),
        "selection_metric": selection_metric,
        "best_selection_score": best_selection_score,
        "checkpoint_dir": str(checkpoint_dir),
        "config_fingerprint": fingerprint,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "resumed_from": resumed_from,
        "history": history,
    }
    _atomic_write_json(checkpoint_dir / "training_summary.json", result)
    return result
