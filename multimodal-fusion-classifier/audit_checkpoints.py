"""Audit best/last training checkpoints and their saved experiment metadata."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mmfc.checkpoints import load_checkpoint
from mmfc.data import write_json
from mmfc.training import (
    _training_config_fingerprint,
    load_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit checkpoint roles, history, fingerprint, and state schema"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _state_schema(state: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    state_numel = 0
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"model_state value is not a tensor: {name}")
        shape = list(value.shape)
        state_numel += value.numel()
        entries.append(
            {
                "name": str(name),
                "shape": shape,
                "dtype": str(value.dtype),
            }
        )
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "tensor_count": len(entries),
        "state_numel": int(state_numel),
        "schema_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    state_summary = _state_schema(checkpoint["model_state"])
    summary = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "keys": sorted(str(key) for key in checkpoint),
        "epoch": int(checkpoint.get("epoch", -1)),
        "val_accuracy": float(checkpoint.get("val_accuracy", float("nan"))),
        "best_val_accuracy": float(
            checkpoint.get("best_val_accuracy", float("nan"))
        ),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_mode": checkpoint.get("selection_mode"),
        "selection_score": checkpoint.get("selection_score"),
        "best_selection_score": checkpoint.get("best_selection_score"),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "config_fingerprint": checkpoint.get("config_fingerprint"),
        "class_names": checkpoint.get("class_names"),
        "history": checkpoint.get("history"),
        "state": state_summary,
    }
    del checkpoint
    gc.collect()
    return summary


def _record_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def audit_checkpoint_directory(
    config: Mapping[str, Any],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    history_path = checkpoint_dir / "history.json"
    training_summary_path = checkpoint_dir / "training_summary.json"
    required_paths = (best_path, last_path, history_path, training_summary_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        return {
            "valid": False,
            "checkpoint_dir": str(checkpoint_dir),
            "errors": [f"missing required artifact: {path}" for path in missing_paths],
        }

    best = _checkpoint_summary(best_path)
    last = _checkpoint_summary(last_path)
    history_document = json.loads(history_path.read_text(encoding="utf-8"))
    training_summary = json.loads(
        training_summary_path.read_text(encoding="utf-8")
    )
    expected_fingerprint = _training_config_fingerprint(dict(config))
    expected_class_names = [
        item["name"] if isinstance(item, dict) else str(item)
        for item in config.get("classes", [])
    ]
    expected_epochs = int(config.get("training", {}).get("epochs", 0))
    selection_metric = str(
        last["selection_metric"]
        or training_summary.get("selection_metric")
        or config.get("training", {}).get("selection_metric", "accuracy")
    ).lower()
    selection_mode = str(
        last.get("selection_mode")
        or training_summary.get("selection_mode")
        or config.get("training", {}).get(
            "selection_mode",
            "max" if selection_metric in {"accuracy", "macro_f1"} else "min",
        )
    ).lower()
    metric_key = f"val_{selection_metric}"
    last_history = last["history"]
    best_history = best["history"]
    errors: list[str] = []
    _record_error(
        errors,
        selection_mode in {"min", "max"},
        "selection_mode must be min or max",
    )
    _record_error(
        errors,
        best["checkpoint_role"] == "best_evaluation",
        "best.pt checkpoint_role must be best_evaluation",
    )
    _record_error(
        errors,
        last["checkpoint_role"] == "last_resume",
        "last.pt checkpoint_role must be last_resume",
    )
    for name, checkpoint in (("best.pt", best), ("last.pt", last)):
        _record_error(
            errors,
            checkpoint["config_fingerprint"] == expected_fingerprint,
            f"{name} configuration fingerprint mismatch",
        )
        _record_error(
            errors,
            checkpoint["class_names"] == expected_class_names,
            f"{name} class_names mismatch",
        )
    _record_error(
        errors,
        isinstance(last_history, list),
        "last.pt history must be a list",
    )
    _record_error(
        errors,
        isinstance(best_history, list),
        "best.pt history must be a list",
    )
    if isinstance(last_history, list):
        _record_error(
            errors,
            len(last_history) == last["epoch"],
            "last.pt history length must equal its epoch",
        )
        _record_error(
            errors,
            last["epoch"] == expected_epochs,
            f"last.pt epoch must equal configured epochs ({expected_epochs})",
        )
        eligible = [
            row
            for row in last_history
            if metric_key in row and row[metric_key] is not None
        ]
        if eligible:
            selector = min if selection_mode == "min" else max
            expected_best = selector(
                eligible,
                key=lambda row: float(row[metric_key]),
            )
            expected_best_epoch = int(expected_best["epoch"])
            expected_best_score = float(expected_best[metric_key])
            _record_error(
                errors,
                best["epoch"] == expected_best_epoch,
                (
                    f"best.pt epoch {best['epoch']} does not match history "
                    f"arg{selection_mode} epoch {expected_best_epoch}"
                ),
            )
            if selection_metric == "accuracy":
                _record_error(
                    errors,
                    abs(best["val_accuracy"] - expected_best_score) <= 1e-12,
                    "best.pt val_accuracy does not match history selection",
                )
        else:
            errors.append(f"history has no usable selection metric: {metric_key}")
    if isinstance(best_history, list):
        _record_error(
            errors,
            len(best_history) == best["epoch"],
            "best.pt history length must equal its epoch",
        )
    _record_error(
        errors,
        best["state"]["schema_sha256"] == last["state"]["schema_sha256"],
        "best.pt and last.pt model state schemas differ",
    )
    _record_error(
        errors,
        int(training_summary.get("completed_epochs", -1)) == expected_epochs,
        "training_summary completed_epochs mismatch",
    )
    _record_error(
        errors,
        training_summary.get("config_fingerprint") == expected_fingerprint,
        "training_summary configuration fingerprint mismatch",
    )
    history_rows = history_document.get("history")
    _record_error(
        errors,
        isinstance(history_rows, list) and len(history_rows) == expected_epochs,
        "history.json row count mismatch",
    )
    _record_error(
        errors,
        history_document.get("config_fingerprint") == expected_fingerprint,
        "history.json configuration fingerprint mismatch",
    )
    resume_fields = {
        "optimizer_state",
        "scheduler_state",
        "data_loader_generator_state",
        "torch_rng_state",
        "cuda_rng_state_all",
    }
    _record_error(
        errors,
        resume_fields.issubset(last["keys"]),
        "last.pt is missing exact-resume state",
    )
    return {
        "valid": not errors,
        "checkpoint_dir": str(checkpoint_dir),
        "expected": {
            "epochs": expected_epochs,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "config_fingerprint": expected_fingerprint,
            "class_names": expected_class_names,
        },
        "best": {key: value for key, value in best.items() if key != "history"},
        "last": {key: value for key, value in last.items() if key != "history"},
        "history_json_rows": (
            len(history_rows) if isinstance(history_rows, list) else None
        ),
        "training_summary_completed_epochs": training_summary.get(
            "completed_epochs"
        ),
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_dir = resolve_project_path(
        args.checkpoint_dir
        or config.get("training", {}).get("checkpoint_dir", "checkpoints")
    )
    report = audit_checkpoint_directory(config, checkpoint_dir)
    output = (
        resolve_project_path(args.output)
        if args.output is not None
        else checkpoint_dir / "checkpoint_audit.json"
    )
    write_json(output, report)
    print(json.dumps({"output": str(output), **report}, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
