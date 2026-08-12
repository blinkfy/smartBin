"""Benchmark deterministic DataLoader concurrency without running the model."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Subset

from mmfc.training import ManifestDataset, resolve_project_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "ablations" / "full_cross_modal_quality.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "training"
    / "INPUT_PIPELINE_BENCHMARK.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic MMFC input-pipeline concurrency"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--equivalence-samples", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


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


def build_dataset(config: dict[str, Any]) -> ManifestDataset:
    data = config["data"]
    classes = config["classes"]
    class_names = [
        item["name"] if isinstance(item, dict) else str(item) for item in classes
    ]
    return ManifestDataset(
        data["train_manifest"],
        class_names=class_names,
        image_size=int(data["image_size"]),
        sensor_manifest=data.get("sensor_manifest"),
        preprocess=config.get("preprocess", {}),
    )


def make_loader(
    dataset: torch.utils.data.Dataset,
    *,
    workers: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def loader_digest(loader: DataLoader, *, epochs: int) -> str:
    digest = hashlib.sha256()
    for epoch in range(epochs):
        digest.update(epoch.to_bytes(4, "little"))
        for batch in loader:
            for sample_id in batch["sample_id"]:
                encoded = str(sample_id).encode("utf-8")
                digest.update(len(encoded).to_bytes(4, "little"))
                digest.update(encoded)
            for name in ("image", "sensor", "label"):
                tensor = batch[name].detach().contiguous().cpu()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def benchmark_loader(
    loader: DataLoader,
    *,
    epochs: int,
) -> dict[str, Any]:
    epoch_seconds = []
    samples_per_epoch = []
    for _ in range(epochs):
        started = time.perf_counter()
        sample_count = 0
        for batch in loader:
            sample_count += int(batch["label"].numel())
        elapsed = time.perf_counter() - started
        epoch_seconds.append(elapsed)
        samples_per_epoch.append(sample_count)
    steady_seconds = epoch_seconds[1:] if len(epoch_seconds) > 1 else epoch_seconds
    steady_samples = samples_per_epoch[1:] if len(samples_per_epoch) > 1 else samples_per_epoch
    total_seconds = sum(steady_seconds)
    return {
        "epoch_seconds": epoch_seconds,
        "samples_per_epoch": samples_per_epoch,
        "steady_state_epoch_seconds_mean": total_seconds / len(steady_seconds),
        "steady_state_samples_per_second": sum(steady_samples) / total_seconds,
    }


def benchmark_options(
    *,
    dataset: torch.utils.data.Dataset,
    workers: Iterable[int],
    batch_size: int,
    epochs: int,
    equivalence_samples: int,
    seed: int,
) -> dict[str, Any]:
    worker_values = list(workers)
    if not worker_values or any(value < 0 for value in worker_values):
        raise ValueError("workers must contain non-negative integers")
    if len(set(worker_values)) != len(worker_values):
        raise ValueError("workers must not contain duplicates")
    if epochs <= 0 or equivalence_samples <= 0:
        raise ValueError("epochs and equivalence_samples must be positive")
    equivalence_dataset = Subset(
        dataset,
        range(min(equivalence_samples, len(dataset))),
    )
    rows = []
    baseline_digest = None
    for worker_count in worker_values:
        digest = loader_digest(
            make_loader(
                equivalence_dataset,
                workers=worker_count,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
            ),
            epochs=epochs,
        )
        if baseline_digest is None:
            baseline_digest = digest
        equivalent = digest == baseline_digest
        throughput = benchmark_loader(
            make_loader(
                dataset,
                workers=worker_count,
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
            ),
            epochs=epochs,
        )
        rows.append(
            {
                "num_workers": worker_count,
                "persistent_workers": False,
                "prefetch_factor": 2 if worker_count > 0 else None,
                "equivalence_sha256": digest,
                "tensor_and_order_equivalent_to_baseline": equivalent,
                **throughput,
            }
        )
    valid_rows = [
        row for row in rows if row["tensor_and_order_equivalent_to_baseline"]
    ]
    fastest = max(
        valid_rows,
        key=lambda item: item["steady_state_samples_per_second"],
    )
    baseline = rows[0]
    return {
        "valid": all(
            row["tensor_and_order_equivalent_to_baseline"] for row in rows
        ),
        "baseline_num_workers": baseline["num_workers"],
        "baseline_samples_per_second": baseline[
            "steady_state_samples_per_second"
        ],
        "recommended_num_workers": fastest["num_workers"],
        "recommended_samples_per_second": fastest[
            "steady_state_samples_per_second"
        ],
        "recommended_speedup": (
            fastest["steady_state_samples_per_second"]
            / baseline["steady_state_samples_per_second"]
        ),
        "options": rows,
    }


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.epochs <= 0 or args.equivalence_samples <= 0:
        raise ValueError("samples, epochs, and equivalence-samples must be positive")
    config_path = _resolve(args.config)
    config = _read_json(config_path)
    full_dataset = build_dataset(config)
    dataset = Subset(full_dataset, range(min(args.samples, len(full_dataset))))
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "deterministic_training_input_pipeline_benchmark",
        "measured_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "sample_count": len(dataset),
        "epochs": args.epochs,
        "equivalence_sample_count": min(
            args.equivalence_samples,
            len(dataset),
        ),
        "batch_size": int(config["training"]["batch_size"]),
        "scientific_invariants": {
            "shuffle": True,
            "shuffle_seed": int(config["training"]["seed"]),
            "image_preprocess_unchanged": True,
            "multi_epoch_sample_order_required_identical": True,
            "multi_epoch_tensor_bytes_required_identical": True,
            "persistent_workers_disabled": True,
            "model_not_executed": True,
            "optimizer_not_executed": True,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "logical_cpu_count": __import__("os").cpu_count(),
        },
        **benchmark_options(
            dataset=dataset,
            workers=args.workers,
            batch_size=int(config["training"]["batch_size"]),
            epochs=args.epochs,
            equivalence_samples=args.equivalence_samples,
            seed=int(config["training"]["seed"]),
        ),
    }
    data = config["data"]
    report["source_bindings"] = {
        "benchmark_source": _binding(Path(__file__)),
        "config": _binding(config_path),
        "training_source": _binding(PROJECT_ROOT / "mmfc" / "training.py"),
        "train_manifest": _binding(
            resolve_project_path(Path(data["train_manifest"]))
        ),
        "sensor_manifest": _binding(
            resolve_project_path(Path(data["sensor_manifest"]))
        ),
    }
    report["recommendation_boundary"] = (
        "The recommendation applies to this deterministic image-conditioned "
        "dataset and preprocessing path. Rebenchmark if stochastic Dataset "
        "transforms or a different storage device are introduced."
    )
    output = _resolve(args.output)
    _atomic_write(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        output.with_suffix(".md"),
        "\n".join(
            [
                "# MMFC training input-pipeline benchmark",
                "",
                "| Workers | Persistent | Samples/s | Equivalent |",
                "|---:|:---:|---:|:---:|",
                *[
                    (
                        f"| {row['num_workers']} | "
                        f"{str(row['persistent_workers']).lower()} | "
                        f"{row['steady_state_samples_per_second']:.2f} | "
                        f"{str(row['tensor_and_order_equivalent_to_baseline']).lower()} |"
                    )
                    for row in report["options"]
                ],
                "",
                (
                    f"Recommended workers: `{report['recommended_num_workers']}`; "
                    f"measured input-only speedup: "
                    f"`{report['recommended_speedup']:.2f}x`."
                ),
                "",
                "This benchmark requires byte-identical images, sensor grids, "
                "labels, and sample order. It does not benchmark model compute.",
                "",
            ]
        ),
    )
    print(
        json.dumps(
            {
                "valid": report["valid"],
                "recommended_num_workers": report["recommended_num_workers"],
                "recommended_speedup": report["recommended_speedup"],
                "output": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
