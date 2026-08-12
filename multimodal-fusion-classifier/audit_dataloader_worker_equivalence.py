"""Verify that loader prefetch workers preserve per-epoch sample order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mmfc.training import ManifestDataset, _SingleWorkerOrderRandomSampler


ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_orders(
    dataset: ManifestDataset,
    *,
    seed: int,
    workers: int,
    epochs: int,
    batch_size: int,
) -> list[list[str]]:
    generator = torch.Generator().manual_seed(seed)
    options: dict[str, Any] = {"batch_size": batch_size, "num_workers": workers}
    if workers > 0:
        options.update(
            {
                "sampler": _SingleWorkerOrderRandomSampler(dataset, generator),
                "generator": torch.Generator().manual_seed(seed + 1_000_003),
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )
    else:
        options.update({"shuffle": True, "generator": generator})
    loader = DataLoader(dataset, **options)
    return [
        [sample_id for batch in loader for sample_id in batch["sample_id"]]
        for _ in range(epochs)
    ]


def digest(order: list[str]) -> str:
    return hashlib.sha256("\n".join(order).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/submission-revision/constant-gate/constant_gate_seed42.json",
            "configs/submission-revision/constant-gate/constant_gate_seed43.json",
            "configs/submission-revision/constant-gate/constant_gate_seed44.json",
        ],
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        default=(
            "reports/qwen-material/submission-revision/"
            "dynamic-vs-constant-gate/DATALOADER_ORDER_AUDIT.json"
        ),
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.workers <= 0:
        raise ValueError("epochs and workers must be positive")

    rows: list[dict[str, Any]] = []
    for config_value in args.configs:
        config_path = (ROOT / config_value).resolve()
        config = load_json(config_path)
        training = config["training"]
        data = config["data"]
        seed = int(training["seed"])
        class_names = [row["name"] for row in config["classes"]]
        dataset = ManifestDataset(
            data["train_manifest"],
            class_names=class_names,
            image_size=int(data["image_size"]),
            sensor_manifest=data["sensor_manifest"],
            preprocess=config.get("preprocess"),
            load_images=False,
        )
        zero = sample_orders(
            dataset,
            seed=seed,
            workers=0,
            epochs=args.epochs,
            batch_size=int(training["batch_size"]),
        )
        parallel = sample_orders(
            dataset,
            seed=seed,
            workers=args.workers,
            epochs=args.epochs,
            batch_size=int(training["batch_size"]),
        )
        rows.append(
            {
                "training_seed": seed,
                "sample_count": len(dataset),
                "epochs_checked": args.epochs,
                "workers_compared": [0, args.workers],
                "identical": zero == parallel,
                "reference_epoch_sha256": [digest(order) for order in zero],
                "parallel_epoch_sha256": [digest(order) for order in parallel],
            }
        )

    report = {
        "valid": all(row["identical"] for row in rows),
        "scope": (
            "Sample-order equivalence only. Image preprocessing is deterministic and "
            "sensor degradation remains in the training process after host transfer."
        ),
        "runs": rows,
    }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
