from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import prepare_equivalent_fast_config as fast_config


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _binding(root: Path, path: Path) -> dict:
    return {
        "path": Path(os.path.relpath(path, root)).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_fast_config_changes_only_input_runtime_fields(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(fast_config, "PROJECT_ROOT", root)
    stub = root / "builder.py"
    stub.write_text("builder", encoding="utf-8")
    monkeypatch.setattr(fast_config, "BUILDER_SOURCE", stub)
    base_path = root / "base.json"
    benchmark_path = root / "benchmark.json"
    output_path = root / "configs" / "fast.json"
    base = {
        "model": {"feature_dim": 512},
        "data": {"train_manifest": "train.jsonl"},
        "training": {
            "seed": 42,
            "epochs": 100,
            "num_workers": 0,
            "checkpoint_dir": "checkpoints/base",
        },
        "experiment": {
            "name": "full_cross_modal_quality",
            "description": "v3",
        },
    }
    _write_json(base_path, base)
    source = root / "source.txt"
    source.write_text("source", encoding="utf-8")
    _write_json(
        benchmark_path,
        {
            "state": "complete",
            "valid": True,
            "sample_count": 7002,
            "epochs": 2,
            "recommended_num_workers": 4,
            "recommended_speedup": 1.75,
            "options": [
                {"tensor_and_order_equivalent_to_baseline": True},
                {"tensor_and_order_equivalent_to_baseline": True},
            ],
            "scientific_invariants": {
                "shuffle": True,
                "multi_epoch_sample_order_required_identical": True,
                "multi_epoch_tensor_bytes_required_identical": True,
                "persistent_workers_disabled": True,
            },
            "source_bindings": {"source": _binding(root, source)},
        },
    )

    report = fast_config.build_fast_config(
        base_config_path=base_path,
        benchmark_path=benchmark_path,
        output_path=output_path,
    )
    generated = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["valid"] is True
    assert generated["training"]["num_workers"] == 4
    assert generated["training"]["seed"] == 42
    assert fast_config.normalized_fast_config(generated) == (
        fast_config.normalized_fast_config(base)
    )
