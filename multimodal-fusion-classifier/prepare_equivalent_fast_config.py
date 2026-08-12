"""Generate an isolated 4-worker config from the bound input benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
BUILDER_SOURCE = Path(__file__).resolve()
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "configs" / "ablations" / "full_cross_modal_quality.json"
)
DEFAULT_BENCHMARK = (
    PROJECT_ROOT
    / "reports"
    / "qwen-material"
    / "training"
    / "INPUT_PIPELINE_BENCHMARK.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "configs"
    / "performance"
    / "full_cross_modal_quality_workers4.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a benchmark-backed input-equivalent fast config"
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _validate_binding(binding: Mapping[str, Any]) -> bool:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    path = _resolve(Path(raw_path))
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return (
        path.is_file()
        and path.stat().st_size == binding.get("size_bytes")
        and _sha256(path) == binding.get("sha256")
    )


def normalized_fast_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    training = value.setdefault("training", {})
    training.pop("num_workers", None)
    training.pop("checkpoint_dir", None)
    experiment = value.setdefault("experiment", {})
    for name in (
        "name",
        "description",
        "performance_equivalent_of",
        "performance_evidence",
    ):
        experiment.pop(name, None)
    return value


def build_fast_config(
    *,
    base_config_path: Path = DEFAULT_BASE_CONFIG,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    base_config_path = _resolve(base_config_path)
    benchmark_path = _resolve(benchmark_path)
    output_path = _resolve(output_path)
    provenance_path = output_path.with_suffix(".provenance.json")
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError("refusing to overwrite performance config evidence")
    base = _read_json(base_config_path)
    benchmark = _read_json(benchmark_path)
    rows = benchmark.get("options", [])
    errors = []
    if (
        benchmark.get("state") != "complete"
        or benchmark.get("valid") is not True
        or benchmark.get("sample_count") != 7002
        or benchmark.get("epochs", 0) < 2
        or benchmark.get("recommended_num_workers") != 4
        or benchmark.get("recommended_speedup", 0.0) <= 1.0
        or not rows
        or not all(
            item.get("tensor_and_order_equivalent_to_baseline") is True
            for item in rows
        )
    ):
        errors.append("input-pipeline benchmark does not authorize workers=4")
    scientific = benchmark.get("scientific_invariants", {})
    if (
        scientific.get("shuffle") is not True
        or scientific.get("multi_epoch_sample_order_required_identical")
        is not True
        or scientific.get("multi_epoch_tensor_bytes_required_identical")
        is not True
        or scientific.get("persistent_workers_disabled") is not True
    ):
        errors.append("benchmark scientific invariants are incomplete")
    source_bindings = benchmark.get("source_bindings", {})
    if not isinstance(source_bindings, Mapping) or not source_bindings:
        errors.append("benchmark source bindings are missing")
    elif not all(
        isinstance(binding, Mapping) and _validate_binding(binding)
        for binding in source_bindings.values()
    ):
        errors.append("benchmark source binding drift")
    if errors:
        raise ValueError("; ".join(errors))

    fast = copy.deepcopy(base)
    base_name = str(base["experiment"]["name"])
    fast_name = f"{base_name}_workers4"
    fast["training"]["num_workers"] = 4
    fast["training"]["checkpoint_dir"] = (
        f"checkpoints/performance/{fast_name}"
    )
    fast["experiment"]["name"] = fast_name
    fast["experiment"]["description"] = (
        str(base["experiment"].get("description", ""))
        + "; deterministic 4-worker input-pipeline variant"
    )
    fast["experiment"]["performance_equivalent_of"] = base_name
    fast["experiment"]["performance_evidence"] = (
        benchmark_path.relative_to(PROJECT_ROOT).as_posix()
    )
    if normalized_fast_config(fast) != normalized_fast_config(base):
        raise ValueError("fast config changes non-performance scientific fields")
    _atomic_write(
        output_path,
        json.dumps(fast, ensure_ascii=False, indent=2) + "\n",
    )
    provenance = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "scope": "input_equivalent_training_runtime_configuration",
        "base_experiment": base_name,
        "fast_experiment": fast_name,
        "authorized_changes": {
            "training.num_workers": {
                "from": int(base["training"].get("num_workers", 0)),
                "to": 4,
            },
            "training.checkpoint_dir": fast["training"]["checkpoint_dir"],
            "experiment_metadata": True,
        },
        "measured_input_speedup": benchmark["recommended_speedup"],
        "equivalence": {
            "sample_count": benchmark["sample_count"],
            "shuffle_epochs": benchmark["epochs"],
            "sample_order_identical": True,
            "image_sensor_and_label_bytes_identical": True,
            "persistent_workers": False,
        },
        "source_bindings": {
            "builder_source": _binding(BUILDER_SOURCE),
            "base_config": _binding(base_config_path),
            "benchmark": _binding(benchmark_path),
            "generated_config": _binding(output_path),
        },
        "boundaries": [
            (
                "The measured speedup covers input preparation, not total "
                "model-training wall time."
            ),
            (
                "The generated config is for future isolated runs and does "
                "not alter or replace completed seed-replication evidence."
            ),
            (
                "Rebenchmark if storage, preprocessing, Dataset randomness, "
                "PyTorch, or hardware changes."
            ),
        ],
        "errors": [],
    }
    _atomic_write(
        provenance_path,
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
    )
    return provenance


def main() -> None:
    args = parse_args()
    report = build_fast_config(
        base_config_path=args.base_config,
        benchmark_path=args.benchmark,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "valid": report["valid"],
                "fast_experiment": report["fast_experiment"],
                "measured_input_speedup": report["measured_input_speedup"],
                "output": _resolve(args.output).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
