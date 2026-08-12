"""Benchmark fused v5/v6 raw forward latency under an identical GPU setup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from mmfc.model import PROJECT_ROOT
from train_patent_aligned_external_sensor_joint_seg import DEFAULT_NAME


REPORTS = PROJECT_ROOT / "reports" / "qwen-material"
V5_NAME = "yolo11s-seg-joint-spatial-contrastive-rule-dev-seed42"
V5_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-spatial"
    / V5_NAME
    / "weights"
    / "best.pt"
)
V6_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
    / "weights"
    / "best.pt"
)
V5_EVALUATION = (
    REPORTS
    / "joint-multimodal-seg-spatial"
    / V5_NAME
    / "evaluation.json"
)
V6_EVALUATION = (
    REPORTS
    / "joint-multimodal-seg-patent"
    / DEFAULT_NAME
    / "evaluation.json"
)
WORKER_EQUIVALENCE = (
    REPORTS
    / "joint-multimodal-seg-external"
    / "WORKER_EQUIVALENCE.json"
)
DEFAULT_OUTPUT = REPORTS / "PATENT_STACK_COMPLEXITY.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the v5 and v6 fused raw model forward"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def summarize_latencies(
    milliseconds: list[float],
    batch_size: int,
) -> dict[str, float]:
    if not milliseconds or batch_size <= 0:
        raise ValueError("latencies and batch_size must be positive")
    ordered = sorted(float(value) for value in milliseconds)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    median_batch = median(ordered)
    return {
        "median_batch_ms": median_batch,
        "p25_batch_ms": percentile(0.25),
        "p75_batch_ms": percentile(0.75),
        "median_ms_per_image": median_batch / batch_size,
        "median_images_per_second": batch_size * 1000.0 / median_batch,
    }


def _benchmark_one(
    checkpoint: Path,
    packed_input: Any,
    *,
    warmup: int,
    repeats: int,
    image_size: int,
    device: Any,
) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    wrapper = YOLO(str(checkpoint))
    model = wrapper.model.fuse(verbose=False).to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    gflops = float(get_flops(model, imgsz=image_size))
    tensor = packed_input.to(device, non_blocking=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):
        for _ in range(warmup):
            _ = model(tensor)
        torch.cuda.synchronize(device)
        latencies: list[float] = []
        for _ in range(repeats):
            start = perf_counter()
            _ = model(tensor)
            torch.cuda.synchronize(device)
            latencies.append((perf_counter() - start) * 1000.0)
    timing = summarize_latencies(latencies, int(tensor.shape[0]))
    timing["repeat_count"] = repeats
    timing["warmup_count"] = warmup
    return {
        "model_class": type(model).__name__,
        "fused_parameter_count": int(parameters),
        "gflops_per_image": gflops,
        "timing": timing,
        "peak_allocated_memory_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from mmfc.joint_seg_patent import (
        PatentAlignedReliableJointSegmentationModel,
    )
    from mmfc.joint_seg_reliable import pack_reliable_sensor_channels
    from mmfc.joint_seg_spatial import (
        SpatialReliableExternalSensorJointSegmentationModel,
    )

    del PatentAlignedReliableJointSegmentationModel
    del SpatialReliableExternalSensorJointSegmentationModel
    output = _resolve(args.output)
    if (
        args.batch_size <= 0
        or args.image_size <= 0
        or args.warmup < 1
        or args.repeats < 3
    ):
        raise ValueError("invalid benchmark dimensions or repeat counts")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("this controlled benchmark requires CUDA")
    paths = {
        "v5_checkpoint": V5_CHECKPOINT,
        "v6_checkpoint": V6_CHECKPOINT,
        "v5_evaluation": V5_EVALUATION,
        "v6_evaluation": V6_EVALUATION,
        "worker_equivalence": WORKER_EQUIVALENCE,
        "benchmark_source": Path(__file__),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "complexity benchmark inputs missing: " + ", ".join(missing)
        )
    v5_evaluation = _read_json(V5_EVALUATION)
    v6_evaluation = _read_json(V6_EVALUATION)
    workers = _read_json(WORKER_EQUIVALENCE)
    if (
        v5_evaluation.get("valid") is not True
        or v6_evaluation.get("valid") is not True
        or workers.get("valid") is not True
        or v5_evaluation["checkpoint"]["sha256"] != _sha256(V5_CHECKPOINT)
        or v6_evaluation["checkpoint"]["sha256"] != _sha256(V6_CHECKPOINT)
    ):
        raise ValueError("benchmark source binding mismatch")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260729)
    rgb = torch.randint(
        0,
        256,
        (
            args.batch_size,
            3,
            args.image_size,
            args.image_size,
        ),
        dtype=torch.uint8,
        generator=generator,
    )
    grid = torch.rand(
        args.batch_size,
        1,
        4,
        4,
        generator=generator,
    )
    availability = torch.ones(args.batch_size, 1)
    packed = pack_reliable_sensor_channels(
        rgb, grid, availability
    ).float().div(255.0)
    device = torch.device(args.device)
    results = {
        "v5_spatial_reliability": _benchmark_one(
            V5_CHECKPOINT,
            packed,
            warmup=args.warmup,
            repeats=args.repeats,
            image_size=args.image_size,
            device=device,
        ),
        "v6_patent_stack": _benchmark_one(
            V6_CHECKPOINT,
            packed,
            warmup=args.warmup,
            repeats=args.repeats,
            image_size=args.image_size,
            device=device,
        ),
    }
    v5 = results["v5_spatial_reliability"]
    v6 = results["v6_patent_stack"]
    parameter_overhead = (
        v6["fused_parameter_count"] / v5["fused_parameter_count"] - 1.0
    )
    gflops_overhead = (
        v6["gflops_per_image"] / v5["gflops_per_image"] - 1.0
    )
    latency_overhead = (
        v6["timing"]["median_batch_ms"]
        / v5["timing"]["median_batch_ms"]
        - 1.0
    )
    report = {
        "schema_version": 1,
        "state": "complete",
        "valid": True,
        "formal_paper_metric": False,
        "scope": "same_gpu_fused_raw_forward_descriptive_benchmark",
        "configuration": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "input_channels": 5,
            "sensor_available": True,
            "autocast_dtype": "float16",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "postprocessing_included": False,
            "dataloader_included": False,
            "host_to_device_transfer_included": False,
        },
        "models": results,
        "v6_over_v5": {
            "parameter_fraction": parameter_overhead,
            "gflops_fraction": gflops_overhead,
            "median_batch_latency_fraction": latency_overhead,
            "throughput_ratio": (
                v6["timing"]["median_images_per_second"]
                / v5["timing"]["median_images_per_second"]
            ),
        },
        "equivalent_training_input_pipeline": {
            "workers_from": workers["authorized_change"]["workers"]["from"],
            "workers_to": workers["authorized_change"]["workers"]["to"],
            "speedup": workers["timing"]["speedup"],
            "exact_model_state": workers["equivalence"]["model_state"][
                "exact"
            ],
            "batch_size_unchanged": workers["recommendation"][
                "batch_size_unchanged"
            ],
        },
        "boundaries": [
            "Latency is a synchronized raw fused-model forward benchmark.",
            "It excludes image decoding, DataLoader, host-to-device transfer, NMS, mask postprocessing, and serialization.",
            "It is descriptive for the current RTX 4090 environment and is not the patent 15.2 ms result.",
            "The workers=4 training speedup comes from a separate exact-equivalence experiment.",
            "Changing training batch size is not authorized by the worker equivalence result.",
        ],
        "bindings": {name: _binding(path) for name, path in paths.items()},
        "errors": [],
    }
    _atomic_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    markdown = [
        "# v5/v6 复杂度与速度边界",
        "",
        f"- 环境：`{report['configuration']['gpu']}`，"
        f"batch={args.batch_size}，imgsz={args.image_size}，AMP FP16。",
        "- 计时范围：融合后的原始模型前向；不含DataLoader、H2D、NMS、mask后处理和写盘。",
        "- v5："
        f"{v5['fused_parameter_count']:,}参数，"
        f"{v5['gflops_per_image']:.2f} GFLOPs/图，"
        f"{v5['timing']['median_ms_per_image']:.3f} ms/图，"
        f"{v5['timing']['median_images_per_second']:.1f}图/s。",
        "- v6："
        f"{v6['fused_parameter_count']:,}参数，"
        f"{v6['gflops_per_image']:.2f} GFLOPs/图，"
        f"{v6['timing']['median_ms_per_image']:.3f} ms/图，"
        f"{v6['timing']['median_images_per_second']:.1f}图/s。",
        "- v6相对v5：参数 "
        f"{100.0 * parameter_overhead:+.2f}%，GFLOPs "
        f"{100.0 * gflops_overhead:+.2f}%，中位batch延迟 "
        f"{100.0 * latency_overhead:+.2f}%。",
        "- 等效训练输入管线：workers 0→4，模型状态逐张量完全一致，"
        f"速度提升{workers['timing']['speedup']:.3f}倍；batch仍保持32。",
        "",
        "本结果是同机描述性原始前向基准，不是端到端延迟，也不复现原专利15.2 ms。",
        "",
    ]
    _atomic_text(output.with_suffix(".md"), "\n".join(markdown))
    return report


def main() -> None:
    result = benchmark(parse_args())
    print(
        json.dumps(
            {
                "state": result["state"],
                "valid": result["valid"],
                "v6_over_v5": result["v6_over_v5"],
                "output": str(DEFAULT_OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
