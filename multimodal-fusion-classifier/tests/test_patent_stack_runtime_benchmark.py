from __future__ import annotations

from benchmark_patent_stack_runtime import parse_args, summarize_latencies


def test_runtime_summary_reports_per_image_throughput() -> None:
    summary = summarize_latencies([30.0, 34.0, 32.0], batch_size=16)

    assert summary["median_batch_ms"] == 32.0
    assert summary["median_ms_per_image"] == 2.0
    assert summary["median_images_per_second"] == 500.0


def test_runtime_benchmark_defaults_match_training_batch(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark"])

    args = parse_args()

    assert args.batch_size == 32
    assert args.image_size == 640
    assert args.device == "cuda:0"
