from __future__ import annotations

import json

import pytest

from aggregate_joint_multimodal_seg import aggregate_runs


METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_run(root, seed: int, value: float) -> str:
    run_name = f"run-seed{seed}"
    run = root / run_name
    run.mkdir()
    metrics = {name: value for name in METRICS}
    _write_json(
        run / "PROTOCOL.json",
        {
            "run_name": run_name,
            "seed": seed,
            "train_args": {"seed": seed, "name": run_name, "batch": 32},
        },
    )
    _write_json(
        run / "evaluation.json",
        {
            "metrics": metrics,
            "checkpoint": {"sha256": f"{seed:064x}"},
            "forward_contract": {"metric_forward_inputs": ["RGB"]},
        },
    )
    _write_json(
        run / "OUTCOME.json",
        {
            "valid": True,
            "training": {"completed_epochs": 2},
            "dataset": {"split_source_counts": {"test": 3}},
            "metric_comparison": {
                name: {"base": 0.5} for name in METRICS
            },
        },
    )
    return run_name


def test_multiseed_aggregate_computes_sample_statistics(tmp_path) -> None:
    run_names = [
        _write_run(tmp_path, 42, 0.6),
        _write_run(tmp_path, 43, 0.7),
        _write_run(tmp_path, 44, 0.8),
    ]

    summary = aggregate_runs(tmp_path, run_names)

    metric = summary["aggregate"]["metrics/mAP50-95(B)"]
    assert summary["valid"] is True
    assert summary["seeds"] == [42, 43, 44]
    assert metric["mean"] == pytest.approx(0.7)
    assert metric["sample_std"] == pytest.approx(0.1)
    assert metric["mean_minus_rgb_baseline_percentage_points"] == pytest.approx(
        20.0
    )


def test_multiseed_aggregate_rejects_duplicate_runs(tmp_path) -> None:
    run_name = _write_run(tmp_path, 42, 0.6)

    with pytest.raises(ValueError, match="run names must be unique"):
        aggregate_runs(tmp_path, [run_name, run_name])


def test_multiseed_aggregate_detects_protocol_drift(tmp_path) -> None:
    first = _write_run(tmp_path, 42, 0.6)
    second = _write_run(tmp_path, 43, 0.7)
    protocol_path = tmp_path / second / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["train_args"]["batch"] = 64
    _write_json(protocol_path, protocol)

    summary = aggregate_runs(tmp_path, [first, second])

    assert summary["valid"] is False
    assert "normalized training protocols differ across runs" in summary["errors"]


def test_multiseed_aggregate_binds_root_protocol(tmp_path) -> None:
    run_names = [
        _write_run(tmp_path, 42, 0.6),
        _write_run(tmp_path, 43, 0.7),
    ]
    protocol_path = tmp_path / "MULTISEED_PROTOCOL.json"
    _write_json(
        protocol_path,
        {
            "run_names": run_names,
            "declaration_status": "test-declaration",
        },
    )

    summary = aggregate_runs(tmp_path, run_names)

    assert summary["valid"] is True
    assert summary["multiseed_protocol"]["path"] == protocol_path.as_posix()
    assert (
        summary["multiseed_protocol"]["declaration_status"]
        == "test-declaration"
    )
