from __future__ import annotations

import json

import pytest

from mmfc.reliability_gate_diagnostics import (
    compute_gate_diagnostics,
    load_gate_predictions,
)


def test_gate_diagnostics_detect_ordered_faults(tmp_path) -> None:
    scenarios = ("clean", "noise_low", "noise_high")
    values = {
        "clean": (0.95, 0.90, 0.85),
        "noise_low": (0.75, 0.70, 0.65),
        "noise_high": (0.25, 0.20, 0.15),
    }
    for seed in (1, 2):
        for scenario in scenarios:
            directory = tmp_path / "seeds" / f"seed-{seed}" / scenario
            directory.mkdir(parents=True)
            lines = [
                json.dumps(
                    {
                        "sample_id": f"sample-{index}",
                        "sensor_reliability": value,
                    }
                )
                for index, value in enumerate(values[scenario])
            ]
            (directory / "predictions.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    loaded = load_gate_predictions(
        tmp_path, seeds=(1, 2), scenarios=scenarios
    )
    report = compute_gate_diagnostics(loaded, scenarios=scenarios)

    assert report["valid"] is True
    assert report["sample_count"] == 3
    assert report["fault_detection"][0]["auroc"]["mean"] == pytest.approx(1.0)
    assert report["fault_detection"][1]["auprc"]["mean"] == pytest.approx(1.0)
    assert report["distribution_summary"][0]["median"] == pytest.approx(0.90)


def test_gate_prediction_loader_rejects_misaligned_samples(tmp_path) -> None:
    for scenario, sample_id in (("clean", "a"), ("fault", "b")):
        directory = tmp_path / "seeds" / "seed-1" / scenario
        directory.mkdir(parents=True)
        (directory / "predictions.jsonl").write_text(
            json.dumps(
                {"sample_id": sample_id, "sensor_reliability": 0.5}
            )
            + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="misaligned"):
        load_gate_predictions(
            tmp_path, seeds=(1,), scenarios=("clean", "fault")
        )
