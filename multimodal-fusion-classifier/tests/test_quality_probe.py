from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mmfc.quality_probe import (
    evaluate_quality_probe,
    fit_quality_probe,
    load_split_sensor_grids,
    quality_regression_metrics,
    write_quality_probe_artifacts,
)


def test_quality_regression_metrics_match_known_values() -> None:
    metrics = quality_regression_metrics(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.1, 0.4, 0.8]),
    )

    assert metrics["target_mean"] == pytest.approx(0.5)
    assert metrics["prediction_mean"] == pytest.approx(1.3 / 3.0)
    assert metrics["mae"] == pytest.approx(0.4 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.06 / 3.0))
    assert metrics["pearson"] is not None


def test_quality_probe_train_evaluate_and_artifacts_smoke(tmp_path) -> None:
    generator = torch.Generator().manual_seed(17)
    train_grids = torch.rand(64, 1, 4, 4, generator=generator)
    test_grids = torch.rand(16, 1, 4, 4, generator=generator)
    model, history = fit_quality_probe(
        train_grids,
        {
            "enabled": True,
            "missing_probability": 0.1,
            "noise_probability": 0.8,
            "noise_std_range": [0.02, 0.15],
            "stuck_probability": 0.2,
            "quality_scale": 0.25,
        },
        device=torch.device("cpu"),
        seed=29,
        epochs=3,
        batch_size=16,
    )
    rows = evaluate_quality_probe(
        model,
        test_grids,
        device=torch.device("cpu"),
        seed=31,
        scenarios={
            "clean": None,
            "missing": {
                "enabled": True,
                "missing_probability": 1.0,
            },
            "noise": {
                "enabled": True,
                "noise_probability": 1.0,
                "noise_std": 0.1,
            },
        },
    )

    assert len(history) == 3
    assert all(np.isfinite(float(item["loss"])) for item in history)
    assert [row["scenario"] for row in rows] == [
        "clean",
        "missing",
        "noise",
    ]
    assert all(row["sample_count"] == 16 for row in rows)
    assert all(np.isfinite(float(row["mae"])) for row in rows)

    artifacts = write_quality_probe_artifacts(
        model,
        history,
        rows,
        tmp_path,
        metadata={"scope": "test"},
    )
    assert set(artifacts) == {
        "summary_csv",
        "summary_json",
        "checkpoint",
        "plot_png",
    }
    for value in artifacts.values():
        assert Path(value).is_file()
    checkpoint = torch.load(
        tmp_path / "quality_probe.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["feature_names"][-1] == "flatness"


def test_load_split_sensor_grids_preserves_manifest_order(tmp_path) -> None:
    manifest = tmp_path / "split.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "second"}),
                json.dumps({"sample_id": "first"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "sensors.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "first",
                        "normalized_grid": np.zeros((4, 4)).tolist(),
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "second",
                        "normalized_grid": np.ones((4, 4)).tolist(),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    grids = load_split_sensor_grids(manifest, sidecar)

    assert grids.shape == (2, 1, 4, 4)
    torch.testing.assert_close(grids[0], torch.ones(1, 4, 4))
    torch.testing.assert_close(grids[1], torch.zeros(1, 4, 4))
