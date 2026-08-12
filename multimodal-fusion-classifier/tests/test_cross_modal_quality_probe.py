from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mmfc.cross_modal_quality_probe import (
    CROSS_MODAL_QUALITY_DIM,
    aggregate_cross_modal_quality_runs,
    cross_modal_grid_quality_features,
    evaluate_cross_modal_quality_probe,
    fit_cross_modal_quality_probe,
    write_cross_modal_quality_probe_artifacts,
)


def test_cross_modal_grid_quality_features_validate_and_report_shape() -> None:
    predicted = torch.rand(4, 1, 4, 4)
    observed = torch.rand(4, 1, 4, 4)
    availability = torch.ones(4, 1)

    features = cross_modal_grid_quality_features(
        predicted,
        observed,
        availability,
    )

    assert features.shape == (4, CROSS_MODAL_QUALITY_DIM)
    assert torch.isfinite(features).all()
    with pytest.raises(ValueError, match="availability"):
        cross_modal_grid_quality_features(
            predicted,
            observed,
            torch.ones(3, 1),
        )


def test_cross_modal_quality_probe_train_evaluate_and_write(tmp_path) -> None:
    generator = torch.Generator().manual_seed(8)
    train_latents = torch.rand(64, 12, generator=generator)
    train_grids = train_latents[:, :12].mean(dim=1, keepdim=True)
    train_grids = train_grids[:, :, None, None].expand(-1, 1, 4, 4).clone()
    test_latents = torch.rand(16, 12, generator=generator)
    test_grids = test_latents[:, :12].mean(dim=1, keepdim=True)
    test_grids = test_grids[:, :, None, None].expand(-1, 1, 4, 4).clone()

    decoder, quality_probe, history = fit_cross_modal_quality_probe(
        train_latents,
        train_grids,
        {
            "enabled": True,
            "missing_probability": 0.1,
            "noise_probability": 0.8,
            "noise_std_range": [0.02, 0.15],
            "bias_probability": 0.2,
            "bias_strength": 0.1,
            "stuck_probability": 0.2,
            "quality_scale": 0.25,
        },
        device=torch.device("cpu"),
        seed=13,
        decoder_epochs=3,
        quality_epochs=3,
        batch_size=16,
    )
    rows, decoder_metrics = evaluate_cross_modal_quality_probe(
        decoder,
        quality_probe,
        test_latents,
        test_grids,
        device=torch.device("cpu"),
        seed=17,
        batch_size=8,
        scenarios={
            "clean": None,
            "noise": {
                "enabled": True,
                "noise_probability": 1.0,
                "noise_std": 0.1,
            },
        },
    )

    assert len(history) == 6
    assert decoder_metrics["mae"] >= 0.0
    assert [row["scenario"] for row in rows] == ["clean", "noise"]
    assert all(np.isfinite(float(row["mae"])) for row in rows)

    artifacts = write_cross_modal_quality_probe_artifacts(
        decoder,
        quality_probe,
        history,
        rows,
        tmp_path,
        metadata={"scope": "test"},
        decoder_metrics=decoder_metrics,
    )
    assert set(artifacts) == {
        "summary_csv",
        "summary_json",
        "checkpoint",
        "plot_png",
    }
    assert all(Path(value).is_file() for value in artifacts.values())
    checkpoint = torch.load(
        tmp_path / "cross_modal_quality_probe.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert len(checkpoint["feature_names"]) == CROSS_MODAL_QUALITY_DIM


def test_aggregate_cross_modal_quality_runs_uses_sample_std() -> None:
    runs = {
        1: [
            {
                "scenario": "noise",
                "sample_count": 8,
                "mae": 0.1,
                "pearson": None,
            }
        ],
        2: [
            {
                "scenario": "noise",
                "sample_count": 8,
                "mae": 0.3,
                "pearson": 0.5,
            }
        ],
    }

    row = aggregate_cross_modal_quality_runs(runs)[0]

    assert row["mae_mean"] == pytest.approx(0.2)
    assert row["mae_std"] == pytest.approx(np.sqrt(0.02))
    assert row["pearson_mean"] == pytest.approx(0.5)
    assert row["pearson_valid_seed_count"] == 1
