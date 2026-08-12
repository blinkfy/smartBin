import json
from pathlib import Path

import pytest

from analyze_class_metric_intervals import build_class_metric_intervals
from audit_paper_readiness import audit_class_metric_intervals_bundle


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_class_metric_intervals_use_source_image_clusters(tmp_path):
    evaluation_path = tmp_path / "evaluation.json"
    predictions_path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "sample_id": "a1",
            "source_image_id": "source-a",
            "target_id": 0,
            "target": "alpha",
            "predicted_id": 0,
            "predicted": "alpha",
        },
        {
            "sample_id": "a2",
            "source_image_id": "source-a",
            "target_id": 1,
            "target": "beta",
            "predicted_id": 0,
            "predicted": "alpha",
        },
        {
            "sample_id": "b1",
            "source_image_id": "source-b",
            "target_id": 1,
            "target": "beta",
            "predicted_id": 1,
            "predicted": "beta",
        },
    ]
    _write_jsonl(predictions_path, rows)
    evaluation_path.write_text(
        json.dumps(
            {
                "dataset": {"evaluated_samples": 3},
                "class_names": ["alpha", "beta"],
                "per_class": [
                    {
                        "label": "alpha",
                        "precision": 0.5,
                        "recall": 1.0,
                        "f1": 2.0 / 3.0,
                    },
                    {
                        "label": "beta",
                        "precision": 1.0,
                        "recall": 0.5,
                        "f1": 2.0 / 3.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    first = build_class_metric_intervals(
        evaluation_path=evaluation_path,
        predictions_path=predictions_path,
        repeats=200,
        confidence=0.95,
        seed=7,
    )
    second = build_class_metric_intervals(
        evaluation_path=evaluation_path,
        predictions_path=predictions_path,
        repeats=200,
        confidence=0.95,
        seed=7,
    )

    assert first == second
    assert first["valid"] is True
    assert first["dataset"] == {
        "sample_count": 3,
        "source_image_count": 2,
        "class_count": 2,
    }
    assert first["method"]["resampling_unit"] == "source_image_id"
    assert first["metrics"][0]["support"] == 1
    assert first["metrics"][1]["support"] == 2
    assert first["metrics"][1]["source_image_count"] == 2
    assert (
        first["metrics"][0]["intervals"]["recall"]["undefined_repeats"] > 0
    )
    assert (
        first["metrics"][0]["intervals"]["f1"]["undefined_repeats"]
        == first["metrics"][0]["intervals"]["recall"]["undefined_repeats"]
    )


def test_class_metric_intervals_reject_sample_count_mismatch(tmp_path):
    evaluation_path = tmp_path / "evaluation.json"
    predictions_path = tmp_path / "predictions.jsonl"
    evaluation_path.write_text(
        json.dumps(
            {
                "dataset": {"evaluated_samples": 2},
                "class_names": ["alpha"],
                "per_class": [],
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "sample_id": "a",
                "source_image_id": "source-a",
                "target_id": 0,
                "target": "alpha",
                "predicted_id": 0,
                "predicted": "alpha",
            }
        ],
    )

    with pytest.raises(ValueError, match="sample count mismatch"):
        build_class_metric_intervals(
            evaluation_path=evaluation_path,
            predictions_path=predictions_path,
            repeats=10,
            confidence=0.95,
            seed=7,
        )


def test_class_metric_interval_audit_recomputes_bound_predictions(tmp_path):
    project_root = tmp_path / "project"
    evaluation_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "full_cross_modal_quality"
        / "test"
        / "evaluation.json"
    )
    predictions_path = evaluation_path.with_name("predictions.jsonl")
    predictions_path.parent.mkdir(parents=True)
    rows = [
        {
            "sample_id": "a",
            "source_image_id": "source-a",
            "target_id": 0,
            "target": "alpha",
            "predicted_id": 0,
            "predicted": "alpha",
        },
        {
            "sample_id": "b",
            "source_image_id": "source-b",
            "target_id": 1,
            "target": "beta",
            "predicted_id": 1,
            "predicted": "beta",
        },
    ]
    _write_jsonl(predictions_path, rows)
    evaluation_path.write_text(
        json.dumps(
            {
                "dataset": {"evaluated_samples": 2},
                "class_names": ["alpha", "beta"],
                "per_class": [
                    {
                        "label": "alpha",
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                    {
                        "label": "beta",
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report = build_class_metric_intervals(
        evaluation_path=evaluation_path,
        predictions_path=predictions_path,
        repeats=2000,
        confidence=0.95,
        seed=20260727,
    )
    report_path = (
        project_root
        / "reports"
        / "qwen-material"
        / "CLASS_METRIC_INTERVALS.json"
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_path.with_suffix(".md").write_text("intervals", encoding="utf-8")

    result = audit_class_metric_intervals_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is True
    assert result["sample_count"] == 2
    assert result["class_count"] == 2
    assert result["minimum_class_support"] == 1

    rows[0]["predicted_id"] = 1
    rows[0]["predicted"] = "beta"
    _write_jsonl(predictions_path, rows)
    result = audit_class_metric_intervals_bundle(
        report_path,
        project_root=project_root,
    )
    assert result["valid"] is False
    assert any("drift" in error for error in result["errors"])
