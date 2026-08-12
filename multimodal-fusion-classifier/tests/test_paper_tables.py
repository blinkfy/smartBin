import json
from pathlib import Path

import pytest

import build_paper_tables as paper_tables_module
from build_paper_tables import (
    MAIN_MODELS,
    _percent,
    build_paper_tables,
    classification_row,
    flatten_scenarios,
    paired_comparison_row,
)
from plot_paper_tables import FIGURE_FILENAMES


def test_classification_row_preserves_metrics_intervals_and_epoch():
    report = {
        "metrics": {
            "sample_count": 826,
            "accuracy": 0.8,
            "macro_precision": 0.7,
            "macro_recall": 0.6,
            "macro_f1": 0.65,
            "weighted_f1": 0.79,
            "ece": 0.03,
            "negative_log_likelihood": 0.4,
            "multiclass_brier_score": 0.2,
        },
        "confidence_intervals": {
            "intervals": {
                "accuracy": {"lower": 0.75, "upper": 0.85},
                "macro_f1": {"lower": 0.55, "upper": 0.72},
            }
        },
        "performance": {
            "active_parameter_count": 10,
            "amortized_forward_ms_per_sample": 1.2,
        },
        "checkpoint": {"epoch": 9},
    }

    row = classification_row("model", "Model", report)

    assert row["sample_count"] == 826
    assert row["accuracy_ci_lower"] == 0.75
    assert row["macro_f1_ci_upper"] == 0.72
    assert row["checkpoint_epoch"] == 9

    report.pop("confidence_intervals")
    row_without_intervals = classification_row("legacy", "Legacy", report)
    assert row_without_intervals["accuracy_ci_lower"] is None
    assert _percent(row_without_intervals["accuracy_ci_lower"]) == "NA"


def test_flatten_scenarios_keeps_threshold_and_prediction_source():
    metrics = {
        "source_image_count": 570,
        "ground_truth_count": 825,
        "detection_count": 800,
        "localization_precision": 0.8,
        "localization_recall": 0.7,
        "classification_accuracy_on_localized": 0.9,
        "end_to_end_precision": 0.72,
        "end_to_end_recall": 0.63,
        "macro_f1": 0.6,
    }

    rows = flatten_scenarios({"0.25": {"detector": metrics}})

    assert rows == [
        {
            "confidence_threshold": 0.25,
            "prediction_source": "detector",
            **metrics,
        }
    ]


def test_paired_comparison_row_uses_candidate_minus_reference():
    comparison = {
        "reference": {"name": "qwen"},
        "candidate": {"name": "rule-v1"},
        "outcome_changes": {
            "candidate_only_correct": 4,
            "reference_only_correct": 2,
        },
        "mcnemar_exact": {"p_value": 0.5},
        "paired_bootstrap": {
            "differences": {
                "accuracy": {
                    "point": 0.02,
                    "lower": -0.01,
                    "upper": 0.04,
                },
                "macro_f1": {
                    "point": 0.03,
                    "lower": 0.0,
                    "upper": 0.05,
                },
            }
        },
    }

    row = paired_comparison_row("rule_v1_vs_qwen", comparison)

    assert row["accuracy_delta_pp"] == pytest.approx(2.0)
    assert row["macro_f1_ci_upper_pp"] == pytest.approx(5.0)
    assert row["candidate_only_correct"] == 4


def test_build_paper_tables_writes_hashed_artifact_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    reports = tmp_path / "reports"

    def write_json(relative_path: str, value: dict) -> None:
        path = reports / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    write_json(
        "FINAL_EVIDENCE_AUDIT.json",
        {"state": "complete", "valid": True},
    )
    classification = {
        "metrics": {
            "sample_count": 826,
            "accuracy": 0.8,
            "macro_precision": 0.7,
            "macro_recall": 0.6,
            "macro_f1": 0.65,
            "weighted_f1": 0.79,
            "ece": 0.03,
            "negative_log_likelihood": 0.4,
            "multiclass_brier_score": 0.2,
        },
        "performance": {
            "active_parameter_count": 10,
            "amortized_forward_ms_per_sample": 1.2,
        },
        "checkpoint": {"epoch": 9},
    }
    for _, _, relative_path in MAIN_MODELS:
        write_json(relative_path, classification)
    write_json(
        "ablations/ablation_summary.json",
        {
            "protocol": "formal_retrained",
            "summary": [
                {
                    "experiment": "full",
                    "accuracy": 0.8,
                    "accuracy_delta_pp": 0.0,
                    "macro_f1": 0.65,
                    "macro_f1_delta_pp": 0.0,
                }
            ],
        },
    )
    write_json(
        "ablations/ablation_pairwise_statistics.json",
        {
            "summary": [
                {
                    "experiment": "vision_only",
                    "accuracy_delta_pp": -1.0,
                    "accuracy_ci_lower_pp": -2.0,
                    "accuracy_ci_upper_pp": 0.0,
                    "macro_f1_delta_pp": -1.5,
                    "macro_f1_ci_lower_pp": -3.0,
                    "macro_f1_ci_upper_pp": 0.0,
                    "mcnemar_holm_p_value": 0.2,
                }
            ]
        },
    )
    write_json(
        "yolo-seg/evaluation.json",
        {
            "dataset": {
                "test_source_images": 570,
                "test_objects": 825,
            },
            "metrics": {
                "metrics/precision(B)": 0.8,
                "metrics/recall(B)": 0.7,
                "metrics/mAP50(B)": 0.75,
                "metrics/mAP50-95(B)": 0.5,
                "metrics/precision(M)": 0.78,
                "metrics/recall(M)": 0.68,
                "metrics/mAP50(M)": 0.72,
                "metrics/mAP50-95(M)": 0.45,
            },
        },
    )
    scene_metrics = {
        "source_image_count": 570,
        "ground_truth_count": 825,
        "detection_count": 800,
        "localization_precision": 0.8,
        "localization_recall": 0.7,
        "classification_accuracy_on_localized": 0.9,
        "end_to_end_precision": 0.72,
        "end_to_end_recall": 0.63,
        "macro_f1": 0.6,
    }
    write_json(
        "yolo-seg/integrated/evaluation.json",
        {"scenarios": {"0.25": {"detector": scene_metrics}}},
    )
    write_json(
        "yolo-seg/integrated/stratified/evaluation.json",
        {
            "rows": [
                {
                    "confidence_threshold": 0.25,
                    "prediction_source": "detector",
                    "dimension": "scene_density",
                    "stratum": "single_object",
                    "support": 486,
                    "localization_recall": 0.7,
                    "end_to_end_recall": 0.6,
                }
            ]
        },
    )
    comparison = {
        "reference": {"name": "qwen"},
        "candidate": {"name": "rule-v1"},
        "outcome_changes": {
            "candidate_only_correct": 4,
            "reference_only_correct": 2,
        },
        "mcnemar_exact": {"p_value": 0.5},
        "paired_bootstrap": {
            "differences": {
                "accuracy": {
                    "point": 0.02,
                    "lower": -0.01,
                    "upper": 0.04,
                },
                "macro_f1": {
                    "point": 0.03,
                    "lower": 0.0,
                    "upper": 0.05,
                },
            }
        },
    }
    write_json(
        "full_cross_modal_quality/PAIRED_COMPARISON.json",
        {
            **comparison,
            "reference": {"name": "full_quality_aware_prior"},
            "candidate": {"name": "full_cross_modal_quality"},
        },
    )
    write_json(
        "full_cross_modal_quality/GATE_PAIRED_COMPARISON.json",
        {
            **comparison,
            "reference": {"name": "same_checkpoint_no_gate"},
            "candidate": {"name": "full_cross_modal_quality_gate"},
        },
    )
    write_json(
        "sensor-source-transfer/evaluation.json",
        {
            "sources": {"qwen": {"metrics": classification["metrics"]}},
            "paired_comparisons": {"rule_v1_vs_qwen": comparison},
        },
    )
    write_json(
        "sensor-source-retraining/rule-v1/test/evaluation.json",
        classification,
    )
    write_json(
        (
            "sensor-source-retraining/"
            "RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json"
        ),
        comparison,
    )

    output = reports / "paper-tables"
    result = build_paper_tables(
        reports_root=reports,
        output_dir=output,
    )
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["state"] == "complete"
    assert manifest["schema_version"] == 2
    assert manifest["formal_ablation_scope"] == (
        "legacy_full_reference_not_v3_component_ablation"
    )
    assert manifest["main_paired_evidence_count"] == 2
    assert len(manifest["source_reports"]) == 15
    assert all(
        not Path(entry["path"]).is_absolute()
        for entry in manifest["source_reports"].values()
    )
    assert manifest["figures"] == list(FIGURE_FILENAMES)
    assert len(manifest["artifact_sha256"]) == 16
    assert all(
        len(value) == 64
        for value in manifest["artifact_sha256"].values()
    )
    assert all(
        (output / filename).stat().st_size > 10_000
        for filename in FIGURE_FILENAMES
    )
    assert "NA" in (output / "TABLES.md").read_text(encoding="utf-8")
    assert "旧 full 参照" in (
        output / "TABLES.md"
    ).read_text(encoding="utf-8")
    assert "v3 - v2b" in (
        output / "TABLES.md"
    ).read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_paper_tables(reports_root=reports, output_dir=output)

    failed_output = reports / "failed-paper-tables"
    write_count = 0
    original_write_csv = paper_tables_module._write_csv

    def fail_during_csv_write(path: Path, rows: list[dict]) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("simulated table write interruption")
        original_write_csv(path, rows)

    monkeypatch.setattr(
        paper_tables_module,
        "_write_csv",
        fail_during_csv_write,
    )
    with pytest.raises(RuntimeError, match="simulated"):
        build_paper_tables(
            reports_root=reports,
            output_dir=failed_output,
        )
    assert not failed_output.exists()
    assert not list(reports.glob(".failed-paper-tables.tmp-*"))
