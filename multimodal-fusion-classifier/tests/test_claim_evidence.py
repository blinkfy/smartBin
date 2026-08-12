import hashlib
import json
from pathlib import Path

import pytest

from build_claim_evidence import (
    CLAIM_TABLE_SOURCE_NAMES,
    _source_paths,
    build_claim_evidence,
    metric_interval_assessment,
    paired_evidence,
)


def test_metric_interval_assessment_respects_metric_direction():
    positive = {"lower": 0.01, "upper": 0.03}
    negative = {"lower": -0.03, "upper": -0.01}
    crossing = {"lower": -0.01, "upper": 0.02}

    assert metric_interval_assessment("accuracy", positive) == (
        "candidate_better"
    )
    assert metric_interval_assessment("accuracy", negative) == (
        "candidate_worse"
    )
    assert metric_interval_assessment("ece", positive) == "candidate_worse"
    assert metric_interval_assessment("ece", negative) == "candidate_better"
    assert metric_interval_assessment("macro_f1", crossing) == (
        "interval_inconclusive"
    )


def test_paired_evidence_preserves_direction_counts_and_mcnemar():
    comparison = {
        "reference": {"name": "qwen"},
        "candidate": {"name": "rule-v1"},
        "mcnemar_exact": {"p_value": 0.25},
        "paired_bootstrap": {
            "sample_count": 826,
            "cluster_count": 570,
            "repeats": 5000,
            "differences": {
                "accuracy": {
                    "point": 0.02,
                    "lower": 0.01,
                    "upper": 0.03,
                },
                "ece": {
                    "point": -0.01,
                    "lower": -0.02,
                    "upper": -0.001,
                },
            },
        },
    }

    result = paired_evidence("rule_vs_qwen", comparison)

    assert result["reference"] == "qwen"
    assert result["candidate"] == "rule-v1"
    assert result["bootstrap_repeats"] == 5000
    assert result["metrics"]["accuracy"]["assessment"] == (
        "candidate_better"
    )
    assert result["metrics"]["ece"]["assessment"] == "candidate_better"


def test_build_claim_evidence_consumes_complete_audited_bundle(
    tmp_path: Path,
):
    reports = tmp_path / "reports"

    def write(relative: str, value: dict) -> Path:
        path = reports / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    final_audit = write(
        "FINAL_EVIDENCE_AUDIT.json",
        {"state": "complete", "valid": True},
    )
    table_manifest_path = write(
        "paper-tables/manifest.json",
        {
            "source_audit_valid": True,
            "source_audit_sha256": hashlib.sha256(
                final_audit.read_bytes()
            ).hexdigest(),
        },
    )
    write(
        "ablations/ablation_summary.json",
        {
            "protocol": "formal_retrained",
            "summary": [{} for _ in range(9)],
        },
    )
    write(
        "ablations/ablation_pairwise_statistics.json",
        {"summary": [{} for _ in range(8)]},
    )
    write(
        "yolo-seg/evaluation.json",
        {
            "metrics": {
                "metrics/mAP50(B)": 0.5,
                "metrics/mAP50(M)": 0.4,
            }
        },
    )
    write(
        "yolo-seg/integrated/evaluation.json",
        {"scenarios": {"0.25": {"detector": {"macro_f1": 0.5}}}},
    )
    write(
        "yolo-seg/integrated/stratified/evaluation.json",
        {
            "protocol": "preregistered_scene_stratified_analysis",
            "dataset": {
                "source_image_count": 570,
                "ground_truth_object_count": 825,
            },
            "rows": [{} for _ in range(42)],
        },
    )
    metrics = {
        "accuracy": 0.8,
        "macro_f1": 0.7,
        "ece": 0.03,
        "negative_log_likelihood": 0.4,
        "multiclass_brier_score": 0.2,
    }
    differences = {
        metric: {
            "point": 0.01,
            "lower": -0.01,
            "upper": 0.03,
        }
        for metric in metrics
    }
    comparison = {
        "reference": {"name": "qwen"},
        "candidate": {"name": "rule-v1"},
        "mcnemar_exact": {"p_value": 0.5},
        "paired_bootstrap": {
            "sample_count": 826,
            "cluster_count": 570,
            "repeats": 5000,
            "differences": differences,
        },
    }
    write(
        "sensor-source-transfer/evaluation.json",
        {
            "protocol": "same_checkpoint_sensor_source_transfer",
            "sample_count": 826,
            "source_image_count": 570,
            "sources": {
                source: {"metrics": metrics}
                for source in ("qwen", "rule-v1", "missing")
            },
            "paired_comparisons": {
                "rule_v1_vs_qwen": comparison,
            },
        },
    )
    write(
        "full_cross_modal_quality/test/evaluation.json",
        {"metrics": metrics},
    )
    write(
        "sensor-source-retraining/rule-v1/test/evaluation.json",
        {"metrics": metrics},
    )
    write(
        (
            "sensor-source-retraining/"
            "RULE_V1_VS_QWEN_RETRAINED_COMPARISON.json"
        ),
        comparison,
    )

    source_paths = _source_paths(reports)
    table_manifest = json.loads(
        table_manifest_path.read_text(encoding="utf-8")
    )
    table_manifest["source_reports"] = {
        table_name: {
            "path": source_paths[claim_name].as_posix(),
            "sha256": hashlib.sha256(
                source_paths[claim_name].read_bytes()
            ).hexdigest(),
        }
        for claim_name, table_name in CLAIM_TABLE_SOURCE_NAMES.items()
    }
    table_manifest_path.write_text(
        json.dumps(table_manifest),
        encoding="utf-8",
    )

    result = build_claim_evidence(reports)
    output = json.loads(
        (reports / "FINAL_CLAIM_EVIDENCE.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["claim_group_count"] == 5
    assert output["state"] == "complete"
    assert all(
        not Path(entry["path"]).is_absolute()
        for entry in output["source_hashes"].values()
    )
    assert output["claims"]["formal_ablations"][
        "experiment_count"
    ] == 9
    assert output["claims"]["formal_ablations"]["reference_model"] == "full"
    assert "not a component-wise ablation of v3" in output["claims"][
        "formal_ablations"
    ]["claim_boundary"]
    assert output["claims"]["C14_scene_strata"]["row_count"] == 42
    assert output["claims"][
        "C15_independent_sensor_retraining"
    ]["comparison"]["metrics"]["accuracy"]["assessment"] == (
        "interval_inconclusive"
    )
    assert (reports / "FINAL_CLAIM_EVIDENCE.md").is_file()

    source_paths["formal_summary"].write_text(
        json.dumps({"protocol": "changed", "summary": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_claim_evidence(reports)
