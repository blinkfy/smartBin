from __future__ import annotations

from write_class_balance_report import build_report


def _evaluation(accuracy: float, macro_f1: float) -> dict:
    return {
        "metrics": {
            "accuracy": accuracy,
            "macro_precision": macro_f1,
            "macro_recall": macro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": accuracy,
            "negative_log_likelihood": 0.3,
            "multiclass_brier_score": 0.15,
            "ece": 0.02,
        }
    }


def test_class_balance_report_states_pass_without_significance_claim():
    report = build_report(
        training_summary={
            "completed_epochs": 100,
            "best_checkpoint_epoch": 90,
            "best_selection_score": 0.84,
            "best_checkpoint_val_accuracy": 0.89,
            "config_fingerprint": "abc",
            "history": [
                {
                    "train_accuracy": 0.92,
                    "val_accuracy": 0.88,
                    "val_macro_f1": 0.83,
                }
            ],
        },
        reference_val=_evaluation(0.88, 0.82),
        candidate_val=_evaluation(0.89, 0.84),
        reference_test=_evaluation(0.88, 0.82),
        candidate_test=_evaluation(0.875, 0.83),
        paired={
            "outcome_changes": {
                "candidate_only_correct": 10,
                "reference_only_correct": 9,
            },
            "mcnemar_exact": {"p_value": 1.0},
            "paired_bootstrap": {
                "differences": {
                    "accuracy": {"lower": -0.01, "upper": 0.02},
                    "macro_f1": {"lower": -0.02, "upper": 0.03},
                }
            },
        },
        decision={
            "passed": True,
            "criteria": {
                "validation_macro_f1_non_decrease": {
                    "reference": 0.82,
                    "candidate": 0.84,
                    "delta": 0.02,
                    "passed": True,
                },
                "test_macro_f1_non_decrease": {
                    "reference": 0.82,
                    "candidate": 0.83,
                    "delta": 0.01,
                    "passed": True,
                },
                "test_accuracy_drop_within_limit": {
                    "reference": 0.88,
                    "candidate": 0.875,
                    "delta": -0.005,
                    "passed": True,
                },
            },
        },
        checkpoint_audit={"valid": True},
        leakage_audit={
            "prediction_sensitivity": {
                "balanced": {
                    "all": {"accuracy": 0.875, "macro_f1": 0.83},
                    "after_high_confidence_exclusion": {
                        "accuracy": 0.874,
                        "macro_f1": 0.829,
                    },
                    "excluded_source_count": 8,
                    "excluded_sample_count": 9,
                }
            },
            "exact_object_duplicate_prediction_sensitivity": {
                "balanced": {
                    "all": {"accuracy": 0.875, "macro_f1": 0.83},
                    "after_exact_object_duplicate_exclusion": {
                        "accuracy": 0.876,
                        "macro_f1": 0.834,
                    },
                    "excluded_sample_count": 1,
                }
            }
        },
    )

    assert "扩展种子 43/44" in report
    assert "不能声称统计显著提升" in report
    assert "固定留出测试" in report
    assert "近重复剔除敏感性" in report
    assert "完全重复对象敏感性" in report
    assert "../class_performance_v2b_v3_balanced.png" in report
