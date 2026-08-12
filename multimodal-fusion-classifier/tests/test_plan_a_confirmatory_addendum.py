from __future__ import annotations

import json

import pytest

from audit_recent_robust_baselines import PRIMARY_PAIRED_CONDITIONS
from audit_recent_robust_baselines import AMENDED_METHOD_TRAINING_SEEDS
from build_plan_a_confirmatory_addendum import (
    COMPONENT_VARIANTS,
    _latex,
    _markdown,
    build_outputs,
    compile_confirmatory_bundle,
)
from evaluate_unseen_sensor_conditions import CONDITIONS
from prepare_recent_robust_baselines import METHODS, SEEDS


def _component_outcome():
    variants = {}
    for index, variant in enumerate(COMPONENT_VARIANTS):
        accuracy_mean = 0.86 + index * 0.01
        macro_f1_mean = 0.80 + index * 0.01
        accuracy_effect_mean = 0.01 - index * 0.002
        macro_f1_effect_mean = 0.012 - index * 0.003
        accuracy_effects = (
            [0.008, 0.010, 0.012]
            if index == 0
            else [-0.002, 0.008, 0.018]
        )
        macro_f1_effects = (
            [-0.001, 0.012, 0.025]
            if index == 0
            else [0.001, 0.009, 0.017]
        )
        variants[variant] = {
            "runs": {
                str(seed): {
                    "test_metrics": {
                        "accuracy": accuracy_mean + offset,
                        "macro_f1": macro_f1_mean + 0.8 * offset,
                    },
                    "full_minus_ablation": {
                        "accuracy": accuracy_effects[seed_index],
                        "macro_f1": macro_f1_effects[seed_index],
                    },
                }
                for seed_index, (seed, offset) in enumerate(
                    zip(SEEDS, (-0.01, 0.0, 0.01))
                )
            },
            "ablation_test_mean": {
                "accuracy": accuracy_mean,
                "macro_f1": macro_f1_mean,
            },
            "full_minus_ablation_mean": {
                "accuracy": accuracy_effect_mean,
                "macro_f1": macro_f1_effect_mean,
            },
            "full_minus_ablation_seed_directions": {
                "accuracy_positive": 3 - index,
                "macro_f1_positive": 2 + index,
            },
        }
    return {
        "state": "complete",
        "valid": True,
        "variants": variants,
    }


def _unseen_outcome():
    summary = []
    for index, (condition, contract) in enumerate(CONDITIONS.items()):
        effect = 0.0 if condition == "clean" else -0.5 * index
        summary.append(
            {
                "condition": condition,
                "category": contract["category"],
                "accuracy_mean": 0.88 - index * 0.002,
                "accuracy_sample_std": 0.003,
                "macro_f1_mean": 0.82 - index * 0.003,
                "macro_f1_sample_std": 0.004,
                "condition_minus_clean_accuracy_pp_mean": effect,
                "condition_minus_clean_macro_f1_pp_mean": effect,
                "reported_availability_mean": 1.0,
                "true_quality_mean": max(0.0, 1.0 - index * 0.08),
                "predicted_reliability_mean": max(
                    0.0, 0.9 - index * 0.06
                ),
            }
        )
    return {
        "state": "complete",
        "valid": True,
        "model_seeds": list(SEEDS),
        "conditions": CONDITIONS,
        "summary": summary,
    }


def _paired(point: float):
    return {
        "repeats": 2000,
        "cluster_count": 570,
        "sample_count": 826,
        "differences": {
            "accuracy": {
                "point": point,
                "lower": point - 0.01,
                "upper": point + 0.01,
            },
            "macro_f1": {
                "point": point * 0.8,
                "lower": point * 0.8 - 0.01,
                "upper": point * 0.8 + 0.01,
            },
        },
    }


def _baseline_outcome():
    summary = []
    comparisons = {}
    for method_index, method in enumerate(METHODS):
        method_seeds = AMENDED_METHOD_TRAINING_SEEDS[method]
        for condition_index, (condition, contract) in enumerate(
            CONDITIONS.items()
        ):
            summary.append(
                {
                    "method": method,
                    "condition": condition,
                    "category": contract["category"],
                    "accuracy_mean": (
                        0.85 + method_index * 0.01 - condition_index * 0.002
                    ),
                    "training_seeds": list(method_seeds),
                    "seed_count": len(method_seeds),
                    "evidence_level": (
                        "confirmatory_multiseed"
                        if len(method_seeds) > 1
                        else "reference_single_seed"
                    ),
                    "accuracy_sample_std": (
                        0.005 if len(method_seeds) > 1 else None
                    ),
                    "macro_f1_mean": (
                        0.79 + method_index * 0.01 - condition_index * 0.003
                    ),
                    "macro_f1_sample_std": (
                        0.006 if len(method_seeds) > 1 else None
                    ),
                    "sensor_branch_weight_mean": 0.3 + method_index * 0.1,
                }
            )
        for condition_index, condition in enumerate(
            PRIMARY_PAIRED_CONDITIONS
        ):
            for seed_index, seed in enumerate(method_seeds):
                point = (
                    -0.01
                    + method_index * 0.004
                    + condition_index * 0.001
                    + seed_index * 0.001
                )
                comparisons[
                    f"{method}_seed{seed}_{condition}"
                ] = _paired(point)
    return {
        "state": "complete",
        "valid": True,
        "methods": list(METHODS),
        "training_seeds": list(SEEDS),
        "method_training_seeds": {
            method: list(values)
            for method, values in AMENDED_METHOD_TRAINING_SEEDS.items()
        },
        "run_count": 4,
        "primary_paired_conditions": list(PRIMARY_PAIRED_CONDITIONS),
        "summary": summary,
        "paired_candidate_minus_v3": comparisons,
    }


def test_compiles_amended_baseline_scope_without_fake_replication():
    bundle = compile_confirmatory_bundle(
        _component_outcome(),
        _unseen_outcome(),
        _baseline_outcome(),
    )

    assert bundle["valid"] is True
    assert len(bundle["component_ablations"]) == 2
    assert len(bundle["unseen_sensor_conditions"]) == len(CONDITIONS)
    assert len(bundle["recent_robust_baselines"]) == (
        len(METHODS) * len(CONDITIONS)
    )
    assert len(bundle["paired_candidate_minus_v3"]) == (
        len(METHODS) * len(PRIMARY_PAIRED_CONDITIONS)
    )
    assert bundle["decision_summary"][
        "worst_unseen_macro_f1_condition"
    ] == list(CONDITIONS)[-1]
    qmf_clean = next(
        row
        for row in bundle["paired_candidate_minus_v3"]
        if row["method"] == "qmf" and row["condition"] == "clean"
    )
    assert qmf_clean["candidate_minus_v3_accuracy_pp_mean"] == pytest.approx(
        -0.9
    )
    assert qmf_clean["macro_f1_candidate_positive_seed_count"] == 0
    assert len(qmf_clean["per_seed_cluster_bootstrap"]) == 3
    tmc_clean = next(
        row
        for row in bundle["paired_candidate_minus_v3"]
        if row["method"] == "tmc" and row["condition"] == "clean"
    )
    assert tmc_clean["seed_count"] == 1
    assert tmc_clean[
        "candidate_minus_v3_accuracy_pp_sample_std"
    ] is None
    assert len(tmc_clean["per_seed_cluster_bootstrap"]) == 1
    component = bundle["component_ablations"][0]
    assert component["ablation_accuracy_sample_std"] == pytest.approx(0.01)
    assert component[
        "full_minus_ablation_macro_f1_pp_sample_std"
    ] == pytest.approx(1.3)


def test_addendum_text_preserves_direction_and_citation_boundaries():
    bundle = compile_confirmatory_bundle(
        _component_outcome(),
        _unseen_outcome(),
        _baseline_outcome(),
    )
    markdown = _markdown(bundle)
    latex = _latex(bundle)

    assert "候选-v3" in markdown
    assert "软件注入故障称为真实硬件故障验证" in markdown
    assert "\\cite{zhang2023quality}" in latex
    assert "\\cite{han2021trusted}" in latex
    assert "不是原作者数据与编码器的逐字复现" in latex
    assert "TMC适配" in latex
    assert "单种子参考" in latex
    assert "n=1; SD=N/A" in markdown
    assert "样本标准差" in latex
    assert "Accuracy和Macro F1的正向种子数分别为" in latex
    assert "3/3和2/3" in latex
    assert "均值±SD" in markdown


def test_rejects_component_aggregate_mean_drift():
    component = _component_outcome()
    component["variants"][COMPONENT_VARIANTS[0]][
        "ablation_test_mean"
    ]["accuracy"] += 0.01

    with pytest.raises(ValueError, match="three seed runs"):
        compile_confirmatory_bundle(
            component,
            _unseen_outcome(),
            _baseline_outcome(),
        )


def test_rejects_incomplete_or_drifted_outcome():
    unseen = _unseen_outcome()
    unseen["model_seeds"] = [42]

    with pytest.raises(ValueError, match="unseen-condition"):
        compile_confirmatory_bundle(
            _component_outcome(),
            unseen,
            _baseline_outcome(),
        )


def test_writes_bound_machine_and_manuscript_artifacts(tmp_path):
    inputs = {}
    for name, value in (
        ("component", _component_outcome()),
        ("unseen", _unseen_outcome()),
        ("baseline", _baseline_outcome()),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(value, ensure_ascii=False),
            encoding="utf-8",
        )
        inputs[name] = path
    output_dir = tmp_path / "output"

    bundle = build_outputs(
        inputs["component"],
        inputs["unseen"],
        inputs["baseline"],
        output_dir,
    )

    assert bundle["valid"] is True
    assert (output_dir / "OUTCOME.json").is_file()
    assert (output_dir / "OUTCOME.md").is_file()
    assert (output_dir / "component_ablations.csv").is_file()
    assert (output_dir / "unseen_conditions.csv").is_file()
    assert (output_dir / "recent_baselines.csv").is_file()
    assert (output_dir / "paired_candidate_minus_v3.csv").is_file()
    assert (output_dir / "MANUSCRIPT_ADDENDUM.tex").is_file()
    stored = json.loads(
        (output_dir / "OUTCOME.json").read_text(encoding="utf-8")
    )
    assert set(stored["source_bindings"]) == {
        "component_outcome",
        "unseen_outcome",
        "baseline_outcome",
        "builder_source",
        "qmf_tmc_bibliography",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_outputs(
            inputs["component"],
            inputs["unseen"],
            inputs["baseline"],
            output_dir,
        )
