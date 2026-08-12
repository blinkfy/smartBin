from __future__ import annotations

import pytest
import torch

import audit_recent_robust_baselines as baseline_audit
from build_recent_robust_baseline_protocol import build as build_protocol
from evaluate_recent_baseline import _validate_checkpoint
from evaluate_recent_baseline_conditions import _condition_metrics
from mmfc.recent_baseline_training import _config_fingerprint
from prepare_recent_robust_baselines import (
    METHODS,
    SEEDS,
    build_configs,
    normalized_matched_settings,
)
from prepare_standard_fusion_baselines import (
    METHODS as STANDARD_METHODS,
    SEEDS as STANDARD_SEEDS,
    build_configs as build_standard_configs,
)
from mmfc.recent_baselines import (
    BRANCH_NAMES,
    RecentFusionBaseline,
    dempster_shafer_combine,
    dirichlet_kl,
    tmc_evidential_loss,
)
from mmfc.recent_baseline_training import (
    CumulativeLossHistory,
    baseline_objective,
    qmf_ranking_loss,
)


def _features(batch_size: int = 4, classes: int = 5):
    generator = torch.Generator().manual_seed(17)
    visual = [
        torch.randn(batch_size, 512, generator=generator)
        for _ in range(3)
    ]
    sensor = torch.randn(batch_size, 512, generator=generator)
    prototypes = torch.randn(classes, 512, generator=generator)
    return visual, sensor, prototypes


@pytest.mark.parametrize("method", ("qmf", "tmc"))
def test_recent_baselines_produce_finite_three_branch_outputs(method: str):
    model = RecentFusionBaseline(
        5,
        method=method,
        dropout=0.0,
    )
    visual, sensor, prototypes = _features()

    logits, aux = model.forward_from_features(
        visual,
        sensor,
        prototypes,
        sensor_available=torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
        return_aux=True,
    )

    assert logits.shape == (4, 5)
    assert aux["branch_logits"].shape == (4, len(BRANCH_NAMES), 5)
    assert aux["branch_confidence"].shape == (4, len(BRANCH_NAMES))
    assert aux["modality_weights"].shape == (4, len(BRANCH_NAMES))
    assert aux["prototype_weights"].shape == (4, 5)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux["branch_confidence"]).all()
    torch.testing.assert_close(
        aux["modality_weights"].sum(dim=1),
        torch.ones(4),
    )


@pytest.mark.parametrize("method", ("concat_mlp", "gmu", "lmf"))
def test_standard_fusion_baselines_produce_finite_outputs(method: str):
    model = RecentFusionBaseline(5, method=method, dropout=0.0)
    visual, sensor, prototypes = _features()

    logits, aux = model.forward_from_features(
        visual,
        sensor,
        prototypes,
        return_aux=True,
    )
    labels = torch.tensor([0, 1, 2, 3])
    loss, components = baseline_objective(
        model,
        logits,
        aux,
        labels,
        epoch=1,
        qmf_history=None,
        sample_ids=None,
        qmf_ranking_weight=0.1,
        tmc_annealing_epochs=10,
    )

    assert logits.shape == (4, 5)
    assert aux["modality_weights"].shape == (4, len(BRANCH_NAMES))
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert components["joint_classification"] > 0


def test_lmf_focal_control_adds_explicit_focal_term():
    model = RecentFusionBaseline(5, method="lmf", dropout=0.0)
    visual, sensor, prototypes = _features()
    logits, aux = model.forward_from_features(
        visual,
        sensor,
        prototypes,
        return_aux=True,
    )
    labels = torch.tensor([0, 1, 2, 3])
    ce_only, _ = baseline_objective(
        model,
        logits,
        aux,
        labels,
        epoch=1,
        qmf_history=None,
        sample_ids=None,
        qmf_ranking_weight=0.1,
        tmc_annealing_epochs=10,
    )
    ce_focal, components = baseline_objective(
        model,
        logits,
        aux,
        labels,
        epoch=1,
        qmf_history=None,
        sample_ids=None,
        qmf_ranking_weight=0.1,
        tmc_annealing_epochs=10,
        classification_focal_weight=1.0,
    )

    assert components["focal"] > 0
    torch.testing.assert_close(ce_focal, ce_only + components["focal"])


def test_qmf_fusion_weights_are_detached_from_joint_path():
    model = RecentFusionBaseline(5, method="qmf", dropout=0.0)
    visual, sensor, prototypes = _features()

    logits, aux = model.forward_from_features(
        visual,
        sensor,
        prototypes,
        return_aux=True,
    )
    joint_gradient = torch.autograd.grad(
        logits.sum(),
        aux["branch_confidence"],
        allow_unused=True,
    )[0]

    assert joint_gradient is None


def test_tmc_ds_combination_is_symmetric_and_positive():
    alpha_a = torch.tensor([[2.0, 1.5, 1.2], [1.1, 3.0, 1.4]])
    alpha_b = torch.tensor([[1.3, 2.4, 1.1], [2.2, 1.2, 1.8]])

    ab = dempster_shafer_combine(alpha_a, alpha_b)
    ba = dempster_shafer_combine(alpha_b, alpha_a)

    torch.testing.assert_close(ab, ba)
    assert torch.all(ab >= 1.0)
    assert torch.isfinite(ab).all()


def test_dirichlet_kl_is_zero_for_uniform_opinion():
    value = dirichlet_kl(torch.ones(3, 5))

    torch.testing.assert_close(value, torch.zeros(3))


def test_tmc_evidential_loss_is_finite_and_anneals():
    labels = torch.tensor([0, 2, 1])
    alpha = torch.tensor(
        [
            [3.0, 1.2, 1.1],
            [1.1, 1.2, 3.0],
            [1.3, 2.8, 1.4],
        ]
    )

    early = tmc_evidential_loss(
        labels,
        alpha,
        epoch=0,
        annealing_epochs=10,
    )
    late = tmc_evidential_loss(
        labels,
        alpha,
        epoch=10,
        annealing_epochs=10,
    )

    assert early.shape == (3,)
    assert torch.isfinite(early).all()
    assert torch.isfinite(late).all()
    assert torch.all(late >= early)


def test_rejects_misnamed_independent_text_branch():
    model = RecentFusionBaseline(5, method="qmf", dropout=0.0)
    visual, sensor, prototypes = _features()

    with pytest.raises(ValueError, match="text_prototypes"):
        model.forward_from_features(
            visual,
            sensor,
            prototypes[:4],
        )


def test_qmf_history_uses_cumulative_sample_loss_and_round_trips():
    history = CumulativeLossHistory(["a", "b", "c"])
    losses = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [1.0, 2.0, 4.0],
            [2.0, 1.0, 3.0],
        ]
    )

    target, margin = history.update_and_pair_targets(
        ["a", "b", "c"],
        losses,
    )
    restored = CumulativeLossHistory(["a", "b", "c"])
    restored.load_state_dict(history.state_dict())

    assert target.shape == (3, 3)
    assert margin.shape == (3, 3)
    assert target[0, 0] == 1
    assert target[0, 2] == -1
    torch.testing.assert_close(restored.losses, history.losses)
    torch.testing.assert_close(restored.updates, history.updates)


def test_qmf_ranking_penalizes_confidence_order_opposite_to_loss():
    confidence = torch.tensor(
        [
            [0.9, 0.2, 0.2],
            [0.1, 0.8, 0.8],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    margin = torch.full_like(target, 0.5)

    loss = qmf_ranking_loss(confidence, target, margin)

    assert loss > 0
    loss.backward()
    assert confidence.grad is not None
    assert torch.isfinite(confidence.grad).all()


@pytest.mark.parametrize("method", ("qmf", "tmc"))
def test_baseline_objective_is_finite(method: str):
    model = RecentFusionBaseline(5, method=method, dropout=0.0)
    visual, sensor, prototypes = _features()
    logits, aux = model.forward_from_features(
        visual,
        sensor,
        prototypes,
        return_aux=True,
    )
    labels = torch.tensor([0, 1, 2, 3])
    history = (
        CumulativeLossHistory(["a", "b", "c", "d"])
        if method == "qmf"
        else None
    )

    loss, components = baseline_objective(
        model,
        logits,
        aux,
        labels,
        epoch=1,
        qmf_history=history,
        sample_ids=["a", "b", "c", "d"],
        qmf_ranking_weight=0.1,
        tmc_annealing_epochs=10,
    )

    assert torch.isfinite(loss)
    assert set(components) == {
        (
            "joint_classification"
            if method == "qmf"
            else "joint_evidential"
        ),
        (
            "branch_classification"
            if method == "qmf"
            else "branch_evidential"
        ),
        "ranking",
    }


def test_prepare_recent_baseline_configs_are_matched_and_isolated(tmp_path):
    paths = build_configs(output_dir=tmp_path)

    assert len(paths) == len(METHODS) * len(SEEDS)
    loaded = [
        __import__("json").loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    reference = normalized_matched_settings(loaded[0])
    assert {item["recent_baseline"]["method"] for item in loaded} == set(METHODS)
    assert {item["training"]["seed"] for item in loaded} == set(SEEDS)
    assert len(
        {item["training"]["checkpoint_dir"] for item in loaded}
    ) == len(loaded)
    assert all(normalized_matched_settings(item) == reference for item in loaded)
    assert all(
        item["recent_baseline"]["independent_text_observation_claimed"] is False
        for item in loaded
    )


def test_prepare_standard_fusion_configs_are_three_seed_and_matched(tmp_path):
    paths = build_standard_configs(output_dir=tmp_path, num_workers=4)
    loaded = [
        __import__("json").loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]

    assert len(paths) == len(STANDARD_METHODS) * len(STANDARD_SEEDS)
    assert {
        item["recent_baseline"]["method"] for item in loaded
    } == set(STANDARD_METHODS)
    assert {item["training"]["seed"] for item in loaded} == set(STANDARD_SEEDS)
    assert all(item["training"]["num_workers"] == 4 for item in loaded)
    assert all(item["training"]["batch_size"] == 16 for item in loaded)
    assert all(item["training"]["epochs"] == 100 for item in loaded)
    assert all(
        item["experiment"]["single_run_reference"] is True for item in loaded
    )


def test_recent_baseline_checkpoint_validation_binds_method_and_config():
    config = {
        "classes": [{"name": "a"}, {"name": "b"}],
        "recent_baseline": {"method": "qmf"},
        "training": {"resume": True},
    }
    checkpoint = {
        "baseline_method": "qmf",
        "config_fingerprint": _config_fingerprint(config),
        "class_names": ["a", "b"],
        "checkpoint_role": "best_evaluation",
    }

    assert _validate_checkpoint(checkpoint, config) == ("a", "b")
    checkpoint["baseline_method"] = "tmc"
    with pytest.raises(ValueError, match="method"):
        _validate_checkpoint(checkpoint, config)


def test_recent_condition_metrics_removes_internal_prediction_array():
    targets = __import__("numpy").asarray([0, 1, 0])
    probabilities = __import__("numpy").asarray(
        [[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]]
    )

    metrics, predictions = _condition_metrics(
        targets,
        probabilities,
        ("a", "b"),
    )

    assert "predictions" not in metrics
    assert predictions.tolist() == [0, 1, 1]


def test_recent_baseline_protocol_builder_freezes_sources(tmp_path):
    protocol = build_protocol(tmp_path / "PROTOCOL.json")

    assert protocol["valid"] is True
    assert protocol["state"] == "frozen_before_baseline_training"
    assert protocol["run_count"] == 6
    assert protocol["methods"] == ["qmf", "tmc"]
    assert protocol["training_seeds"] == [42, 43, 44]
    assert protocol["adaptation_contract"][
        "independent_text_observation_claimed"
    ] is False
    baseline_audit.validate_protocol(protocol)


def test_recent_baseline_audit_accepts_transparent_mixed_seed_amendment(
    tmp_path,
):
    protocol = build_protocol(tmp_path / "PROTOCOL.json")
    protocol.update(
        {
            "state": baseline_audit.AMENDED_PROTOCOL_STATE,
            "method_training_seeds": {
                method: list(values)
                for method, values in (
                    baseline_audit.AMENDED_METHOD_TRAINING_SEEDS.items()
                )
            },
            "run_count": 4,
            "original_run_count": 6,
            "completed_run_count": 4,
            "scope_reduction": {
                "single_run_reported_as_replication": False,
                "partial_run_used_for_results": False,
            },
            "original_protocol_binding": {},
        }
    )

    baseline_audit.validate_protocol(protocol)
    assert baseline_audit._mean_std([0.8]) == {
        "mean": 0.8,
        "sample_std": None,
    }


def test_paired_condition_comparison_aligns_samples(tmp_path, monkeypatch):
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    rows = [
        {
            "condition": "clean",
            "sample_id": "a",
            "source_image_id": "source-a",
            "target_id": 0,
            "probabilities": [0.8, 0.2],
        },
        {
            "condition": "clean",
            "sample_id": "b",
            "source_image_id": "source-b",
            "target_id": 1,
            "probabilities": [0.4, 0.6],
        },
    ]
    reference.write_text(
        "".join(__import__("json").dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    candidate_rows = [dict(row) for row in rows]
    candidate_rows[1]["probabilities"] = [0.2, 0.8]
    candidate.write_text(
        "".join(
            __import__("json").dumps(row) + "\n"
            for row in candidate_rows
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline_audit, "BOOTSTRAP_REPEATS", 10)

    result = baseline_audit._paired_comparison(
        reference,
        candidate,
        condition="clean",
        class_names=["a", "b"],
        seed=9,
    )

    assert result["sample_count"] == 2
    assert result["cluster_count"] == 2
    assert "accuracy" in result["differences"]
