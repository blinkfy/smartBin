from __future__ import annotations

import torch

from mmfc.joint_seg_patent import PatentAlignedReliableFeatureFusion


def test_patent_fusion_exposes_required_stack_and_preserves_initial_maps() -> None:
    fusion = PatentAlignedReliableFeatureFusion(
        (8, 16, 32),
        torch.randn(3, 512),
        semantic_dim=8,
        sensor_hidden_dim=16,
        spatial_hidden_dim=16,
        patent_attention_dim=128,
        patent_attention_heads=8,
    )
    fusion.eval()
    features = [
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]

    with torch.inference_mode():
        output = fusion(
            features,
            torch.rand(2, 1, 4, 4),
            torch.ones(2, 1),
        )

    assert all(
        torch.equal(candidate, source)
        for candidate, source in zip(output, features)
    )
    assert fusion.patent_film_gamma.shape == (3, 512)
    assert torch.equal(
        fusion.patent_film_gamma,
        torch.ones_like(fusion.patent_film_gamma),
    )
    assert torch.equal(
        fusion.patent_film_beta,
        torch.full_like(fusion.patent_film_beta, 0.1),
    )
    auxiliary = fusion.last_aux
    assert auxiliary["patent_modality_weights"].shape == (2, 3)
    assert auxiliary["patent_scale_weights"].shape == (2, 3)
    assert auxiliary["patent_interaction_weights"].shape == (2, 8, 3, 3)
    assert auxiliary["patent_final_fusion_feature"].shape == (2, 512)
    assert auxiliary["patent_sensor_grid_16x16"].shape == (2, 1, 16, 16)
    assert torch.allclose(
        auxiliary["patent_scale_weights"].sum(dim=1),
        torch.ones(2),
    )
