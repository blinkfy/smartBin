# Results As Validation

The manuscript contributions are indexed here as C1 (fallback formulation), C2 (reliability-controlled residual fusion), and C3 (multiseed evaluation and mechanism diagnostics).

| Results unit | Contribution Claim Tested | Evidence | Allowed interpretation | Interpretation not allowed |
|---|---|---|---|---|
| Main clean benchmark | C3 | Visual-only 88.98/83.95; Proposed 88.26/82.26 Accuracy/Macro F1; three seeds | Vision is the clean discriminative core | Proposed is the overall best clean classifier |
| Fusion families | C3 | LMF-CE has 83.28 Macro F1; Proposed has better NLL/Brier/ECE than objective-matched LMF | LMF is strongest in clean Macro F1 among evaluated three-modality fusion families | LMF is the strongest classifier overall |
| Low-light stress | C3 | Proposed exceeds Visual-only by 1.54 Accuracy pp and 2.03 Macro-F1 pp | Conditional complementarity under one predefined software transform | General low-light or field superiority |
| Low-light sidecar counterfactual | C1, C3 | Aligned/missing/permuted Macro F1: 27.06/24.21/26.21; corresponding NLL: 3.024/3.234/3.087 | The trained model uses sample-sidecar correspondence | Independent physical-modality causality |
| Compound subsystem | C1, C2 | Full minus retrained ablation: +0.81 Accuracy pp, +0.51 Macro-F1 pp | Joint fallback/reliability benefit | Isolated gate contribution |
| Dynamic versus constant gate | C2, C3 | Clean dynamic-minus-constant: -0.20/-0.17 Accuracy/Macro-F1 pp; high-noise: +0.48/+0.98 pp; mixed: -0.19/-0.21 pp | No consistent average classification advantage; fallback drives primary robustness, while \(r\) provides fault ordering, fault-conditioned calibration, and selective high-noise utility | Uniform classification gain from sample adaptivity |
| Fixed-checkpoint paths | C1, C2 | FiLM and attention bypass effects; full-path bypass | Reliance of selected trained checkpoints on active paths | Retrained module contribution |
| Controlled faults | C1, C2, C3 | Three trained pairs times five fault seeds | Behavior under specified software interventions | Coverage of arbitrary hardware failures |
| Direct gate score | C2, C3 | Available-fault AUROC/AUPRC 0.704/0.710 across three models | Fault-sensitive ordering for defined available corruptions | Device-health probability or learned detection of missingness |
| Class-memory probe | C3 | Orthogonal class-ID keys match/exceed text-derived keys | Class-aligned retrieval | Unique natural-language semantic advantage |
| Variance diagnostic | C2, C3 | Spearman 0.892; scale ratio 44.3 | Ranking of reconstruction difficulty | Absolutely calibrated physical uncertainty |

The archived \(4\times4\) sidecars are image-conditioned estimates according to current provenance. No result in this table establishes synchronized independent hardware sensing.
