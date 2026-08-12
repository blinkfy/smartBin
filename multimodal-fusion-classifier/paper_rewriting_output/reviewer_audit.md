# Reviewer Audit

## Reviewer Value Map

| Criterion | Manuscript evidence | Remaining boundary |
|---|---|---|
| Novelty | Learned fallback plus evidence-dependent continuous residual control | A system formulation, not a universal fusion theorem |
| Significance | Low-light comparison, aligned/missing/permuted counterfactual, matched subsystem ablation, direct gate diagnostics | Clean RGB remains stronger overall |
| Technical soundness | One principal 100-epoch model, distinct symbols, uniform three-seed protocols, prediction-level provenance, and a nonselected constant-gate result | Three seeds bound observed training variability but do not replace external replication |
| Evidence sufficiency | Clean, visual-stress, sidecar-counterfactual, matched-retraining, fault, gate-score, variance, and class-memory controls form a linked evidence chain | The repeatedly analyzed split remains an internal benchmark rather than an external blind test |
| Reproducibility | Exact manifest join, stored tensor contract, configs, checkpoints, predictions, and scripts | No independent hardware acquisition metadata |
| Clarity | Editable vector figures, bounded claims, fixed-checkpoint interventions labeled as path reliance | Final DOCX/PDF must pass page-by-page visual QA |
| Venue fit | The paper combines an explicit neural architecture, controlled algorithmic ablations, calibrated prediction metrics, and reproducible mechanism diagnostics | AANN fit depends on presenting the method as an algorithmic reliability study, not as validated sensing hardware |

## Reviewer Objection Register

| Likely objection | Severity | Preemptive fix | Status |
|---|---|---|---|
| Proposed model and class-memory extension are conflated | Critical | Principal model uses \(\mu_{\mathrm{base}}\); class-memory is a separate developmental probe | Resolved |
| Symbols are overloaded | Critical | Semantic query, reliability target, fusion weights, semantic loss weight, and fallback endpoints use distinct notation | Resolved |
| Visual-only is better on clean input | Critical | Stated explicitly; utility claim is conditional and includes direct low-light sidecar counterfactuals | Resolved with bounded claim |
| Low-light advantage may not use the paired sidecar | Critical | Same-checkpoint aligned/missing/permuted comparison added for seeds 42/43/44 | Resolved for sample correspondence only |
| Sidecars are described as independent physical measurements | Critical | Provenance disclosed as 8,639 image-conditioned estimates plus 15 deterministic fallbacks | Resolved |
| Table VI and Table IX clean effects disagree | Major | Difference traced to evaluation batch/worker settings; both now use batch 16/workers 0 | Resolved |
| Dynamic gate lacks a matched constant-gate control | Major | Three-seed matched retraining preserves fallback, fusion, objective, order, and checkpoint rule except sample adaptivity of \(r\); the mixed result is reported without selection | Resolved with bounded claim |
| Missingness inflates learned gate AUROC | Major | Available-fault AUROC/AUPRC reported separately; missingness is handled by availability | Resolved |
| Class-memory proves language semantics | Major | Orthogonal class-ID control matches or exceeds text keys | Resolved with class-alignment claim only |
| LMF objective is unmatched | Major | CE and CE+Focal LMF controls are both reported | Resolved |
| FiLM bypass is presented as contribution | Major | All fixed-checkpoint interventions are called path reliance | Resolved |
| Hardware parameters are vague | Major | Only the stored \(4\times4\), finite, bounded interface is specified; absent hardware values are not invented | Resolved at archive-interface level |
| Internal test split has guided decisions | Major | Explicitly labeled a repeatedly analyzed internal benchmark | Resolved with limitation |

## Editorial Fit Map

| Editorial question | Current response | Residual risk |
|---|---|---|
| Venue fit | The submission centers on a reliability-gated neural fusion algorithm, matched controls, and direct reliability diagnostics aligned with an advanced-algorithms and neural-networks audience. | The application-specific sidecar archive should not be presented as a hardware contribution. |
| Editor-facing value | The paper offers an explicit fallback equation, multiseed mechanism evidence, reusable fault protocols, and a transparent negative constant-gate result. | Conditional rather than universal gains require the abstract and conclusion to remain carefully bounded. |
| Desk-reject risks | Model identity, notation, data provenance, baseline scope, and causal language have been audited across the manuscript and result artifacts. | External hardware validation and an untouched external benchmark remain future transfer studies. |

## Submission Gate

The experimental gate is complete. The rebuilt source sections pass numerical and package audits, and the 19-page final PDF has completed page-by-page visual QA. The remaining boundaries concern external hardware transfer and an untouched external benchmark, not unresolved internal inconsistencies.
