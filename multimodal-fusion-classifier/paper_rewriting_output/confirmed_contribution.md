# Confirmed Contribution

## Core Contribution

| Field | Content |
|---|---|
| Main contribution statement | A reliability-gated visual--sensor-state classifier predicts a fallback latent and continuously controls the archived sidecar residual according to sample-wise quality evidence. |
| Contribution type | System method |
| Reviewer payoff | The method preserves an exact missing-input path while exposing when the auxiliary state is used, attenuated, or replaced. |

The principal contribution is a reliability-gated visual--sensor-state fusion model with a learned fallback. Visual and fixed class-text context predict a fallback latent state and reference grid. Availability, archived-grid statistics, predicted--observed discrepancy, latent agreement, and bounded variance evidence produce a sample-wise reliability coefficient. The effective state is

\[
s_{\mathrm{eff}}=\mu_{\mathrm{base}}+r(s-\mu_{\mathrm{base}}),
\]

which reaches the fallback exactly when the sidecar is absent and otherwise controls the archived residual continuously.

## Evidence Required And Available

| Evidence layer | Available evidence | Claim supported |
|---|---|---|
| Clean classification | Five modality configurations and six fusion families, seeds 42/43/44 | Visual-only is the strongest clean classifier; Proposed is not claimed to exceed RGB overall |
| Visual stress | Low-light, blur, occlusion, and low-confidence subsets | Conditional complementarity under the predefined low-light transform |
| Sample correspondence | Same low-light tensors/checkpoints with aligned, missing, and globally permuted sidecars | The trained model uses sample-sidecar correspondence |
| Compound subsystem | Three-seed matched retraining with and without fallback/reliability | Positive joint effect of the compound path, not an isolated gate contribution |
| Gate behavior | Three trained checkpoints and five fault realizations | Fault-sensitive quality ordering for available corruptions; missingness is deterministic |
| Class-memory probe | Text keys, shuffled keys, class names, and orthogonal class-ID keys | Class-aligned retrieval, not a unique natural-language semantic advantage |
| Dynamic versus constant gate | Three-seed matched retraining across clean and five fault conditions | No consistent average classification gain from sample adaptivity; \(r\) supports fault ordering, fault-conditioned calibration, and selective high-noise utility |

## Why This Contribution Is Needed

| Field | Content |
|---|---|
| Field problem | Auxiliary structured inputs can help under visually difficult conditions but may also be missing, noisy, biased, or spatially uninformative at inference. |
| Specific gap | Conventional fusion operators do not provide both an explicit learned replacement state and a continuous, auditable path between that state and an available record. |
| Concrete challenge | The model must retain useful sample correspondence without allowing an unreliable sidecar to become a mandatory or uniformly trusted input. |
| Why prior work leaves it unresolved | Existing quality weighting and missing-representation methods address adjacent pieces, while this setting requires fallback prediction, record comparison, and residual control within one classifier. |

## How This Paper Responds

| Field | Content |
|---|---|
| Design response | Visual and class-text context predict \(\mu_{\mathrm{base}}\); a reliability head estimates \(r\); the effective state is \(\mu_{\mathrm{base}}+r(s-\mu_{\mathrm{base}})\). |
| Evidence required | Clean and degraded classification, exact missing-input behavior, sample-sidecar correspondence, direct gate ordering, and matched subsystem controls are all required. |
| Evidence available | Three-seed comparisons, aligned/missing/permuted low-light counterfactuals, matched dynamic/constant training, five-fault analyses, and prediction-level audits are available. |
| Evidence missing | Independent hardware acquisition metadata, an external blind benchmark, and cross-device transfer measurements are not present in the current archive. |

## Data And Hardware Scope

The benchmark contains real RGB object crops paired by `sample_id` with normalized \(4\times4\) sidecars. Current archive provenance identifies 8,639 sidecars as image-conditioned vision-language estimates and 15 as deterministic fallbacks. Device identifiers, units, operating settings, site spacing, timing, and raw calibration references are absent. The present experiments therefore evaluate a structured sensor-state interface and do not establish independent hardware sensing.

## Claim Boundary

| Field | Content |
|---|---|
| Strong claims allowed | The compound fallback/reliability subsystem improves the matched ablation, and the learned score orders the predefined available corruptions while preserving deterministic absence handling. |
| Claims to soften or avoid | Do not claim overall clean superiority to RGB, language-specific semantic benefit, independent physical sensing, device-health probability, or isolated component gains from fixed-checkpoint bypasses. |
| Novelty risk | The individual building blocks are established; novelty rests on their explicit organization around a bounded residual path and structured sidecar interface. |
| Significance risk | Clean gains are not the value proposition, so significance must remain tied to conditional complementarity, missing-input fallback, and directly diagnosed fault behavior. |

- Visual-only remains the overall strongest clean-input classifier.
- LMF is described as the highest clean Macro-F1 method only among the evaluated three-modality fusion families.
- The class-memory experiment is developmental and separate from the principal 100-epoch model.
- Fixed-checkpoint bypasses measure path reliance, not independently retrained module contributions.
- The current internal split is not presented as an untouched external blind test.
- No unrecorded sensor hardware or acquisition parameter is inferred.
