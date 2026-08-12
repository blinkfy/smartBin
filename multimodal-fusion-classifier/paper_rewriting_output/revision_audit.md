# Revision Audit

## Scope

- Comparison source: `qa/patent_source.txt`, extracted from
  `2026.4.3-1基于多模态识别的医疗废品识别方法.docx`.
- Revised source: `final_paper/main.tex`.
- Workflow note: this is a build-from-materials task rather than a sentence-level
  rewrite. The comparison therefore checks that the paper did not merely rephrase
  the patent while preserving its unsupported hardware and medical-waste claims.

## Automated Result

- Original paragraphs: 32
- Revised paragraphs: 35
- Near-identical revised paragraphs: 0 (0.0%)
- Mostly new revised paragraphs: 20 (57.1%)
- Likely deleted original paragraphs: 9 (28.1%)
- Addition-heavy: no
- Shallow-warning: no

The highest paragraph similarity was 0.491. It occurred in the CLIP description,
where reuse of the model name and ViT-B/16 backbone is technically necessary.
Other retained structural terms include YOLO11s-seg, the 4x4-to-16x16 grid path,
Adapter and FiLM. These are implementation lineage, not copied argumentation.

## Material Changes

| Source-material claim or structure | Final manuscript treatment | Verdict |
|---|---|---|
| Medical-waste recognition as the application scope | Reframed as a 19-class general object-level waste taxonomy and explicitly states that the labels are not a medical-waste or statutory waste system. | PASS |
| RGB-D input and synchronous physical 4x4 sensor acquisition | Replaced by the actually implemented RGB crop plus image-conditioned soft-sensor sidecar; hardware equivalence is explicitly denied. | PASS |
| Component recipe centered on YOLO, CLIP, Adapter, FiLM and attention | Rebuilt around a testable reliability question: prediction, fallback, cross-modal quality evidence and continuous residual suppression under degradation. | PASS |
| Detection and segmentation as the principal output | Object-level classification is the main task; the RGB segmentation head is reported only as a separately scoped system extension. | PASS |
| No formal data-leakage or uncertainty boundary | Adds source-image grouping, duplicate and near-duplicate audits, clustered bootstrap, seed stability, ECE and latent uncertainty diagnostics. | PASS |
| Strong implementation assertions without matched evidence | Replaces them with three matched training seeds, same-checkpoint intervention, registered degradation scenarios, negative results and explicit external-validity limits. | PASS |

## Verdict

**PASS.** The final manuscript is neither near-identical nor append-only. It
preserves only the implementation lineage supported by the codebase and rebuilds
the paper's problem, evidence chain and claim boundary around the implemented
soft-sensor reliability system.
