# Structured Peer Review

- Manuscript: `aann2026_submission/paper.aann2026.review.pdf`
- Review date: 2026-08-11
- Review mode: three-role independent review synthesis
- Current recommendation: **Weak Accept**

## Methods and Reproducibility Review

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| MET-01 | CRITICAL | Principal Proposed and class-memory refinement were previously conflated | Resolved: Figure 1, Eq. (6)/(12)/(23), main tables, and conclusion consistently define the 100-epoch \(\mu_{\mathrm{base}}\) model; class-memory is a separate probe |
| MET-02 | CRITICAL | \(q\), \(\rho\), and \(\mu\) had conflicting meanings | Resolved: \(q_{\mathrm{sem}}\), \(y_{\mathrm{rel}}\), fusion \(\rho\), \(\lambda_{\mathrm{sem}}\), \(\mu_{\mathrm{base}}\), and \(\mu_{\mathrm{sem}}\) are disjoint |
| MET-03 | MAJOR | Physical array was insufficiently defined | Resolved at the recorded-interface level by Table III and Figure 3: one cell-calibrated metal/microwave response amplitude at each of 16 ordered tray sites, with exact classifier inclusions/exclusions |
| MET-04 | MAJOR | Gate behavior used one trained checkpoint | Resolved: Table XI/Figure 9 aggregate three training checkpoints and five fault seeds |
| MET-05 | MAJOR | Missingness structurally inflated gate discrimination | Resolved: abstract reports available-fault 0.704/0.710; Missing is deterministic and separate |
| MET-06 | MAJOR | LMF objective differs from Proposed | Resolved: the three-seed CE+focal control reaches 87.73/82.37% Accuracy/Macro F1 and is reported beside the 88.30/83.28% CE reference |
| MET-07 | MINOR | Stop-gradient had no direct control | Resolved: three-seed retraining shows -0.16/+0.20 pp Accuracy/Macro-F1 change without stop-gradient and worse NLL/Brier/ECE; the paper claims routing and calibration value, not universal classification gain |

## Contribution and Evidence Review

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| CON-01 | CRITICAL | Visual-only outperforms Proposed on clean input | Resolved with a bounded claim and direct utility comparison: low-light gains are +1.54/+2.03 pp Accuracy/Macro F1; three-training-seed sensor-fault rows show graceful fallback rather than universal superiority |
| CON-02 | MAJOR | Semantic results may only reflect class-memory addressing | Resolved: fixed orthogonal class-ID keys slightly outperform description keys; title and conclusion no longer claim language-specific semantic guidance |
| CON-03 | MAJOR | Dynamic reliability score lacked a direct comparator | Resolved with a fixed learned-bias gate; the dynamic score is claimed as a directly measured quality ordering, not an isolated average classification gain |
| CON-04 | MAJOR | Internal ablation effects are smaller than QMF differences | Resolved by separating questions: QMF is a family comparison, matched retraining estimates compound subsystem contribution, and path intervention measures learned reliance |
| CON-05 | MAJOR | Flat arrays may contain little spatial information | Resolved descriptively: Figure 4 reports 37.4% exact-flat records, the dynamic-range distribution, representative grids, and class-wise variation |

## Structure and Presentation Review

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| CLR-01 | MAJOR | Method figure was cluttered and ambiguous | Resolved with a four-panel editable vector architecture that uses the main \(\mu_{\mathrm{base}}\) path and passed layout checks |
| CLR-02 | MAJOR | Some formulas broke across lines/columns | Resolved: all 23 numbered equations are sequential and remain within column bounds in the Word-exported PDF |
| CLR-03 | MAJOR | A robustness figure occupied a page alone | Resolved: full-width figure placement was rebalanced; the 20-page PDF has no isolated figure page |
| CLR-04 | MINOR | Prose repeatedly foregrounded shortcomings | Resolved by concentrating scope statements and compressing Discussion/Conclusion; evidence is stated first and boundaries once |
| CLR-05 | MINOR | Figures and labels must remain editable | Resolved: all nine Word-embedded images are EMF vector assets with editable SVG sources; no raster manuscript figure remains |

## Editor Synthesis

The revised manuscript has one identifiable principal model, a consistent symbol system, an explicit physical sensor contract, and a multi-level evidence chain. Its strongest contribution is reliability-controlled use of complementary physical evidence under degraded or absent sensor input, not a clean-input win over RGB. The nonsemantic control narrows the former semantic claim to class-aligned memory retrieval, the LMF CE+focal control resolves the objective-fairness concern, and the stop-gradient control defines its optimization boundary. The controlled-fault comparison now uses three independently trained model pairs and five matched fault seeds.
