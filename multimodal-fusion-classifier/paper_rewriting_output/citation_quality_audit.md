# Citation Quality Audit

- Output directory: `paper_rewriting_output`
- Scene: conference
- Target citation count: 20
- Entries analyzed: 30
- Verified: 30 | Mismatched: 0 | Dead: 0
- Overall quality score: 84/100
- Status: PASS

> Each entry below includes a teaching note explaining *why* the citation quality matters.

## Per-Citation Analysis

| ID | DOI | Type | Resolves | Title Match | Year Match | Score | Status |
|---|---|---|---|---|---|---|---|
| C001 | 10.1117/12.2673838 | sota | yes | 100% | yes | 85 | verified |
| C002 | 10.1117/12.2673838 | sota | yes | 100% | yes | 85 | verified |
| C003 | 10.1117/12.2673838 | sota | yes | 100% | yes | 85 | verified |
| C004 | 10.1117/12.2673838 | sota | yes | 100% | yes | 85 | verified |
| C005 | 10.1109/IROS58592.2024.1080179 | benchmark | yes | 100% | yes | 95 | verified |
| C006 | 10.1109/IROS58592.2024.1080179 | benchmark | yes | 100% | yes | 95 | verified |
| C007 | 10.1109/IROS58592.2024.1080179 | benchmark | yes | 100% | yes | 95 | verified |
| C008 | 10.1109/IROS58592.2024.1080179 | benchmark | yes | 100% | yes | 95 | verified |
| C009 | 10.1109/IROS58592.2024.1080179 | benchmark | yes | 100% | yes | 95 | verified |
| C010 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C011 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C012 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C013 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C014 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C015 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C016 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C017 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C018 | https://openaccess.thecvf.com/ | benchmark | yes | 0% | no | 90 | verified |
| C019 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 75 | verified |
| C020 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 75 | verified |
| C021 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 75 | verified |
| C022 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 75 | verified |
| C023 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 75 | verified |
| C024 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 75 | verified |
| C025 | https://proceedings.mlr.press/ | sota | yes | 0% | no | 75 | verified |
| C026 | https://openaccess.thecvf.com/ | sota | yes | 0% | no | 75 | verified |
| C027 | https://openaccess.thecvf.com/ | sota | yes | 0% | no | 75 | verified |
| C028 | https://openaccess.thecvf.com/ | sota | yes | 0% | no | 75 | verified |
| C029 | https://openaccess.thecvf.com/ | sota | yes | 0% | no | 75 | verified |
| C030 | https://openaccess.thecvf.com/ | sota | yes | 0% | no | 75 | verified |

## Citation Diversity Gaps

**Missing foundational method or theory papers.** Only 0 of 30 entries (0%). Cite the 2-3 methods you build on. Be specific about what you inherit vs. change. Consider adding 1-3 foundational method or theory paper references.

**Missing survey, review, or meta-analysiss.** Only 0 of 30 entries (0%). Cite 1 recent survey if it helps position your work concisely. Consider adding 1-3 survey, review, or meta-analysis references.

**Missing domain-application or impact papers.** Only 0 of 30 entries (0%). Optional. Consider adding 1-3 domain-application or impact paper references.


## Scene-Specific Citation Strategy

For **conference** papers, your citation strategy should:

- **direct task or state-of-the-art paper**: Cite the 5-8 most recent competing methods. Conference reviewers check recency aggressively.
- **foundational method or theory paper**: Cite the 2-3 methods you build on. Be specific about what you inherit vs. change.
- **dataset, benchmark, or evaluation protocol paper**: Cite all datasets used. Standard benchmarks are expected.
- **survey, review, or meta-analysis**: Cite 1 recent survey if it helps position your work concisely.
- **domain-application or impact paper**: Optional.
- **limitation, robustness, reproducibility, or ethics paper**: Optional but helpful for discussion section.

## Citation Strategy Principles

- **Diversity over density.** A narrow citation pool makes your Introduction read as insular. Mix SOTA, foundational, benchmark, survey, and application papers.
- **Recency signals engagement.** Most citations should be from the last 3 years. Older citations are fine for foundational work, but they need a reason to be there.
- **Verifiability is non-negotiable.** Every DOI must resolve. A dead DOI in your final paper is a credibility failure that reviewers notice immediately.
- **Type matters by venue.** Journals expect deep SOTA coverage. Reports expect broad survey coverage. Competitions expect benchmark and leaderboard coverage. Match your strategy to your scene.
