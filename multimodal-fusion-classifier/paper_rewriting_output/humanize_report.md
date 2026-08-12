# Humanize Check Report

- Matrix path: `paper_rewriting_output\humanize_matrix.md`
- Humanize tier: light
- Matrix rows: 8
- Manuscript paragraphs: 68
- Coverage: 12%
- Sentence length stddev: 13.9
- Connector density: 0.62/1k chars
- Status: PASS

## Dimension Scores

### D1 sentence structure: WARNING [required]
- Metrics: sentence_count=900, length_stddev=13.88, sentence_length_cv=0.639, repeated_start_ratio=0.31, uniform_length_runs=152, short_sentence_ratio=0.49, long_sentence_ratio=0.0
- Affected units: S3-S5, S4-S6, S8-S10, S9-S11, S10-S12
- D1 consecutive sentences have near-identical lengths: ['S3-S5', 'S4-S6', 'S8-S10', 'S9-S11', 'S10-S12'].

### D2 paragraph similarity: WARNING [advisory]
- Metrics: paragraph_count=68, max_4gram_count=20, repeated_4gram_ratio=0.1409, paragraph_length_stddev=242.83, repeated_opening_ratio=0.18, min_paragraph_length=51, max_paragraph_length=1800, adjacent_paragraph_similarity_mean=0.095, adjacent_paragraph_similarity_max=0.272
- D2 repeated 4-gram ratio is elevated: 0.141 > 0.08 (max repeat count 20).

### D3 information density: PASS [advisory]
- Metrics: generic_phrase_density=0.0, information_anchor_density=32.68, generic_phrase_count=0, anchor_count=593, mechanism_term_count=88, ttr=0.4839, token_count=6902, unique_token_count=3340
- No dimension-specific risk found.

### D4 connector frequency: PASS [required]
- Metrics: connector_count=13, connector_density=0.62, max_paragraph_connector_density=8.93
- No dimension-specific risk found.

### D5 term-context matching: PASS [advisory]
- Metrics: frequent_terms_checked=12, contexts_checked=96, generic_context_ratio=0.0, mechanism_contexts=64, risky_terms=
- No dimension-specific risk found.

## Required Findings

- None

## Advisory Findings

- Coverage 12%: 8 rows for 68 paragraphs. Minimum 50%.
- D1 consecutive sentences have near-identical lengths: ['S3-S5', 'S4-S6', 'S8-S10', 'S9-S11', 'S10-S12'].
- D2 repeated 4-gram ratio is elevated: 0.141 > 0.08 (max repeat count 20).

## Threshold Profile

- adjacent_similarity_max_fail: 0.65
- adjacent_similarity_mean_warning: 0.45
- max_4gram_count_warning: 5
- max_connector_density: 8
- max_generic_density: 7
- max_paragraph_connector_density: 14
- max_repeated_start_ratio: 0.35
- max_term_generic_context_ratio: 0.45
- min_info_anchor_density: 2.5
- min_paragraph_length_stddev: 25
- min_sentence_length_stddev: 6
- repeated_4gram_ratio_fail: 0.15
- repeated_4gram_ratio_warning: 0.08
- sentence_length_cv_fail: 0.25
- sentence_length_cv_warning: 0.35
- ttr_fail_en: 0.25
- ttr_fail_zh: 0.35
- ttr_warning_en: 0.32
- ttr_warning_zh: 0.42
