# Week 8: Original-sentence listing summarization

- Listings summarized: 1,000
- Main Summary output: extractive, remarks only
- All selected sentences verified against source: True
- Reference cases: 20; split: test
- Human-verified references: 0/20
- Teammate ratings: 20/20

## Same-source-reference comparison (macro F1)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| baseline_2_sentences | 0.5981 | 0.5366 | 0.5732 |
| baseline_3_sentences | 0.6599 | 0.5992 | 0.6343 |
| extractive | 0.6130 | 0.5457 | 0.5767 |
| template_comparison_only | 0.2621 | 0.0667 | 0.1804 |

The previous 0.200/0.452 results used references supplemented with MLS facts and are archived. They are not directly comparable with this source-only reference protocol.

Revised selector minus same-reference three-sentence baseline: -0.0577 ROUGE-L F1. The revised selector does not improve the three-sentence baseline on this test cohort. Do not claim a general improvement; use additional development data before future tuning.

The baselines and revised selector above use identical reference summaries and listing IDs. The revised method selects up to 3 informative source sentences; the 2- and 3-sentence baselines expose the sentence-budget difference.

Reference drafts were selected from original remarks by the assistant, independently of model predictions. They need human confirmation. High overlap with source-selected references is expected and is not proof of factual correctness or human evaluation completion.

The new test cohort uses seed 20261115 and excludes all 40 previously inspected development/test IDs. Model code is fingerprinted above. Future tuning should use development records only.

MLS beds/baths, price and city remain available as separate metadata. Missing facts are not inserted into the extracted prose. Some source descriptions do not supply two useful complete sentences; these cases are counted explicitly.

ROUGE implementation: https://github.com/google-research/google-research/tree/master/rouge
