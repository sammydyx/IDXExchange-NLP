# Week 9 Fair Housing language screening

Synthetic regression results; all case labels need independent human validation.

| Split | Cases | TP | FP | FN | TN | Precision | Known-violation recall | Decision accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| development | 68 | 38 | 0 | 0 | 30 | 100.00% | 100.00% | 100.00% |
| test | 103 | 58 | 0 | 0 | 45 | 100.00% | 100.00% | 100.00% |

Only error/warning findings count as flagged; info is excluded. Metrics are computed per case, not by counting overlapping phrase matches.

Synthetic examples and rules share an author. Text splits are disjoint but phrase families overlap. No deployment-wide recall, legal certification, or independent human benchmark is established.

## Real sample scan

{
  "listing_count": 1000,
  "listings_flagged_for_correction_or_review": 19,
  "decisions_by_field": {
    "remarks": {
      "pass": 981,
      "review": 19
    },
    "summary": {
      "pass": 987,
      "review": 13
    }
  },
  "ground_truth_available": false,
  "interpretation": "Counts of screening flags, not counts of proven violations; no precision or recall is calculated on unlabeled real listings."
}

## Errors to inspect

No mismatches in this synthetic set. This is not evidence of perfect real-world detection.

Language screening never publishes a listing or grants an exemption.

Sources: https://www.ecfr.gov/current/title-24/subtitle-B/chapter-I/part-100/subpart-B/section-100.75 and https://uscode.house.gov/us/usc/t42/s3607/
