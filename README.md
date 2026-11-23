# IDXExchange-NLP

A 12-week Natural Language Processing project for real estate listing intelligence.

## Week 0: Environment Setup

This project uses Python, Docker, MySQL, and pytest.

### Start the local environment

```bash
source .venv/bin/activate
docker compose up -d
pytest -q tests/test_setup.py
```

## Week 8: Listing Summarization and Answerability

The [Week 8 notebook](notebooks/week8_listing_summarization.ipynb) demonstrates
original-sentence extractive summaries, separate MLS metadata, and Week 4 query
validation with pre/post-search explanations. Its default workflow runs offline.
The primary `summary` copies remarks; the old template is comparison output only.
See [the Week 8 guide](docs/week8_listing_summarization.md) for source-only evaluation,
same-reference baselines, limitations, and the interactive review workflow.

The current extractive ROUGE-L F1 is 0.577; the three-sentence scoring baseline
is 0.634 on the same 20 test records. References are assistant-prepared drafts
awaiting human verification. Twenty self-reviews are saved, with mean ratings
of 5.00/5 for factual accuracy, 3.55/5 for completeness, and 5.00/5 for readability.
These self-ratings do not replace the requested teammate review.

```bash
python scripts/build_week8_artifacts.py
python -m pytest -q tests/test_week8.py
python -m pytest -q tests/test_week8_widget.py
```

For the notebook, activate the project environment and register its kernel:

```bash
python -m ipykernel install --user --name idx-exchange-nlp --display-name "IDX Exchange NLP (.venv)"
```

Select **IDX Exchange NLP (.venv)** in VS Code, then run the notebook. Section 8
contains Previous/Next navigation, ratings, notes, saved drafts, and an
**Update evaluation** button. Completed ratings and evaluation results are
versioned; local widget drafts and generated review seeds remain local.

## Week 9: Fair Housing Compliance Screening

The [Week 9 notebook](notebooks/week9_fair_housing_compliance.ipynb) demonstrates
25 source-linked rules across seven federal protected categories, error/warning/info
findings, context controls, and a simulated listing-submission workflow that also
checks Week 8 summaries. No listing is published and no legal compliance is certified.

The synthetic test split contains 103 examples: 41 stipulated known violations,
17 review-required cases and 45 clean controls. Its measured known-violation recall
and screening precision are both 100%. These fixtures were authored alongside
the rules and are not an independent human-verified legal benchmark.
The unlabeled 1,000-listing scan flagged 19 records for review; those are not
counts of proven violations.

```bash
source .venv/bin/activate
python scripts/evaluate_week9.py
python -m pytest -q tests/test_week9.py
```

Use **IDX Exchange NLP (.venv)** for the notebook. See
[the team guide](docs/week9_fair_housing.md) for rule sources, severity meanings,
metric definitions, examples and remaining validation work.


The [English manual-review workbook](outputs/week9_manual_review/week9_manual_review.xlsx)
contains all 19 flagged listings, full remarks, Week 8 summaries, and editable review
fields. Select a listing on **Listing Review**, then record the conclusion and evidence
on **Review Checklist**. Human review is still pending; no conclusions are prefilled.
The full real-sample screening evidence is saved in
[week9_sample_screening.jsonl](data/processed/week9_sample_screening.jsonl).
