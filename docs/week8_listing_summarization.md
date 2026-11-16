# Week 8: Listing Summarization and Answerability

## Primary output

`scripts/listing_summarizer.py` implements genuine sentence extraction:

- `split_sentences()` uses local NLTK Punkt rules, HTML decoding and whitespace normalization. No resource downloads are required.
- `extractive_details()` identifies room counts, listed-price mentions, location, features and material repair warnings in original sentences. It rewards complementary information and discounts repeated features, boilerplate and excessive length. Feature contributions are capped and prioritized. It restores source order and records sentence indices, roles and scores.
- `extractive_summary(remarks, entities, num_sentences=2)` exposes the guide-style text API. Entity values are scoring hints only; they cannot create prose.
- `summarize_listing(record)` makes the extractive text the main `summary`, selecting up to three sentences. `selected_sentences` provides provenance. `facts` separately retains MLS fields; `template_summary` is the old field-based prose for comparison only.

Prices, beds/baths and city absent from remarks cannot be inserted into a genuine extractive summary. They remain available as separate metadata for a search-result card. `missing_fields` describes information not covered by selected sentences; `not_in_remarks` describes information not detected anywhere in the source. Both are heuristic diagnostics. Potential room-count discrepancies remain metadata warnings, not added sentences. Historical rent is not assigned as the listed sale price.

Some descriptions have fewer than two useful sentences, contain fragments or use very long unpunctuated sentences. The selector reports these exceptions rather than inventing content to force a two-sentence output. Whole sentences can still be longer than ideal; this remains a limitation.

## Answerability and integration

`scripts/answerability.py` implements pre-query checks with the Week 4 parser and schema validation. It explains unsupported questions, invalid cities/values, unsupported logic and suspicious SQL fragments. Post-query checks distinguish no matches, empty content, useful results and an unavailable database.

`search_and_summarize()` joins these checks to Week 4's parameterized MySQL executor and the main extractive summarizer. The notebook uses an injected local executor by default, so Docker is optional. `scripts/search_listings.py` retains original remarks in the returned records.

## Run and inspect

From the project root:

```bash
source .venv/bin/activate
python scripts/build_week8_artifacts.py --development-only
python scripts/build_week8_artifacts.py
python -m pytest -q tests/test_week8.py
python -m pytest -q tests/test_week8_widget.py
python -m ipykernel install --user --name idx-exchange-nlp --display-name "IDX Exchange NLP (.venv)"
```

Open `notebooks/week8_listing_summarization.ipynb`, select **IDX Exchange NLP (.venv)** (or the project's `.venv/bin/python`) and Run All. The widget cell also initializes its own project path and can run independently. Its default workflow is offline. To run the optional MySQL example, start Docker/MySQL and set `RUN_MYSQL_DEMO = True`.

| File | Purpose |
|---|---|
| `data/processed/week8_listing_summaries.jsonl` | 1,000 main extractive summaries, selected source sentences and separate MLS metadata |
| `data/processed/week8_rouge_scores.csv` | Per-record ROUGE precision, recall and F1 for all four comparison variants |
| `data/processed/week8_summary_evaluation.json` | Canonical aggregate metrics, model fingerprint, provenance and actual review progress |
| `data/processed/week8_summary_report.md` | Readable evaluation report |
| `outputs/week8/week8_summary_review.xlsx` | 20 current extractive outputs, full remarks, draft references and blank human ratings |
| `outputs/week8/archive/` | Previous template-based workbook retained for comparison |

## Evaluation protocol

The 20 development records use seed 8. Both earlier development and test cohorts (40 records) had already been inspected. The revised test cohort therefore contains 20 different records selected with seed 20261115, excluding all those IDs. Inputs and source-only references are frozen in `tests/fixtures/week8_extractive_development.json` and `tests/fixtures/week8_extractive_test.json`.

`scripts/create_week8_extractive_references.py` uses separately selected source-sentence anchors; it does not read model predictions or use the scorer to generate references. Each draft reference is composed only of original sentences, with source indices and a source hash. Drafts were selected by the assistant before evaluating revised test predictions and remain `needs_human_review`. They are not independent human gold labels. Future tuning should use development data, not this test set.

Google Research `rouge-score==0.1.2` computes ROUGE-1/2/L precision, recall and F1 with Porter stemming. The report uses macro means across listings. The target is **mean test ROUGE-L F1 > 0.4** for the primary extractive output.

The report compares the previous sentence scorer at budgets 2 and 3, the revised selector (up to 3) and the old template on the exact same test records and references. Previous 0.200/0.452 results used references supplemented with MLS facts and are archived in `data/processed/week8_archive/`. They are not a direct before/after comparison. Source-selected references naturally reward shared sentences; a high ROUGE score does not establish factual accuracy or complete human evaluation.

Use the generated JSON/report for current numbers. BART/T5 fine-tuning is optional and has not been performed.

The saved run contains 20 completed self-reviews, with mean factual-accuracy, completeness and readability ratings of 5.00, 3.55 and 5.00 out of 5. These are subjective self-ratings; teammate review is still pending. Reference approvals are separate and currently 0/20, so the ROUGE result remains provisional. The coverage-based selector scores 0.577 ROUGE-L F1, compared with 0.634 for the three-sentence scoring baseline; it has not demonstrated an improvement over that baseline.

## Complete the 20-summary human evaluation

### Interactive notebook review

Section 8 of `notebooks/week8_listing_summarization.ipynb` includes an `ipywidgets` review interface. Run its widget cell using the project `.venv` kernel. In VS Code, the Python and Jupyter extensions must be enabled to render notebook widgets; a running MySQL service is not needed.

Each page displays full original remarks, the main extractive summary and separate MLS fields. Use Previous / Next or the Sample selector to move between the 20 records. Navigation saves the current draft. Enter your name, three 1–5 ratings and notes, then use Save review before closing. Unrated fields default to blank. A collapsed panel allows independent reference editing/approval; another shows the template summary for comparison, clearly separated from the summary being rated.

Drafts are saved by listing ID and summary SHA-256 in `data/processed/week8_widget_reviews.json`. They reload automatically, including partial work and the last position. Reviews of changed summaries are retained separately and do not count as current ratings. A second stale widget cannot silently overwrite newer saved work. The widget is implemented in `scripts/week8_review_widget.py`.

Click **Update evaluation** to import drafts into `week8_human_ratings.json`, record explicit reference approvals, and rebuild the evaluation report. This uses the same `apply_review_frame()` validation as the Excel importer. A completed draft is not included in the canonical report until this action succeeds. The widget does not fill in scores on anyone's behalf.

### Excel alternative

The existing workbook remains available. Use one review interface at a time to avoid conflicting edits.

On the Review sheet, enter a reviewer name and integer ratings from 1 to 5:

- **Factual accuracy:** Does the summary faithfully represent the original remarks, including negation, partial counts and repair warnings? Note conflicts with separate MLS fields rather than silently correcting them.
- **Completeness:** Does it include available bed/bath counts, price, location and important features? Do not require a fact absent from remarks. Note relevant source information omitted by the selector.
- **Readability:** Is it concise, coherent and readable? Record long or broken source sentences in Notes.

Use 1 for poor, 3 for acceptable and 5 for excellent. Review the separate draft Reference summary; correct it using information in remarks and explicitly approve it with `yes`. Ratings and reference approvals are separate actions. Original remarks are retained in both Review and Sources; MLS fields are separate comparison data. Blank ratings remain pending.

```bash
python scripts/import_week8_review.py /path/to/week8_summary_review.xlsx
```

The importer checks IDs, rating ranges and the current Summary text. It binds ratings to the summary's SHA-256 hash, so old template ratings cannot count toward revised extractive outputs. It records actual names and explicit approvals and regenerates metrics. Do not replace a completed workbook with a blank template.

## Limitations

Selection uses rules and the Week 1 taxonomy. It can miss paraphrases, long-range negation, ambiguous prices or location mentions. Original descriptions and MLS fields may themselves contain errors. Copying a sentence proves provenance, not truth. The 20-case test is small, and reference drafts still need human confirmation. Answerability is a conservative scope check that inherits Week 4's parser limits.

Metric implementation: [Google Research ROUGE](https://github.com/google-research/google-research/tree/master/rouge).
