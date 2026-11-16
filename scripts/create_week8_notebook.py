"""Create the Week 8 demonstration notebook; no database needed by default."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

KERNEL_NAME = 'idx-exchange-nlp'
KERNEL_DISPLAY_NAME = 'IDX Exchange NLP (.venv)'

BOOTSTRAP = '''from pathlib import Path
import sys
import importlib.util

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "scripts/query_parser.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook inside the IDX Exchange NLP project folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

missing = [name for name in ("nltk", "pandas", "ipywidgets", "rouge_score")
           if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(
        "This notebook is using the wrong or incomplete Python environment. "
        "Select kernel 'IDX Exchange NLP (.venv)' (or the project's .venv/bin/python), "
        "then run this cell again. Missing: " + ", ".join(missing) +
        ". Current Python: " + sys.executable
    )
'''

WIDGET_CELL = BOOTSTRAP + '''
import importlib
import scripts.import_week8_review as review_io
import scripts.week8_review_widget as review_ui
importlib.reload(review_io)
importlib.reload(review_ui)

if "review_widget" in globals():
    review_widget.view.close()
review_widget = review_ui.create_week8_review_widget(root=ROOT)
print("Python kernel:", sys.executable)
review_widget.display()
'''


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook(cells=[
        md('# Week 8: Listing Summarization and Answerability\n\nGenerate concise listing summaries, evaluate ROUGE-L, and explain unsupported or empty searches. BART/T5 fine-tuning is optional and is not implemented.'),
        code('from pathlib import Path\nimport sys, json\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "scripts/query_parser.py").exists())\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\nimport pandas as pd\nfrom IPython.display import display, Markdown\nfrom scripts.listing_summarizer import ListingSummarizer\nfrom scripts.answerability import AnswerabilityChecker, search_and_summarize\nfrom scripts.build_week8_artifacts import TEST_REFERENCE_PATH, run_pipeline\nsummarizer = ListingSummarizer()\nsample = pd.read_csv(ROOT / "data/processed/listing_sample.csv")\ncases = json.loads(TEST_REFERENCE_PATH.read_text())["cases"]\nprint(f"Local listings: {len(sample):,}; independent test records: {len(cases)}")'),
        md('## 1. Main deliverable: original-sentence extraction\n\nThe main `summary` selects up to three original sentences. It scores explicit room counts, price, location and supported features, then rewards new information and penalizes repeated content, boilerplate and excessive length. Feature-list scores are capped so kitchen details do not overwhelm bed/bath counts. Selected sentences retain their source order. Local NLTK Punkt rules work offline. Empty input generates no invented prose; descriptions with fewer than two useful sentences are reported as exceptions.'),
        code('record = cases[0]["record"]\nprint("SOURCE REMARKS:\\n", record["remarks"])\nprint("\\nMAIN EXTRACTIVE SUMMARY:\\n", summarizer.summarize_listing(record)["summary"])'),
        md('## 2. Trace selected sentences and keep MLS fields separate\n\n`summary` and `extractive_summary` contain the same original sentences. Listed price, city and numeric fields remain separate in `facts`. They are never inserted into the main prose. `selected_sentences` records source indices, scoring roles and scores. `missing_fields` identifies information not covered by the selected sentences; `not_in_remarks` identifies information not detected anywhere in the remarks. These are heuristic diagnostics, not proof of absence. Historical rent is not treated as listed price. Potential room-count conflicts remain metadata warnings. The old template appears only as `template_summary` for comparison.'),
        code('examples = [summarizer.summarize_listing(case["record"]) for case in cases[:5]]\ndisplay(pd.DataFrame([{k: row[k] for k in ["listing_id", "summary", "top_features", "missing_fields", "warnings"]} for row in examples]))\ndisplay(pd.DataFrame(examples[0]["feature_evidence"]))'),
        code('conflicts = [summarizer.summarize_listing(case["record"]) for case in cases if summarizer.summarize_listing(case["record"])["warnings"]]\nif conflicts:\n    display(pd.DataFrame(conflicts)[["listing_id", "summary", "facts", "warnings"]].head(3))\nelse:\n    print("No explicit early numeric room-count discrepancies in this test sample.")'),
        code('details = summarizer.summarize_listing(record)\ndisplay(pd.DataFrame(details["selected_sentences"]))\nprint("Separate MLS fields:", details["facts"])\nprint("Not detected in remarks:", details["not_in_remarks"])\nprint("Template comparison only:", details["template_summary"])'),
        md('## 3. ROUGE evaluation and dataset provenance\n\nThe development cohort contains 20 listings (seed 8). Because all 40 earlier development/test records had been inspected, the revised test cohort contains 20 different listings (seed 20261115), excluding all 40. References are independently assistant-selected sentences from original remarks, selected before inspecting revised test predictions. They do not insert MLS prices or room counts absent from remarks. They still require human confirmation. Do not tune on this test cohort.\n\nCompare the previous sentence scorer at budgets 2 and 3, the revised main selector and the old template against the **same** source-only references. Report macro-average ROUGE-1/2/L precision, recall and F1 using `rouge-score`, with Porter stemming. The target is **mean test ROUGE-L F1 > 0.4**. The old 0.200/0.452 scores used different references and are archived, not a direct before/after comparison. Source-selected references naturally reward sentence overlap; this does not establish factual correctness or replace human review.'),
        code('report = run_pipeline()\nscore_table = pd.DataFrame(report["scores"]).T\ndisplay(score_table[["rouge1_fmeasure", "rouge2_fmeasure", "rougeL_precision", "rougeL_recall", "rougeL_fmeasure"]].round(4))\nprint("Reference status:", report["evaluation_status"])\nprint("Human-verified references:", report["human_verified_references"], "/", report["reference_count"])\nprint("Main extractive summary target met numerically:", report["primary_rouge_target_met_numerically"])\nprint("All summaries match source sentences:", report["all_summaries_are_source_sentences"])\nprint("Fewer than two useful source sentences:", report["source_sentences_under_two"])\nprint("Ratings:", report["human_evaluation"])\nprint("Saved summaries:", report["summary_count"])'),
        md('## 4. Answerability before query execution\n\nReuse the Week 4 parser and validation. Accept property searches even when they omit words such as “home” but specify a recognized city. Reject unsupported questions, invalid cities/values and suspicious SQL fragments with specific messages. These rules describe the supported database-search scope; a supported query can still produce zero matches.'),
        code('checker = AnswerabilityChecker()\nqueries = ["Find 3+ bedroom homes in Irvine under $1M with a pool", "Irvine under $1M", "Find homes in mars", "Predict home prices next year", "Homes with school ratings in Irvine", "Find 99 bedroom homes in Irvine", "Homes; DROP TABLE rets_property", "Tell me a bedtime story"]\ndisplay(pd.DataFrame([{"query": q, **checker.assess_query(q)} for q in queries])[["query", "answerable", "reason", "message"]])'),
        md('## 5. Answerability after execution\n\nDistinguish zero matches, all-empty content, useful results, and an unavailable database. A connection failure is not evidence that no listings match.'),
        code('post_cases = {"no matches": pd.DataFrame(), "null content": pd.DataFrame([{"L_ListingID": "123", "remarks": None, "price": None}]), "useful results": sample.head(2), "database unavailable": None}\ndisplay(pd.DataFrame([{"example": name, **checker.assess_results(frame)} for name, frame in post_cases.items()]))'),
        md('## 6. End-to-end offline demonstration\n\nThis example searches the local Week 1 CSV using parsed filters, then runs the same post-check and summarizer as the database workflow. It does not contact MySQL. The production function defaults to the existing parameterized Week 4 MySQL executor.'),
        code('def local_search(query, *, limit, parser):\n    f = parser.parse(query)\n    frame = sample.copy()\n    if f.city:\n        frame = frame[frame.L_City.str.casefold() == f.city.casefold()]\n    for attr, col, comparison in [("price_min", "price", "ge"), ("price_max", "price", "le"), ("beds_min", "beds", "ge"), ("beds_max", "beds", "le"), ("baths_min", "baths", "ge"), ("baths_max", "baths", "le")]:\n        value = getattr(f, attr)\n        if value is not None:\n            frame = frame[getattr(frame[col], comparison)(value)]\n    for term in f.amenities_include + f.terms_include:\n        frame = frame[frame.remarks.fillna("").str.contains(term, case=False, regex=False)]\n    for term in f.amenities_exclude + f.terms_exclude:\n        frame = frame[~frame.remarks.fillna("").str.contains(term, case=False, regex=False)]\n    return frame.head(limit)\nresponse = search_and_summarize("Homes in Irvine", checker=checker, search_function=local_search, limit=3)\nprint(response["message"])\ndisplay(pd.DataFrame(response["results"])[["listing_id", "summary"]])'),
        md('## 7. Optional live MySQL demo\n\nStart Docker/MySQL and set `RUN_MYSQL_DEMO = True` to execute actual parameterized SQL. An unavailable service produces a specific explanation.'),
        code('RUN_MYSQL_DEMO = False\nif RUN_MYSQL_DEMO:\n    mysql_response = search_and_summarize("Homes in Irvine under $1M", limit=3)\n    print(mysql_response["message"])\n    display(pd.DataFrame(mysql_response["results"]))\nelse:\n    print("Live MySQL demo not run; offline demonstration completed above.")'),
        md('## 8. Teammate evaluation: 20 extractive summaries\n\nOpen `outputs/week8/week8_summary_review.xlsx`. Main Summary contains the new extractive output. Original remarks and a separate draft Reference summary are included. Rate factual accuracy, completeness against information actually present in remarks, and readability from 1 to 5, and enter a reviewer name. Do not penalize extraction for omitting a price absent from remarks; listed price remains separate metadata. Check negation, partial counts, material repair warnings and missing useful source content. Approve or correct the draft reference separately. Blank ratings are pending.\n\nImport the completed workbook with:\n```bash\npython scripts/import_week8_review.py /path/to/week8_summary_review.xlsx\n```\nThe importer verifies IDs and current Summary text, so ratings from the old template cannot count as ratings of the revised extractive summary. Old files remain in `outputs/week8/archive/`. Ratings and reference approvals remain separate.'),
        md('### Interactive review widget\n\nRun the next cell to review the same 20 test samples one at a time. Previous, Next and the Sample selector save your current draft; use Save review before closing. The original remarks, main extractive summary and separate MLS fields are visible. Enter your name, three 1–5 ratings and notes. Blank ratings remain pending. The reference-review panel and template comparison are optional to expand.\n\nDrafts are stored in `data/processed/week8_widget_reviews.json` and reload automatically. Click **Update evaluation** to import the saved reviews through the same validation as Excel and refresh the canonical report. Summary ratings do not automatically approve a reference. Excel remains an alternative; avoid editing both at the same time. No database is required.'),
        code(WIDGET_CELL),
        code('saved_report = json.loads((ROOT / "data/processed/week8_summary_evaluation.json").read_text())\nprint("Human evaluation:", saved_report["human_evaluation"])\nprint("Reference verification:", saved_report["human_verified_references"], "/", saved_report["reference_count"])\nprint("Re-run this cell after using Update evaluation to see the saved totals.")'),
        md('## 9. Findings and remaining work\n\n- The main output is genuine sentence extraction. MLS facts and template prose are separate comparison metadata.\n- Coverage-aware selection favors complementary room counts and feature sentences; it does not rewrite, add or silently correct original text. Missing facts, conflicting counts, malformed and very long source sentences remain limitations.\n- The same-reference baselines above measure the revised selector fairly. Changing reference protocol is not an algorithmic improvement by itself.\n- Test records exclude all previously inspected records, but references are still assistant-selected drafts. Numerical ROUGE is provisional until reviewed.\n- Twenty teammate ratings and reference confirmation remain required. No human scores have been fabricated.\n- ROUGE measures lexical overlap, not truth. Source descriptions may themselves contain incorrect or misleading claims.\n- Answerability uses explicit rules and inherits Week 4 parser limits. BART/T5 fine-tuning is optional and was not performed.\n\nROUGE implementation: [Google Research](https://github.com/google-research/google-research/tree/master/rouge).'),
    ], metadata={'kernelspec': {'display_name': KERNEL_DISPLAY_NAME, 'language': 'python', 'name': KERNEL_NAME},
                 'language_info': {'name': 'python'}})
    notebook.cells[1].source = BOOTSTRAP + '\n' + notebook.cells[1].source
    path = ROOT / 'notebooks/week8_listing_summarization.ipynb'
    nbf.write(notebook, path)
    print(path)


if __name__ == '__main__':
    main()
