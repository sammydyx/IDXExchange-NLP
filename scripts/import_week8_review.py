"""Import actual teammate ratings and explicit reference approvals from Excel."""
from pathlib import Path
import argparse
import copy
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_week8_artifacts import TEST_REFERENCE_PATH, OUTPUT_DIR, run_pipeline, summary_hash
from scripts.listing_summarizer import ListingSummarizer


def parse_review_rows(frame, cases, *, expected_summaries=None):
    required = {"Listing ID", "Reviewer", "Factual accuracy (1-5)", "Completeness (1-5)",
                "Readability (1-5)", "Approve reference", "Notes", "Reference summary"}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing review columns: {sorted(required - set(frame.columns))}")
    frame = frame.loc[frame['Listing ID'].notna()].copy()
    frame['Listing ID'] = frame['Listing ID'].astype(str).str.replace(r'\.0$', '', regex=True)
    if frame['Listing ID'].duplicated().any() or set(frame['Listing ID']) != {case['listing_id'] for case in cases}:
        raise ValueError('Review listing IDs must exactly match the 20 test cases, without duplicates')
    updated = copy.deepcopy(cases)
    by_id = {case['listing_id']: case for case in updated}
    ratings = []
    for source in frame.to_dict('records'):
        clean = ListingSummarizer.text
        reviewer = clean(source['Reviewer'])
        entry = {'listing_id': source['Listing ID'], 'reviewer': reviewer, 'notes': clean(source['Notes'])}
        if expected_summaries is not None:
            expected = expected_summaries[source['Listing ID']]
            if clean(source.get('Summary')) != expected:
                raise ValueError(f"Summary changed for listing {source['Listing ID']}; review the current extractive output")
            entry['summary_sha256'] = summary_hash(expected)
        for name, column in [('factual_accuracy', 'Factual accuracy (1-5)'),
                             ('completeness', 'Completeness (1-5)'), ('readability', 'Readability (1-5)')]:
            value = clean(source[column])
            if value and (not value.replace('.', '', 1).isdigit() or not 1 <= float(value) <= 5 or not float(value).is_integer()):
                raise ValueError(f'{column} must be an integer from 1 to 5')
            entry[name] = int(float(value)) if value else None
        entry['status'] = 'complete' if reviewer and all(entry[k] is not None for k in ('factual_accuracy', 'completeness', 'readability')) else 'incomplete'
        approved = clean(source['Approve reference']).lower()
        if approved not in {'', 'yes', 'no'}:
            raise ValueError('Approve reference must be yes, no, or blank')
        if approved == 'yes':
            reference = clean(source['Reference summary'])
            if not reviewer or not reference:
                raise ValueError('Reference approval requires a reviewer name and non-empty reference summary')
            case = by_id[source['Listing ID']]
            case.update(reference_summary=reference, review_status='manual_verified',
                        reference_source='human_reviewed', reference_reviewer=reviewer)
            if reference != ' '.join(case.get('reference_sentences', [])):
                # Preserve a human correction as prose, not stale sentence indices.
                case.pop('reference_sentences', None)
                case.pop('reference_sentence_indices', None)
        ratings.append(entry)
    return updated, ratings


def apply_review_frame(frame, *, reference_path=TEST_REFERENCE_PATH, output_dir=OUTPUT_DIR,
                       summarizer=None, rebuild=True):
    """Shared validation/import route for Excel and notebook widget reviews."""
    reference_path, output_dir = Path(reference_path), Path(output_dir)
    payload = json.loads(reference_path.read_text())
    summarizer = summarizer or ListingSummarizer()
    expected = {case['listing_id']: summarizer.summarize_listing(case['record'])['summary'] for case in payload['cases']}
    cases, ratings = parse_review_rows(frame, payload['cases'], expected_summaries=expected)
    payload['cases'] = cases
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'week8_human_ratings.json').write_text(json.dumps(ratings, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if rebuild:
        return run_pipeline(reference_path=reference_path, output_dir=output_dir)
    return {'cases': cases, 'ratings': ratings}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('review_path', type=Path)
    args = cli.parse_args()
    frame = pd.read_excel(args.review_path, sheet_name='Review', header=5)
    report = apply_review_frame(frame)
    print(json.dumps({'human_evaluation': report['human_evaluation'], 'human_verified_references': report['human_verified_references']}, indent=2))


if __name__ == '__main__':
    main()
