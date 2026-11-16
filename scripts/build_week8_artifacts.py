"""Week 8: original-sentence summaries and source-only ROUGE evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.listing_summarizer import ListingSummarizer

REFERENCE_PATH = ROOT / 'tests/fixtures/week8_extractive_development.json'
TEST_REFERENCE_PATH = ROOT / 'tests/fixtures/week8_extractive_test.json'
OUTPUT_DIR = ROOT / 'data/processed'


def summary_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def evaluate(cases, summarizer=None):
    if not cases:
        raise ValueError('Evaluation requires at least one reference case')
    summarizer = summarizer or ListingSummarizer()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rows = []
    for case in cases:
        reference = case['reference_summary'].strip()
        if not reference:
            raise ValueError('Reference summaries must not be blank')
        remarks = case['record']['remarks']
        if case.get('reference_sentences') and case.get('review_status') != 'manual_verified':
            assert reference == ' '.join(case['reference_sentences'])
            assert all(sentence in summarizer.split_sentences(remarks) for sentence in case['reference_sentences'])
        output = summarizer.summarize_listing(case['record'])
        city = {'city': output['facts']['location']}
        variants = {'baseline_2_sentences': summarizer.baseline_extractive_summary(remarks, city, 2),
                    'baseline_3_sentences': summarizer.baseline_extractive_summary(remarks, city, 3),
                    'extractive': output['summary'],
                    'template_comparison_only': output['template_summary']}
        for method, prediction in variants.items():
            scores = scorer.score(reference, prediction)
            rows.append({'listing_id': case['listing_id'], 'method': method,
                         'reference_summary': reference, 'prediction': prediction,
                         **{f'{name}_{key}': float(getattr(score, key))
                            for name, score in scores.items() for key in ('precision', 'recall', 'fmeasure')}})
    frame = pd.DataFrame(rows)
    columns = [name for name in frame if name.endswith(('_precision', '_recall', '_fmeasure'))]
    means = {name: {column: float(group[column].mean()) for column in columns}
             for name, group in frame.groupby('method', sort=False)}
    reviewed = sum(case.get('review_status') == 'manual_verified' for case in cases)
    return {'reference_count': len(cases), 'human_verified_references': reviewed,
            'split': cases[0].get('split', 'development'),
            'reference_protocol': 'remarks-only; original-sentence drafts selected independently by assistant; human confirmation pending',
            'evaluation_status': 'human_verified' if reviewed == len(cases) else 'provisional_draft_references',
            'primary_method': 'extractive', 'target_metric': 'rougeL_fmeasure', 'target_threshold': .4,
            'rouge_implementation': 'google-research rouge-score 0.1.2; Porter stemming; macro mean per listing',
            'model_sha256': summary_hash((ROOT / 'scripts/listing_summarizer.py').read_text()),
            'scores': means, 'rows': rows}


def run_pipeline(*, input_path=None, reference_path=TEST_REFERENCE_PATH, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    summarizer = ListingSummarizer()
    source = pd.read_csv(input_path or OUTPUT_DIR / 'listing_sample.csv')
    records = source.to_dict('records')
    outputs = [summarizer.summarize_listing(record) for record in records]
    by_id = {row['listing_id']: row for row in outputs}
    with (output_dir / 'week8_listing_summaries.jsonl').open('w', encoding='utf-8') as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    cases = json.loads(reference_path.read_text())['cases']
    evaluation = evaluate(cases, summarizer)
    details = evaluation.pop('rows')
    pd.DataFrame(details).to_csv(output_dir / 'week8_rouge_scores.csv', index=False)
    review = [{**case, **by_id[case['listing_id']], 'reviewer': '',
               'summary_sha256': summary_hash(by_id[case['listing_id']]['summary']),
               'factual_accuracy': None, 'completeness': None, 'readability': None,
               'reference_approved': '', 'notes': ''} for case in cases]
    (output_dir / 'week8_human_review_seed.json').write_text(json.dumps(review, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    ratings_path = output_dir / 'week8_human_ratings.json'
    ratings = json.loads(ratings_path.read_text()) if ratings_path.exists() else []
    test_ids = {c['listing_id'] for c in cases}
    completed = sum(row.get('status') == 'complete' and row.get('listing_id') in test_ids
                    and row.get('summary_sha256') == summary_hash(by_id[row['listing_id']]['summary']) for row in ratings)
    primary_score = evaluation['scores']['extractive']['rougeL_fmeasure']
    source_copy_ok = all(row['summary'] == ' '.join(s['text'] for s in row['selected_sentences'])
                         and all(s['text'] in summarizer.split_sentences(record['remarks']) for s in row['selected_sentences'])
                         for row, record in zip(outputs, records))
    evaluation.update({'summary_count': len(outputs), 'all_summaries_are_source_sentences': source_copy_ok,
                       'primary_rouge_target_met_numerically': primary_score > .4,
                       'potential_room_count_conflicts': sum(bool(row['warnings']) for row in outputs),
                       'remarks_missing_information': {key: sum(key in row['not_in_remarks'] for row in outputs) for key in ('beds', 'baths', 'price', 'location')},
                       'source_sentences_under_two': sum(len(row['selected_sentences']) < 2 for row in outputs),
                       'human_evaluation': {'required': 20, 'completed': completed,
                                            'status': 'complete' if completed == 20 else 'awaiting_teammate_ratings'}})
    baseline_3 = evaluation['scores']['baseline_3_sentences']['rougeL_fmeasure']
    evaluation['extractive_minus_baseline_3_rougeL'] = primary_score - baseline_3
    (output_dir / 'week8_summary_evaluation.json').write_text(json.dumps(evaluation, indent=2) + '\n', encoding='utf-8')
    lines = ['# Week 8: Original-sentence listing summarization', '',
             f'- Listings summarized: {len(outputs):,}', '- Main Summary output: extractive, remarks only',
             f'- All selected sentences verified against source: {source_copy_ok}',
             f"- Reference cases: {len(cases)}; split: {evaluation['split']}",
             f"- Human-verified references: {evaluation['human_verified_references']}/{len(cases)}",
             f'- Teammate ratings: {completed}/20', '', '## Same-source-reference comparison (macro F1)', '',
             '| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |', '|---|---:|---:|---:|']
    for name, scores in evaluation['scores'].items():
        lines.append(f"| {name} | {scores['rouge1_fmeasure']:.4f} | {scores['rouge2_fmeasure']:.4f} | {scores['rougeL_fmeasure']:.4f} |")
    lines += ['', 'The previous 0.200/0.452 results used references supplemented with MLS facts and are archived. They are not directly comparable with this source-only reference protocol.',
              '', f"Revised selector minus same-reference three-sentence baseline: {primary_score - baseline_3:+.4f} ROUGE-L F1. " + ('The revised selector does not improve the three-sentence baseline on this test cohort. Do not claim a general improvement; use additional development data before future tuning.' if primary_score < baseline_3 else 'This comparison is limited to the small test cohort.'),
              '', 'The baselines and revised selector above use identical reference summaries and listing IDs. The revised method selects up to 3 informative source sentences; the 2- and 3-sentence baselines expose the sentence-budget difference.',
              '', 'Reference drafts were selected from original remarks by the assistant, independently of model predictions. They need human confirmation. High overlap with source-selected references is expected and is not proof of factual correctness or human evaluation completion.',
              '', 'The new test cohort uses seed 20261115 and excludes all 40 previously inspected development/test IDs. Model code is fingerprinted above. Future tuning should use development records only.',
              '', 'MLS beds/baths, price and city remain available as separate metadata. Missing facts are not inserted into the extracted prose. Some source descriptions do not supply two useful complete sentences; these cases are counted explicitly.',
              '', 'ROUGE implementation: https://github.com/google-research/google-research/tree/master/rouge']
    (output_dir / 'week8_summary_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return evaluation


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--development-only', action='store_true')
    args = cli.parse_args()
    if args.development_only:
        report = evaluate(json.loads(REFERENCE_PATH.read_text())['cases'])
        report.pop('rows')
    else:
        report = run_pipeline()
    print(json.dumps(report, indent=2))
