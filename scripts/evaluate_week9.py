"""Evaluate frozen synthetic labels and scan real samples without inventing labels."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.compliance_checker import ComplianceChecker

OUTPUT = ROOT/'data/processed'
FIXTURES = ROOT/'tests/fixtures'


def evaluate_cases(cases, checker=None):
    if not cases:
        raise ValueError('Evaluation needs labeled cases')
    checker = checker or ComplianceChecker()
    counts = Counter()
    rows = []
    known, known_found, known_blocked = 0, 0, 0
    categories = {}
    for case in cases:
        if case['label'] not in {'known_violation', 'review', 'clean'}:
            raise ValueError('Unknown ground-truth label')
        result = checker.check_listing(case['text'])
        positive = case['label'] != 'clean'
        predicted = result['decision'] in {'block', 'review'}
        outcome = 'tp' if positive and predicted else 'fn' if positive else 'fp' if predicted else 'tn'
        counts[outcome] += 1
        expected = {'known_violation':'block', 'review':'review', 'clean':'pass'}[case['label']]
        if case['label'] == 'known_violation':
            known += 1
            known_found += predicted
            known_blocked += result['decision'] == 'block'
            group = categories.setdefault(case['category'], {'support': 0, 'detected': 0})
            group['support'] += 1
            group['detected'] += predicted
        rows.append({**case, 'predicted_decision':result['decision'], 'outcome':outcome,
                     'decision_correct':result['decision']==expected, 'findings':result['findings']})
    tp,fp,fn,tn = (counts[k] for k in ('tp','fp','fn','tn'))
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    for group in categories.values():
        group['recall'] = group['detected']/group['support']
    return {'count':len(rows), 'confusion':dict(zip(('tp','fp','fn','tn'),(tp,fp,fn,tn))),
            'screening_precision':precision,'screening_recall':recall,
            'screening_f1':2*precision*recall/(precision+recall) if precision+recall else 0.,
            'decision_accuracy':sum(r['decision_correct'] for r in rows)/len(rows),
            'known_violation_support':known, 'known_violations_detected':known_found,
            'known_violation_recall':known_found/known if known else None,
            'known_violation_error_recall':known_blocked/known if known else None,
            'per_category_known_violation_recall':categories,
            'targets_met':{'known_violation_recall_100':bool(known and known_found==known),
                           'precision_over_80':precision>.8},
            'definition':'Case-level: error or warning = flagged; info alone = not flagged. Known violations and review-required examples are positive; clean examples are negative.',
            'rows':rows}


def run_evaluation(*, scan_samples=True):
    checker = ComplianceChecker()
    reports, all_rows = {}, []
    for split in ('development','test'):
        fixture_path = FIXTURES/f'week9_compliance_{split}.json'
        payload = json.loads(fixture_path.read_text())
        result = evaluate_cases(payload['cases'], checker)
        all_rows.extend(result.pop('rows'))
        result.update(annotation_source=payload['annotation_source'],
                      human_verified=False, fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest())
        reports[split] = result
    report = {'rule_version':checker.version, 'rule_count':len(checker.rules),
              'rule_library_sha256':hashlib.sha256((ROOT/'data/fair_housing_patterns.json').read_bytes()).hexdigest(),
              'implementation_sha256':hashlib.sha256((ROOT/'scripts/compliance_checker.py').read_bytes()).hexdigest(),
              'evaluation_status':'synthetic_regression_only_not_human_verified',
              'limitations':'Synthetic examples and rules share an author. Text splits are disjoint but phrase families overlap. No deployment-wide recall, legal certification, or independent human benchmark is established.',
              'splits':reports}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'week9_compliance_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUTPUT/'week9_case_results.jsonl').open('w') as handle:
        for row in all_rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    if scan_samples:
        import pandas as pd
        source_path = OUTPUT/'listing_sample.csv'
        if source_path.exists():
            source = pd.read_csv(source_path)
            week8_path = OUTPUT/'week8_listing_summaries.jsonl'
            summaries = {str(r['listing_id']):r['summary'] for r in
                         (json.loads(line) for line in week8_path.read_text().splitlines())} if week8_path.exists() else {}
            decisions = {'remarks':Counter(),'summary':Counter()}
            flagged, scanned = 0, 0
            with (OUTPUT/'week9_sample_screening.jsonl').open('w') as handle:
                for record in source.to_dict('records'):
                    listing_id = str(record['L_ListingID'])
                    fields = {'remarks':checker.check_listing(record.get('remarks'))}
                    if listing_id in summaries:
                        fields['summary'] = checker.check_listing(summaries[listing_id])
                    for field, result in fields.items():
                        decisions[field][result['decision']] += 1
                    flagged += any(result['decision']!='pass' for result in fields.values())
                    scanned += 1
                    handle.write(json.dumps({'listing_id':listing_id,'checks':fields},ensure_ascii=False)+'\n')
            report['real_sample_scan'] = {'listing_count':scanned,'listings_flagged_for_correction_or_review':flagged,
                'decisions_by_field':{k:dict(v) for k,v in decisions.items()},
                'ground_truth_available':False,
                'interpretation':'Counts of screening flags, not counts of proven violations; no precision or recall is calculated on unlabeled real listings.'}
            (OUTPUT/'week9_compliance_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Week 9 Fair Housing language screening', '',
             'Synthetic regression results; all case labels need independent human validation.', '',
             '| Split | Cases | TP | FP | FN | TN | Precision | Known-violation recall | Decision accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split,result in reports.items():
        c=result['confusion']
        lines.append(f"| {split} | {result['count']} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | {result['screening_precision']:.2%} | {result['known_violation_recall']:.2%} | {result['decision_accuracy']:.2%} |")
    lines += ['', 'Only error/warning findings count as flagged; info is excluded. Metrics are computed per case, not by counting overlapping phrase matches.',
              '', report['limitations'], '', '## Real sample scan', '',
              json.dumps(report.get('real_sample_scan',{'status':'not run'}),indent=2), '',
              '## Errors to inspect', '']
    failures = [r for r in all_rows if r['outcome'] in {'fp','fn'} or not r['decision_correct']]
    if failures:
        lines += [f"- {r['case_id']}: {r['label']} -> {r['predicted_decision']} ({r['text']})" for r in failures]
    else:
        lines += ['No mismatches in this synthetic set. This is not evidence of perfect real-world detection.']
    lines += ['', 'Language screening never publishes a listing or grants an exemption.', '',
              'Sources: https://www.ecfr.gov/current/title-24/subtitle-B/chapter-I/part-100/subpart-B/section-100.75 and https://uscode.house.gov/us/usc/t42/s3607/']
    (OUTPUT/'week9_compliance_report.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__ == '__main__':
    report=run_evaluation()
    print(json.dumps({'rule_count':report['rule_count'],'splits':report['splits'],
                      'real_sample_scan':report.get('real_sample_scan')},indent=2))
