"""Week 9 integration: prepare a submission; never publish externally."""
from collections.abc import Mapping
from scripts.compliance_checker import ComplianceChecker


def prepare_listing_submission(record, *, checker=None, summarizer=None, generate_summary=True):
    if not isinstance(record, Mapping):
        raise TypeError('record must be a mapping')
    checker = checker or ComplianceChecker()
    remarks = record.get('remarks', record.get('L_Remarks'))
    reports = {'remarks': checker.check_listing(remarks)}
    supplied = record.get('summary')
    if supplied:
        reports['summary'] = checker.check_listing(supplied)
    raw_decision = reports['remarks']['decision']
    if raw_decision == 'pass' and generate_summary and not supplied:
        if summarizer is None:
            from scripts.listing_summarizer import ListingSummarizer
            summarizer = ListingSummarizer()
        supplied = summarizer.summarize_listing(record)['summary']
        reports['summary'] = checker.check_listing(supplied)
    decisions = [result['decision'] for result in reports.values()]
    decision = 'block' if 'block' in decisions else 'review' if 'review' in decisions else 'pass'
    return {
        'listing_id': str(record.get('listing_id', record.get('L_ListingID', ''))),
        'status': {'block': 'blocked', 'review': 'pending_review', 'pass': 'ready_for_submission'}[decision],
        'can_submit': decision == 'pass', 'published': False,
        'summary': supplied, 'checks': reports,
        'message': {
            'block': 'Correct the flagged language before resubmitting.',
            'review': 'Qualified human review is required before submission.',
            'pass': 'Language screening passed; continue the remaining submission checks.',
        }[decision],
    }
