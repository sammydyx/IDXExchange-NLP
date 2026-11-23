"""Known examples, benign controls, context guards and submission integration."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compliance_checker import ComplianceChecker, CATEGORIES
from scripts.evaluate_week9 import evaluate_cases
from scripts.listing_submission import prepare_listing_submission

CASES=[case for split in ('development','test') for case in
       json.loads((ROOT/f'tests/fixtures/week9_compliance_{split}.json').read_text())['cases']]

@pytest.fixture(scope='module')
def checker():
    return ComplianceChecker()


@pytest.mark.parametrize('case',CASES,ids=lambda case:case['case_id'])
def test_labeled_language_and_severity(checker,case):
    result=checker.check_listing(case['text'])
    expected={'known_violation':'block','review':'review','clean':'pass'}[case['label']]
    assert result['decision']==expected, result
    if case['category']:
        assert any(f['category']==case['category'] and f['severity'] in {'error','warning'}
                   for f in result['findings'])
    for f in result['findings']:
        if f['span']:
            assert result['normalized_text'][slice(*f['span'])]==f['matched_text']


def test_fixtures_have_disjoint_texts_and_no_claimed_human_gold():
    a=json.loads((ROOT/'tests/fixtures/week9_compliance_development.json').read_text())
    b=json.loads((ROOT/'tests/fixtures/week9_compliance_test.json').read_text())
    assert set(c['text'] for c in a['cases']).isdisjoint(c['text'] for c in b['cases'])
    assert a['review_status']==b['review_status']=='needs_independent_human_review'
    assert {c['category'] for c in b['cases'] if c['label']=='known_violation'}==CATEGORIES


def test_metrics_do_not_hide_false_positives_or_false_negatives():
    cases=[{'case_id':str(i),'label':label,'text':text,'category':'race'} for i,(label,text) in
           enumerate([('known_violation','miss'),('known_violation','hit'),('clean','hit'),('clean','safe')])]
    fake=Mock()
    fake.check_listing.side_effect=lambda text:{'decision':'block' if text=='hit' else 'pass','findings':[]}
    r=evaluate_cases(cases,fake)
    assert r['confusion']=={'tp':1,'fp':1,'fn':1,'tn':1}
    assert r['screening_precision']==r['screening_recall']==r['known_violation_recall']==.5
    assert not any(r['targets_met'].values())


def test_info_is_not_counted_as_a_violation(checker):
    result=checker.check_listing('Wheelchair accessible entrance.')
    assert result['counts']['info']>0 and not result['violations']
    assert result['passed_screening'] and not result['legal_compliance_certified']


@pytest.mark.parametrize('text',['NO   CHILDREN','<p>No</p><p>children</p>','Ｎｏ ｃｈｉｌｄｒｅｎ','No\u00a0children'])
def test_format_variations_still_detect_exclusions(checker,text):
    assert checker.check_listing(text)['decision']=='block'


@pytest.mark.parametrize('text',[None,'',42,[],float('nan'),'   ','<p></p>'])
def test_missing_or_invalid_input_never_passes(checker,text):
    result=checker.check_listing(text)
    assert result['decision']=='block' and result['findings'][0]['category']=='input_validation'


def test_inclusive_slogan_cannot_cancel_a_current_exclusion(checker):
    assert checker.check_listing('All are welcome. No children.')['decision']=='block'
    assert checker.check_listing("The owner says 'no children'.")['decision']=='block'


def test_adult_only_policy_and_absence_have_different_meanings(checker):
    assert checker.check_listing('An adults-only policy applies to this home.')['decision']=='block'
    assert checker.check_listing('Adults-only restrictions are enforced here.')['decision']=='block'
    assert checker.check_listing('There is no adults-only policy.')['decision']=='pass'


def test_claimed_senior_exception_never_auto_passes(checker):
    result=checker.check_listing('55+ senior housing; adults only, no children.')
    assert result['decision']=='review'
    assert not result['passed_screening'] and not result['legal_compliance_certified']
    assert any('s3607' in source for finding in result['findings'] for source in finding['sources'])


def test_bad_regex_library_is_rejected(tmp_path):
    payload=json.loads((ROOT/'data/fair_housing_patterns.json').read_text())
    payload['rules'][0]['severity']='guaranteed_illegal'
    path=tmp_path/'bad.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        ComplianceChecker(path)


def test_error_and_warning_stop_submission_before_summarization(checker):
    summarizer=Mock()
    for text,status in [('No children.','blocked'),('Perfect for singles.','pending_review')]:
        result=prepare_listing_submission({'remarks':text},checker=checker,summarizer=summarizer)
        assert result['status']==status and not result['can_submit']
        assert result['published'] is False
    summarizer.summarize_listing.assert_not_called()


def test_generated_summary_is_also_checked(checker):
    summarizer=Mock()
    summarizer.summarize_listing.return_value={'summary':'White tenants only.'}
    result=prepare_listing_submission({'remarks':'A bright home with a pool.'},checker=checker,summarizer=summarizer)
    assert result['status']=='blocked' and result['checks']['remarks']['decision']=='pass'
    assert result['checks']['summary']['decision']=='block' and not result['published']


def test_provided_summary_cannot_hide_discriminatory_original(checker):
    result=prepare_listing_submission({'remarks':'No kids.','summary':'A bright home.'},checker=checker)
    assert result['status']=='blocked'


def test_good_submission_is_ready_but_never_published(checker):
    summarizer=Mock()
    summarizer.summarize_listing.return_value={'summary':'A 3-bedroom home with a pool.'}
    result=prepare_listing_submission({'L_ListingID':123,'remarks':'A 3-bedroom home with a pool.'},
                                     checker=checker,summarizer=summarizer)
    assert result['status']=='ready_for_submission' and result['can_submit']
    assert result['listing_id']=='123' and result['published'] is False


def test_week8_summarizer_integration(checker):
    from scripts.listing_summarizer import ListingSummarizer
    record={'L_ListingID':'integration','remarks':'This home has 3 bedrooms and 2 bathrooms. A private pool is included.'}
    result=prepare_listing_submission(record,checker=checker,summarizer=ListingSummarizer())
    assert result['status']=='ready_for_submission' and result['summary']


def test_testset_targets_are_reported_explicitly(checker):
    cases=[c for c in CASES if c['split']=='test']
    result=evaluate_cases(cases,checker)
    assert result['known_violation_recall']==1.
    assert result['screening_precision']>.8
