"""Exercise review persistence and real widget callbacks using temporary drafts."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.week8_review_widget import SummaryReviewStore, ListingReviewWidget
from scripts.import_week8_review import apply_review_frame


@pytest.fixture
def items():
    return [{'listing_id': str(i), 'summary': f'Home {i} with a pool.',
             'record': {'remarks': f'Home {i} with a pool. A private garage is included.'},
             'facts': {'beds': 3, 'baths': 2, 'price': 500000, 'location': 'Irvine'},
             'reference_summary': f'Home {i} with a pool.', 'review_status': 'needs_human_review',
             'template_summary': 'An optional template comparison.'} for i in range(3)]


@pytest.fixture
def store(tmp_path, items):
    return SummaryReviewStore(items, tmp_path/'reviews.json')


def test_constructing_widget_does_not_write_or_complete_reviews(store):
    ui = ListingReviewWidget(store)
    assert not store.path.exists() and store.completed == 0
    assert all(field.value == 0 for field in ui.scores.values())
    assert ui.previous.disabled and not ui.next.disabled


def test_next_previous_callbacks_save_and_restore_notes_and_scores(store):
    ui = ListingReviewWidget(store)
    ui.reviewer.value = 'Test reviewer'
    ui.notes.value = 'A test-only note in a temporary file.'
    for field in ui.scores.values():
        field.value = 4
    ui.next.click()
    assert ui.index == 1 and store.completed == 1
    assert ui.notes.value == '' and all(f.value == 0 for f in ui.scores.values())
    ui.previous.click()
    assert ui.index == 0 and ui.notes.value.startswith('A test-only note')
    assert all(f.value == 4 for f in ui.scores.values())


def test_save_and_reopen_recovers_partial_draft_and_last_position(store, items):
    ui = ListingReviewWidget(store)
    ui.notes.value = 'Incomplete draft'
    ui.scores['readability'].value = 3
    ui.jump.value = 2
    assert ui.index == 2 and ui.next.disabled
    reopened = SummaryReviewStore(items, store.path)
    assert reopened.last_index == 2 and reopened.completed == 0
    assert reopened.get('0')['notes'] == 'Incomplete draft'
    assert reopened.get('0')['readability'] == 3


@pytest.mark.parametrize('value', [0, 6, 2.5, True, 'excellent'])
def test_invalid_scores_do_not_overwrite_drafts(store, value):
    store.save('0', {'notes': 'Keep this'})
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match='whole numbers'):
        store.save('0', {'factual_accuracy': value})
    assert store.path.read_bytes() == before
    assert store.get('0')['factual_accuracy'] is None


def test_all_scores_without_reviewer_are_still_incomplete(store):
    store.save('0', dict.fromkeys(('factual_accuracy', 'completeness', 'readability'), 5))
    assert store.completed == 0


def test_reference_approval_requires_name_and_does_not_navigate_on_error(store):
    ui = ListingReviewWidget(store)
    ui.approval.value = 'yes'
    ui.jump.value = 2
    assert ui.index == 0 and ui.jump.value == 0
    assert not store.path.exists()
    assert 'reviewer name' in ui.status.value


def test_stale_reviews_are_preserved_but_not_counted(items, tmp_path):
    path = tmp_path/'reviews.json'
    original = SummaryReviewStore(items, path)
    original.save('0', {'reviewer': 'Test reviewer', 'factual_accuracy': 5, 'completeness': 5, 'readability': 5})
    changed = copy.deepcopy(items)
    changed[0]['summary'] = 'A revised summary.'
    reopened = SummaryReviewStore(changed, path)
    assert reopened.completed == 0 and reopened.get('0')['factual_accuracy'] is None
    assert reopened.stale_reviews[0]['summary'] == items[0]['summary']
    reopened.save('0', {'notes': 'Review the changed version'})
    assert json.loads(path.read_text())['stale_reviews'][0]['reviewer'] == 'Test reviewer'


def test_concurrent_window_cannot_silently_overwrite_saved_work(store, items):
    second = SummaryReviewStore(items, store.path)
    store.save('0', {'notes': 'First window saved this'})
    with pytest.raises(ValueError, match='another window'):
        second.save('0', {'notes': 'Second window'})
    assert json.loads(store.path.read_text())['reviews'][0]['notes'] == 'First window saved this'


def test_source_and_notes_cannot_inject_html_into_the_cards(store):
    store.items[0]['record']['remarks'] = '<script>alert(1)</script>'
    ui = ListingReviewWidget(store)
    assert '<script>' not in ui.source.value and '&lt;script&gt;' in ui.source.value


def test_apply_button_uses_same_validated_review_import_as_excel(store, tmp_path):
    reference_path = tmp_path/'references.json'
    reference_path.write_text(json.dumps({'cases': store.items}))

    class CurrentSummarizer:
        def summarize_listing(self, record):
            return {'summary': record['remarks'].split('. ')[0]+'.'}

    def apply(frame):
        result = apply_review_frame(frame, reference_path=reference_path, output_dir=tmp_path,
                                    summarizer=CurrentSummarizer(), rebuild=False)
        return {'human_evaluation': {'completed': sum(r['status']=='complete' for r in result['ratings']), 'required': 3},
                'human_verified_references': sum(c.get('review_status')=='manual_verified' for c in result['cases'])}

    ui = ListingReviewWidget(store, on_apply=apply)
    ui.reviewer.value = 'Test reviewer'
    for field in ui.scores.values():
        field.value = 4
    ui.approval.value = 'yes'
    ui.apply_button.click()
    assert 'Evaluation updated: 1 / 3' in ui.status.value
    ratings = json.loads((tmp_path/'week8_human_ratings.json').read_text())
    assert ratings[0]['status'] == 'complete' and ratings[1]['status'] == 'incomplete'
    assert json.loads(reference_path.read_text())['cases'][0]['review_status'] == 'manual_verified'


def test_failed_evaluation_keeps_local_draft(store):
    def fail(_):
        raise ConnectionError('Test-only evaluation failure')
    ui = ListingReviewWidget(store, on_apply=fail)
    ui.notes.value = 'Keep my work'
    assert not ui.apply()
    assert store.get('0')['notes'] == 'Keep my work'
    assert 'still saved' in ui.status.value


def test_import_rejects_changed_summary_before_writing_any_results(store, tmp_path):
    path = tmp_path/'reference.json'
    path.write_text(json.dumps({'cases': store.items}))
    before = path.read_bytes()

    class ChangedSummarizer:
        def summarize_listing(self, _):
            return {'summary': 'Changed since the review was prepared.'}

    with pytest.raises(ValueError, match='Summary changed'):
        apply_review_frame(store.to_frame(), reference_path=path, output_dir=tmp_path,
                           summarizer=ChangedSummarizer(), rebuild=False)
    assert path.read_bytes() == before
    assert not (tmp_path/'week8_human_ratings.json').exists()
