"""Persistent, one-listing-at-a-time Week 8 human review inside a notebook."""
from __future__ import annotations

import copy
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from scripts.build_week8_artifacts import ROOT, TEST_REFERENCE_PATH, summary_hash

SCORE_FIELDS = ('factual_accuracy', 'completeness', 'readability')


class SummaryReviewStore:
    """Save drafts by listing ID and summary hash, without inventing ratings."""

    def __init__(self, items, path, *, existing_reviews=()):
        self.items = copy.deepcopy(items)
        if not self.items or len({row['listing_id'] for row in items}) != len(items):
            raise ValueError('Review samples must be non-empty and have unique listing IDs')
        self.path = Path(path)
        self.by_id = {row['listing_id']: row for row in self.items}
        self.reviews = {row['listing_id']: self._blank(row) for row in self.items}
        self.stale_reviews = []
        self.last_index = 0
        self._disk_hash = None
        payload = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            self._disk_hash = hashlib.sha256(raw).hexdigest()
            payload = json.loads(raw)
            if payload.get('version') != 1:
                raise ValueError('Unsupported review draft version; keep the original file')
            self.stale_reviews = payload.get('stale_reviews', [])
            self.last_index = min(max(int(payload.get('last_index', 0)), 0), len(items) - 1)
        drafts = {row['listing_id']: row for row in payload.get('reviews', [])}
        saved = {row['listing_id']: row for row in existing_reviews}
        saved.update(drafts)
        for listing_id, row in saved.items():
            item = self.by_id.get(listing_id)
            if item is None or row.get('summary_sha256') != summary_hash(item['summary']):
                self.stale_reviews.append(copy.deepcopy(row))
                continue
            self.reviews[listing_id] = self._validated(item, {**self.reviews[listing_id], **row})

    @staticmethod
    def _blank(item):
        return {'listing_id': item['listing_id'], 'summary': item['summary'],
                'summary_sha256': summary_hash(item['summary']), 'reviewer': '',
                **dict.fromkeys(SCORE_FIELDS), 'notes': '', 'reference_approved': '',
                'reference_summary': item.get('reference_summary', ''), 'status': 'incomplete'}

    @staticmethod
    def _validated(item, values):
        row = {key: values.get(key) for key in SummaryReviewStore._blank(item)}
        row.update(listing_id=item['listing_id'], summary=item['summary'],
                   summary_sha256=summary_hash(item['summary']))
        for key in ('reviewer', 'notes', 'reference_summary', 'reference_approved'):
            row[key] = str(row.get(key) or '').strip()
        for key in SCORE_FIELDS:
            value = row[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not float(value).is_integer() or not 1 <= value <= 5):
                raise ValueError('Ratings must be whole numbers from 1 to 5, or left blank')
            row[key] = int(value) if value is not None else None
        if row['reference_approved'] not in ('', 'yes', 'no'):
            raise ValueError('Reference decision must be pending, approved, or needs revision')
        if row['reference_approved'] == 'yes' and not (row['reviewer'] and row['reference_summary']):
            raise ValueError('Enter your reviewer name and a reference summary before approving it')
        row['status'] = 'complete' if row['reviewer'] and all(row[k] is not None for k in SCORE_FIELDS) else 'incomplete'
        return row

    @property
    def completed(self):
        return sum(row['status'] == 'complete' for row in self.reviews.values())

    def get(self, listing_id):
        return copy.deepcopy(self.reviews[listing_id])

    def save(self, listing_id, values, *, last_index=0):
        if not 0 <= last_index < len(self.items):
            raise ValueError('Review index is outside the sample list')
        row = self._validated(self.by_id[listing_id], {**self.reviews[listing_id], **values})
        if self.path.exists():
            current_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if current_hash != self._disk_hash:
                raise ValueError('Reviews changed in another window. Reopen this widget before saving.')
        elif self._disk_hash is not None:
            raise ValueError('The review file was moved. Reopen this widget before saving.')
        updated = {**self.reviews, listing_id: row}
        payload = {'version': 1, 'last_index': last_index, 'reviews': list(updated.values()),
                   'stale_reviews': self.stale_reviews}
        raw = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(dir=self.path.parent, prefix='.week8-review-', delete=False)
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(raw)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.reviews, self.last_index = updated, last_index
        self._disk_hash = hashlib.sha256(raw).hexdigest()

    def to_frame(self):
        return pd.DataFrame([{
            'Listing ID': row['listing_id'], 'Summary': row['summary'], 'Reviewer': row['reviewer'],
            'Factual accuracy (1-5)': row['factual_accuracy'], 'Completeness (1-5)': row['completeness'],
            'Readability (1-5)': row['readability'], 'Approve reference': row['reference_approved'],
            'Notes': row['notes'], 'Reference summary': row['reference_summary'],
        } for row in self.reviews.values()])


def _card_markup(title, text):
    return (
        '<section style="border:1px solid #cbd5e1;border-radius:8px;padding:14px;'
        'background:#fff;color:#1f2937;height:100%;box-sizing:border-box">'
        f'<strong>{html.escape(title)}</strong><div style="white-space:pre-wrap;'
        'overflow:auto;max-height:380px;line-height:1.6;margin-top:10px">'
        f'{html.escape(str(text))}</div></section>')


def _card(title, text):
    return widgets.HTML(value=_card_markup(title, text), layout=widgets.Layout(width='100%'))


class ListingReviewWidget:
    """Navigation saves the current draft; applying reviews refreshes metrics."""

    def __init__(self, store, *, on_apply=None):
        self.store, self.on_apply = store, on_apply
        self.index = store.last_index
        self._loading = True
        self.progress = widgets.HTML()
        self.status = widgets.HTML()
        self.previous = widgets.Button(description='Previous', icon='arrow-left')
        self.next = widgets.Button(description='Next', icon='arrow-right')
        self.save_button = widgets.Button(description='Save review', icon='save', button_style='primary')
        self.apply_button = widgets.Button(description='Update evaluation', icon='check', disabled=on_apply is None,
                                           layout=widgets.Layout(width='170px'))
        self.jump = widgets.Dropdown(options=[(f"{i+1}. {row['listing_id']} — {row.get('facts', {}).get('location') or 'Unknown city'}", i)
                                              for i, row in enumerate(store.items)], description='Sample:',
                                     layout=widgets.Layout(width='420px'))
        self.metadata = widgets.HTML()
        self.source, self.summary = _card('Original remarks', ''), _card('Summary to rate — extractive', '')
        self.reviewer = widgets.Text(description='Reviewer:', placeholder='Your name', continuous_update=True,
                                     style={'description_width': '90px'}, layout=widgets.Layout(width='360px'))
        # A real dropdown value keeps the pending label visible in the browser.
        # Convert this UI-only sentinel to None before saving any rating.
        options = [('Not rated', 0)] + [(str(i), i) for i in range(1, 6)]
        self.scores = {key: widgets.Dropdown(options=options, description=label,
                      style={'description_width': '110px'}, layout=widgets.Layout(width='240px'))
                       for key, label in zip(SCORE_FIELDS, ('Accuracy:', 'Completeness:', 'Readability:'))}
        self.notes = widgets.Textarea(description='Notes:', placeholder='Errors, missing information, or comments…',
                                     rows=4, continuous_update=True, style={'description_width': '90px'},
                                     layout=widgets.Layout(width='100%'))
        self.reference = widgets.Textarea(description='Reference:', rows=5, continuous_update=True,
                                          style={'description_width': '90px'}, layout=widgets.Layout(width='100%'))
        self.approval = widgets.Dropdown(options=[('Pending', ''), ('Approve reference', 'yes'), ('Needs revision', 'no')],
                                         description='Decision:', style={'description_width': '90px'},
                                         layout=widgets.Layout(width='340px'))
        reference_panel = widgets.VBox([widgets.HTML('Reference approval is separate from summary ratings. '
            'Check or edit this draft using facts supported by the original remarks.'), self.reference, self.approval])
        self.reference_panel = widgets.Accordion(children=[reference_panel], selected_index=None)
        self.reference_panel.set_title(0, 'Optional: inspect or approve the reference summary')
        self.comparison = widgets.Accordion(children=[_card('Template + MLS fields — comparison only', '')], selected_index=None)
        self.comparison.set_title(0, 'Other method: template summary (not the summary being rated)')
        self.view = widgets.VBox([
            widgets.HTML('<h3 style="margin:4px 0">Week 8 · Summary review</h3>'
                         '<p>Previous, Next and Sample save your current review. Use Save review before closing. '
                         'Ratings: 1 = poor, 3 = acceptable, 5 = excellent; blank = pending.</p>'),
            self.progress,
            widgets.HBox([self.previous, self.next, self.jump], layout=widgets.Layout(flex_flow='row wrap', grid_gap='8px')),
            self.metadata,
            widgets.Box([self.source, self.summary], layout=widgets.Layout(display='grid',
                         grid_template_columns='repeat(auto-fit, minmax(min(100%, 360px), 1fr))', grid_gap='12px')),
            self.reviewer,
            widgets.HBox(list(self.scores.values()), layout=widgets.Layout(flex_flow='row wrap')),
            self.notes, self.reference_panel, self.comparison,
            widgets.HBox([self.save_button, self.apply_button], layout=widgets.Layout(grid_gap='8px')),
            self.status,
        ], layout=widgets.Layout(width='100%', max_width='1180px', grid_gap='10px'))
        self.previous.on_click(lambda _: self.go(self.index-1))
        self.next.on_click(lambda _: self.go(self.index+1))
        self.jump.observe(self._jumped, names='value')
        self.save_button.on_click(lambda _: self.save())
        self.apply_button.on_click(lambda _: self.apply())
        for field in [self.reviewer, *self.scores.values(), self.notes, self.reference, self.approval]:
            field.observe(self._changed, names='value')
        self._render()
        if store.stale_reviews:
            self._message(f'{len(store.stale_reviews)} older reviews were preserved separately because their sample or summary changed.', error=True)

    def display(self):
        display(self.view)

    def _message(self, text, *, error=False):
        color = '#b91c1c' if error else '#4b5563'
        self.status.value = f'<span style="color:{color}">{html.escape(text)}</span>'

    def _changed(self, change):
        if not self._loading:
            self._message('Unsaved changes. Save review or navigate to keep them.')

    def _refresh_progress(self):
        self.progress.value = f'<b>Sample {self.index+1} / {len(self.store.items)}</b> &nbsp; · &nbsp; Completed ratings: <b>{self.store.completed} / {len(self.store.items)}</b>'

    def _render(self):
        self._loading = True
        item = self.store.items[self.index]
        row = self.store.get(item['listing_id'])
        reviewer_hint = self.reviewer.value
        self.jump.value = self.index
        self.source.value = _card_markup('Original remarks', item['record']['remarks'])
        self.summary.value = _card_markup('Summary to rate — extractive', item['summary'])
        self.comparison.children[0].value = _card_markup('Template + MLS fields — comparison only', item.get('template_summary', ''))
        f = item.get('facts', {})
        price = f"${f['price']:,.0f}" if isinstance(f.get('price'), (int, float)) else 'unavailable'
        text = f"Listing {item['listing_id']} · {f.get('location') or 'Unknown city'} · MLS beds: {f.get('beds')} · baths: {f.get('baths')} · listed price: {price}"
        self.metadata.value = '<span>' + html.escape(text) + '</span>'
        self.reviewer.value = row['reviewer'] or reviewer_hint
        for key, field in self.scores.items():
            field.value = row[key] if row[key] is not None else 0
        self.notes.value = row['notes']
        self.reference.value = row['reference_summary']
        self.approval.value = row['reference_approved']
        self.previous.disabled = self.index == 0
        self.next.disabled = self.index == len(self.store.items)-1
        self._refresh_progress()
        self._loading = False
        self._message('Loaded saved review.' if row['reviewer'] or row['notes'] else 'Not yet rated.')

    def save(self, *, next_index=None):
        values = {'reviewer': self.reviewer.value, 'notes': self.notes.value,
                  'reference_summary': self.reference.value, 'reference_approved': self.approval.value,
                  **{key: field.value or None for key, field in self.scores.items()}}
        try:
            self.store.save(self.store.items[self.index]['listing_id'], values,
                            last_index=self.index if next_index is None else next_index)
        except (OSError, ValueError) as exc:
            self._message(str(exc), error=True)
            return False
        self._refresh_progress()
        self._message('Saved locally. Use Update evaluation to refresh the Week 8 report.')
        return True

    def go(self, index):
        if not 0 <= index < len(self.store.items):
            return False
        if not self.save(next_index=index):
            return False
        self.index = index
        self._render()
        return True

    def _jumped(self, change):
        if not self._loading and not self.go(change['new']):
            self._loading = True
            self.jump.value = self.index
            self._loading = False

    def apply(self):
        if not self.save() or self.on_apply is None:
            return False
        self.apply_button.disabled = True
        self._message('Updating saved ratings and evaluation…')
        try:
            report = self.on_apply(self.store.to_frame())
            progress = report['human_evaluation']
            self._message(f"Evaluation updated: {progress['completed']} / {progress['required']} ratings complete; "
                          f"{report['human_verified_references']} references approved.")
        except Exception as exc:
            self._message(f'Evaluation was not updated: {exc}. Your local draft is still saved.', error=True)
            return False
        finally:
            self.apply_button.disabled = False
        return True


def create_week8_review_widget(*, root=ROOT, draft_path=None):
    """Load current test summaries; constructing the widget does not write ratings."""
    from scripts.listing_summarizer import ListingSummarizer
    from scripts.import_week8_review import apply_review_frame
    root = Path(root)
    reference_path = root / TEST_REFERENCE_PATH.relative_to(ROOT)
    cases = json.loads(reference_path.read_text())['cases']
    summarizer = ListingSummarizer()
    items = [{**case, **summarizer.summarize_listing(case['record'])} for case in cases]
    output_dir = root / 'data/processed'
    ratings_path = output_dir / 'week8_human_ratings.json'
    existing = json.loads(ratings_path.read_text()) if ratings_path.exists() else []
    store = SummaryReviewStore(items, draft_path or output_dir / 'week8_widget_reviews.json', existing_reviews=existing)
    return ListingReviewWidget(store, on_apply=lambda frame: apply_review_frame(
        frame, reference_path=reference_path, output_dir=output_dir))
