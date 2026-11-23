"""Week 9: source-linked, context-aware Fair Housing advertising screening."""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {'race', 'color', 'national_origin', 'religion', 'sex', 'familial_status', 'disability'}
LEVELS = {'error', 'warning', 'info'}


class ComplianceChecker:
    """Screen advertising language, without certifying legal compliance.

    It normalizes formatting, matches language patterns, and checks the surrounding context.
    The output includes the matched phrase, its category, severity, and an explanation.

    three severity level:
    error: stop this submission until the text is corrected.
    warning: route to qualified human review; no automatic publication.
    info: explanatory finding, not a violation and not a publication blocker.
    """

    def __init__(self, pattern_path=None):
        payload = json.loads(Path(pattern_path or ROOT/'data/fair_housing_patterns.json').read_text())
        self.version = payload['version']
        self.sources = {source['id']: source for source in payload['sources']}
        self.rules = payload['rules']
        ids = [rule['id'] for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError('Rule IDs must be unique')
        self.patterns = []
        for rule in self.rules:
            if rule['category'] not in CATEGORIES or rule['severity'] not in LEVELS:
                raise ValueError('Invalid category or severity in pattern library')
            if not set(rule['source_ids']) <= self.sources.keys():
                raise ValueError('Every rule must reference a known source')
            self.patterns.append((rule, re.compile(rule['regex'], re.I)))

    @staticmethod
    def normalize(text):
        text = unicodedata.normalize('NFKC', html.unescape(text))
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub('[\u2010-\u2015-]', ' ', text)
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def _context(text, start, end):
        """Return local context; a distant disclaimer cannot erase an exclusion."""
        before, after = text[max(0, start-140):start].lower(), text[end:end+140].lower()
        return before, after

    def _adjust(self, rule, text, start, end):
        before, after = self._context(text, start, end)
        # Suppression requires an explicit rejection/quotation context, not merely
        # the presence of an inclusive slogan somewhere else in the listing.
        rejected = bool(re.search(r"(?:do not|don't|never|must not|should not)\s+(?:write|use|say|include|advertise)\s+(?:the\s+phrase\s+)?[\"']?\s*$", before))
        rejected |= bool(re.search(r"(?:reject|prohibit|avoid)\s+(?:the\s+)?(?:phrase|wording|expression)\s+[\"']?\s*$", before))
        explicitly_negated = bool(re.search(r'\bnot\s+$', before) and re.match(r"\s*[\"']?(?:[;,.]|$)", after))
        historical = bool(re.search(r'\b(?:previously|formerly)\s*[\"\']?\s*$', before) and
                          re.search(r'\bnow\s+(?:all\s+ages|families|everyone)\s+(?:(?:are|is)\s+)?welcome\b', after))
        no_exclusion = text[start:end].lower().startswith('no ') and bool(
            re.match(r'\s+(?:are|will be)\s+excluded\b', after))
        absent_adult_policy = rule['id'] == 'familial.adults_only' and bool(
            re.search(r'\b(?:no|without)\s+$', before) and
            re.match(r'\s+(?:restrictions?|policy|policies)\b', after))
        if rule['severity'] != 'info' and (rejected or explicitly_negated or historical or no_exclusion or absent_adult_policy):
            return 'info', 'explicit_rejection_or_historical_quote', 'The local text rejects or replaces this wording; inspect the context if it is repurposed.'
        category = rule['category']
        if category == 'familial_status' and rule['severity'] == 'error':
            if rule['id'] == 'familial.adults_only' and re.match(r'\s+(?:pool|spa|swim|event|class|lounge)\b', after):
                return 'warning', 'amenity_policy_context', 'An amenity policy is mentioned; review its context rather than treating it as a verified dwelling occupancy restriction.'
            if re.search(r'\b(?:55|62)\s*\+|\b(?:senior|retirement)\s+(?:housing|community|living)\b|\b(?:55|62)\s+(?:and|or)\s+(?:over|older)\b', text, re.I):
                return 'warning', 'older_person_housing_claim_unverified', 'Age-restricted housing is claimed; verify actual statutory eligibility. This does not grant an exemption.'
        if category == 'religion' and rule['severity'] == 'error':
            if re.search(r'\b(?:owned|operated)\s+by\s+(?:a\s+)?(?:religious\s+organi[sz]ation|church|synagogue|mosque)\b|\bnoncommercial\s+religious\s+housing\b', text, re.I):
                return 'warning', 'religious_organization_claim_unverified', 'A limited statutory exception is claimed; ownership, operation and noncommercial facts require review.'
        if category == 'sex' and rule['severity'] == 'error':
            if re.search(r'\b(?:roommate|room mate|housemate|shared bathroom|shared kitchen)\b', text, re.I):
                return 'warning', 'shared_living_context_unverified', 'Shared-living context requires qualified review; it is not an automatic exemption.'
        return rule['severity'], None, rule['message']

    def check_listing(self, text):
        if not isinstance(text, str) or not self.normalize(text):
            return self._result('', [{'rule_id': 'input.empty', 'category': 'input_validation',
                'severity': 'error', 'pattern': None, 'matched_text': '', 'span': None,
                'context': '', 'context_reason': None, 'message': 'Listing text must be a non-empty string.',
                'suggestion': 'Provide the proposed listing description before submission.', 'sources': []}])
        normalized = self.normalize(text)
        if len(normalized) > 100000:
            raise ValueError('Listing text exceeds the 100,000-character screening limit')
        findings = []
        for rule, pattern in self.patterns:
            for match in pattern.finditer(normalized):
                severity, reason, message = self._adjust(rule, normalized, *match.span())
                source_ids = list(rule['source_ids'])
                if rule['id'] == 'familial.senior_claim' or reason in {
                    'older_person_housing_claim_unverified', 'religious_organization_claim_unverified'
                }:
                    source_ids.append('fha_3607')
                findings.append({
                    'rule_id': rule['id'], 'category': rule['category'], 'severity': severity,
                    'pattern': rule['regex'], 'matched_text': match.group(),
                    'span': list(match.span()), 'context': normalized[max(0, match.start()-90):match.end()+90],
                    'context_reason': reason, 'message': message, 'suggestion': rule['suggestion'],
                    'sources': [self.sources[key]['url'] for key in dict.fromkeys(source_ids)],
                })
        findings.sort(key=lambda f: ({'error': 0, 'warning': 1, 'info': 2}[f['severity']], f['span'][0], f['rule_id']))
        return self._result(normalized, findings)

    def _result(self, normalized, findings):
        counts = Counter(row['severity'] for row in findings)
        decision = 'block' if counts['error'] else 'review' if counts['warning'] else 'pass'
        return {
            'compliant': decision == 'pass',  # Compatibility alias: screening only.
            'passed_screening': decision == 'pass', 'requires_review': decision == 'review',
            'decision': decision, 'legal_compliance_certified': False, 'rule_version': self.version,
            'normalized_text': normalized, 'span_basis': 'normalized_text',
            'counts': {level: counts[level] for level in ('error', 'warning', 'info')},
            'findings': findings, 'violations': [f for f in findings if f['severity'] in {'error', 'warning'}],
        }
