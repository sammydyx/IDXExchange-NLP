# Week 9: Fair Housing advertising language screening

## Scope and legal basis

The checker screens proposed listing descriptions and summaries for language suggesting a resident preference, exclusion or limitation. The federal Fair Housing advertising provisions cover race, color, national origin, religion, sex, familial status and disability. See [42 U.S.C. 3604(c)](https://uscode.house.gov/us/usc/t42/s3604/) and [24 CFR 100.75](https://www.ecfr.gov/current/title-24/subtitle-B/chapter-I/part-100/subpart-B/section-100.75).

This is a limited English-language NLP screening exercise. A result identifies matches under a documented policy; it is not a legal finding, accessibility certification, verification of property facts, or assurance of compliance with all federal, California or local law. It does not infer an applicant's protected characteristics or make tenant eligibility decisions.

## How to write and review property descriptions

Describe objective features of the dwelling: room counts, layout, physical amenities, access features and factual location details. Resident exclusions and preferences receive correction or review findings. Each finding includes its matched text, category, severity, local context, explanation, suggested editing direction and source links.

The sample phrases below illustrate the project's screening policy, rather than a government-issued universal banned-word list:

| Description | Screening outcome | Editing direction |
|---|---|---|
| No children | Error in ordinary dwelling-ad context | Remove the household exclusion; describe the dwelling |
| Must be able-bodied | Error | Remove the resident physical-ability requirement |
| White tenants only | Error | Remove the racial resident preference |
| Christians only | Error unless a claimed exception is routed to review | Remove the preference or obtain qualified review of exception facts |
| Perfect for singles | Warning | Describe layout and features without choosing a household profile |
| White cabinets / ethnic restaurants nearby / diverse area | No risk rule hit by these phrases alone | Preserve factual property/location wording; no universal legality claim |
| Wheelchair accessible entrance | Informational finding | Verify the physical claim; do not treat it as a resident exclusion |
| 55+ community; adults only | Warning | Verify eligibility; the words alone grant no exemption |

The quoted 1995 HUD guidance reproduced in a [DOJ court filing appendix](https://www.justice.gov/sites/default/files/crt/legacy/2012/07/10/miamivalleybrief.pdf) helps explain why phrase context matters. It is historical explanatory material; current statutes and regulations govern.

## Exceptions are not granted by matching words

[42 U.S.C. 3607](https://uscode.house.gov/us/usc/t42/s3607/) contains limited older-person housing and religious-organization/private-club provisions. A senior or 55+/62+ claim is sent to review rather than used to automatically approve child-exclusion wording. Religious ownership claims and shared-living contexts also require qualified review; this tool does not verify eligibility. It does not assume that an owner-occupied or single-family listing is automatically exempt from advertising restrictions.

Program and ownership facts, actual occupancy, policies and reliable verification may matter. Text alone is insufficient. Claimed exceptions never suppress race or color exclusion findings. State/local requirements can differ and are outside this rule library.

## Severity and submission behavior

- **error**: block this submission until the text is corrected.
- **warning**: require qualified human review; the demo does not automatically approve or publish it.
- **info**: explain contextual wording or a physical-access description. It does not count as a violation or positive detection in precision/recall.

The compatibility key compliant means the same as passed_screening: no blocking/review finding under these rules. The output always includes legal_compliance_certified = false.

Matching uses case-insensitive regular expressions, Unicode NFKC, HTML-entity decoding, tag removal, dash normalization and whitespace normalization. Findings contain offsets into normalized_text, not raw HTML. Explicitly rejected phrases and clear former wording followed by an inclusive replacement retain informational evidence. A distant inclusive slogan cannot cancel a current exclusion; an owner's quoted exclusion remains flagged. These are bounded heuristics and can still be fooled by difficult negation, hypothetical wording, other languages, obfuscation and long-distance context.

## Files and usage

| File | Responsibility |
|---|---|
| data/fair_housing_patterns.json | Versioned pattern rules, severities, messages, suggestions and official source URLs |
| scripts/compliance_checker.py | ComplianceChecker.check_listing and contextual adjustment |
| scripts/listing_submission.py | prepare_listing_submission: raw-description check and Week 8 summary check |
| scripts/evaluate_week9.py | Evaluation metrics, code/rule fingerprints, error cases and unlabeled real-sample scan |
| tests/fixtures/week9_compliance_development.json | 68 synthetic development examples |
| tests/fixtures/week9_compliance_test.json | 103 synthetic test examples |
| tests/test_week9.py | Classification, severity, formatting, metric accounting and submission behavior tests |
| notebooks/week9_fair_housing_compliance.ipynb | Executed explanations, findings, evaluation and submission demonstration |

From the project root:

    source .venv/bin/activate
    python scripts/evaluate_week9.py
    python -m pytest -q tests/test_week9.py

Select the existing IDX Exchange NLP (.venv) kernel for the notebook. No Docker service or external publication account is required.

## Evaluation protocol

The 171 fixtures are assistant-authored synthetic examples with unverified author labels. Development and test texts are distinct, but phrase families overlap and the rules were developed with awareness of the test examples. They are regression fixtures, not an independent held-out legal benchmark. No independent human review or real-world generalization claim is made. Test cases are frozen in JSON; the evaluator records fixture, implementation and rule-library SHA-256 hashes.

Labels are known_violation (an explicit resident exclusion/preference in the stipulated ordinary housing-ad context), review (ambiguous language or claimed exception), and clean (no policy risk finding expected). Case-level screening counts error/warning as positive and info-only as negative. Overlapping rules do not count as separate examples.

- Screening precision = TP / (TP + FP); target greater than 0.80.
- Screening recall = TP / (TP + FN).
- Known-violation recall = known_violation examples flagged / all known_violation examples; target 1.00 on this set.
- Known-violation error recall additionally measures how many are blocked, rather than merely sent to review.
- Decision accuracy checks block vs review vs pass separately.

Per-category recall and individual errors are included. The reporting code is tested with intentionally introduced TP/FP/FN/TN, so failures are not omitted from metric calculations. During initial development, a historical quote followed by now everyone is welcome was falsely blocked; local historical-context handling was corrected without relabeling the fixture. This iteration is another reason not to call the test benchmark blind.

The real Week 1 sample and saved Week 8 summaries are also screened. Their flags are saved for inspection, but these records have no independent compliance labels. Flag counts are not counts of proven violations, and no precision/recall is claimed for that scan.

Generated results are in data/processed/week9_compliance_evaluation.json, week9_compliance_report.md, week9_case_results.jsonl and week9_sample_screening.jsonl. Inspect the JSON/report for the current measured values rather than assuming targets were achieved.

## Integration example

prepare_listing_submission screens original remarks before calling the summarizer, so an innocuous summary cannot hide a restriction in the original description. If a summary is supplied or generated, it is also checked. The return status is blocked, pending_review or ready_for_submission. It never contacts a publishing service; published is always false.

This is one language-screening step, not validation of every listing field or approval of a transaction. A qualified review process would need documented decisions, scope and audit records before a real deployment.
