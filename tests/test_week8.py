"""Behavior tests for source-grounded summaries and answerability routing."""
from pathlib import Path
import json
import sys
from unittest.mock import Mock

import pandas as pd
import pytest
from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.listing_summarizer import ListingSummarizer
from scripts.answerability import AnswerabilityChecker, search_and_summarize
from scripts.query_parser import QueryParser
from scripts.build_week8_artifacts import evaluate, REFERENCE_PATH, TEST_REFERENCE_PATH, summary_hash
from scripts.import_week8_review import parse_review_rows


@pytest.fixture(scope="module")
def summarizer():
    return ListingSummarizer()


@pytest.fixture()
def listing():
    return {"L_ListingID": "demo", "L_City": "Irvine", "beds": 3, "baths": 2.5,
            "price": 950000, "remarks": "A bright 3-bedroom home in Irvine. A private pool and attached garage offer convenience. Contact the agent today."}


@pytest.fixture(scope="module")
def checker():
    return AnswerabilityChecker(parser=QueryParser(valid_cities=["Irvine", "San Diego", "Upland"]))


def test_main_summary_uses_source_sentences_and_keeps_mls_facts_separate(summarizer, listing):
    result = summarizer.summarize_listing(listing)
    assert "3-bedroom" in result["summary"] and "Irvine" in result["summary"]
    assert "$950,000" not in result["summary"] and "2.5 bathrooms" not in result["summary"]
    assert result['facts']['price'] == 950000 and result['facts']['baths'] == 2.5
    assert len(result["top_features"]) == 2
    assert {'baths', 'price'} <= set(result["missing_fields"])
    assert result['method'] == 'extractive'
    assert result['summary'] == ' '.join(row['text'] for row in result['selected_sentences'])
    assert 2 <= len(summarizer.split_sentences(result["summary"])) <= 3
    assert all(e["sentence"] in listing["remarks"] for e in result["feature_evidence"])


def test_extraction_contains_only_original_sentences_in_source_order(summarizer, listing):
    result = summarizer.extractive_summary(listing["remarks"])
    original = summarizer.split_sentences(listing["remarks"])
    selected = summarizer.split_sentences(result)
    assert all(sentence in original for sentence in selected)
    assert [original.index(s) for s in selected] == sorted(original.index(s) for s in selected)
    assert len(selected) == 2


def test_decimal_and_abbreviation_sentence_boundaries(summarizer):
    sentences = summarizer.split_sentences("A 2.5-bath home with 1,200 sq. ft. of living space. It has a pool.")
    assert len(sentences) == 2
    assert "2.5" in sentences[0] and "sq. ft." in sentences[0]


@pytest.mark.parametrize("number", [0, 4, -1, 2.5, True])
def test_invalid_sentence_count_is_rejected(summarizer, number):
    with pytest.raises(ValueError):
        summarizer.extractive_summary("A home. A pool.", num_sentences=number)


@pytest.mark.parametrize("text", [None, "", float("nan")])
def test_empty_remarks_do_not_generate_extracted_content(summarizer, text):
    assert summarizer.extractive_summary(text) == ""


@pytest.mark.parametrize("value", [None, [], "a listing"])
def test_listing_requires_a_mapping(summarizer, value):
    with pytest.raises(TypeError):
        summarizer.summarize_listing(value)


def test_html_and_entities_are_cleaned_without_losing_text(summarizer):
    result = summarizer.extractive_summary("<p>Home &amp; garden.</p> <p>A pool is included.</p>")
    assert "<p>" not in result and "Home & garden." in result and "pool" in result


def test_rental_price_in_remarks_never_replaces_list_price(summarizer, listing):
    listing["remarks"] = "It was rented for $2,300. A private pool is included."
    result = summarizer.summarize_listing(listing)
    assert result['facts']['price'] == 950000
    assert '$950,000' not in result['summary']
    assert '$950,000' in result['template_summary']
    del listing["price"]
    result = summarizer.summarize_listing(listing)
    assert "price" in result["missing_fields"] and result['facts']['price'] is None
    assert 'price' in result['not_in_remarks']
    # A historical-rent sentence may be copied, but cannot be relabeled as sale price.
    assert all(row['text'] in listing['remarks'] for row in result['selected_sentences'])


def test_nested_week6_attributes_take_priority(summarizer, listing):
    listing["listing_attributes"] = {"city": "San Diego", "listed_bedrooms": 4,
                                     "listed_bathrooms": 3, "listed_price": 1200000}
    listing["entities"] = {"bedrooms": 2, "price": 999}
    result = summarizer.summarize_listing(listing)
    assert result["facts"] == {"beds": 4, "baths": 3, "price": 1200000, "location": "San Diego"}


def test_missing_numeric_fields_are_not_hallucinated(summarizer):
    result = summarizer.summarize_listing({"remarks": "A sunny home with a pool."})
    assert {"beds", "baths", "price", "location"} <= set(result["missing_fields"])
    assert "bedroom" not in result["summary"]


def test_room_count_conflict_is_visible(summarizer, listing):
    listing["remarks"] = "This home has 4 bedrooms and 3 bathrooms. A pool is included."
    result = summarizer.summarize_listing(listing)
    assert len(result["warnings"]) == 2
    assert "verify" not in result["summary"].lower()
    assert "4 bedrooms and 3 bathrooms" in result['summary']
    assert result["facts"]["beds"] == 3
    assert len(summarizer.split_sentences(result["summary"])) == 2


@pytest.mark.parametrize("text,excluded", [
    ("No pool or spa is included.", "pool"),
    ("Without a pool, this home has a garden.", "pool"),
    ("There is room for a pool.", "pool"),
    ("There is space to add a pool.", "pool"),
    ("A spa-inspired bathroom is included.", "spa"),
    ("The garage has been converted into living space.", "garage"),
])
def test_negated_hypothetical_or_converted_features_are_not_claimed(summarizer, text, excluded):
    assert not any(excluded == e["term"] for e in summarizer.feature_evidence(text))


def test_repeated_feature_is_not_counted_twice(summarizer):
    terms = [e["term"] for e in summarizer.feature_evidence("A pool is included. Enjoy the pool.")]
    assert terms.count("pool") == 1


@pytest.mark.parametrize("query", [
    "Find 3+ bedroom homes in Irvine under $1M with a pool but no spa",
    "Irvine under $1M",
    "Find 2 to 5 bedroom homes in Upland with at least 3 bathrooms",
    "Homes with no pool in San Diego",
])
def test_supported_query_reuses_week4_filters(checker, query):
    assert checker.assess_query(query)["answerable"]


@pytest.mark.parametrize("query,reason", [
    ("", "empty_query"), (None, "empty_query"),
    ("Find homes in mars", "invalid_city"),
    ("Find 99 bedroom homes in Irvine", "invalid_filters"),
    ("Find homes between $2M and $1M", "invalid_filters"),
    ("Find homes for -$100", "invalid_value"),
    ("Tell me a bedtime story", "out_of_domain"),
    ("Predict home prices next year", "unsupported_question"),
    ("Homes with school ratings in Irvine", "unsupported_question"),
    ("Homes with at least 2000 sqft", "unsupported_question"),
    ("How many homes are in Irvine?", "unsupported_question"),
    ("Homes in Irvine or Upland", "unsupported_logic"),
    ("Homes; DROP TABLE rets_property", "unsafe_query"),
    ("Homes OR 1=1 --", "unsafe_query"),
])
def test_unanswerable_query_has_a_specific_reason(checker, query, reason):
    result = checker.assess_query(query)
    assert not result["answerable"] and result["reason"] == reason
    assert result["message"]


def test_rejected_query_never_reaches_database(checker):
    executor = Mock()
    result = search_and_summarize("DROP TABLE rets_property;", checker=checker, search_function=executor)
    executor.assert_not_called()
    assert not result["answerable"]


@pytest.mark.parametrize("frame,reason", [
    (pd.DataFrame(), "no_matches"),
    (pd.DataFrame([{"price": None, "city": None}]), "empty_content"),
    (pd.DataFrame([{"L_ListingID": "123", "remarks": "  "}]), "empty_content"),
    (None, "data_unavailable"),
])
def test_empty_or_meaningless_results_are_explained(checker, frame, reason):
    assert checker.assess_results(frame)["reason"] == reason


def test_db_connection_failure_is_distinct_from_no_matches(checker):
    result = search_and_summarize("homes in Irvine", checker=checker,
                                  search_function=Mock(side_effect=ConnectionError("offline")))
    assert result["reason"] == "database_unavailable"


def test_end_to_end_summary_with_injected_executor(checker, summarizer, listing):
    result = search_and_summarize("homes in Irvine", checker=checker, summarizer=summarizer,
                                  search_function=Mock(return_value=pd.DataFrame([listing])))
    assert result["answerable"] and len(result["results"]) == 1
    assert result["filters"]["city"] == "Irvine"
    assert result["results"][0]["facts"]["price"] == 950000


def test_rouge_identical_and_disjoint_text():
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    assert scorer.score("A home with a pool", "A home with a pool")["rougeL"].fmeasure == 1
    assert scorer.score("pool garage", "ocean mountains")["rougeL"].fmeasure == 0


def test_references_are_fixed_real_records_with_honest_review_status(summarizer):
    cases = json.loads(REFERENCE_PATH.read_text())["cases"]
    assert len(cases) == len({c["listing_id"] for c in cases}) == 20
    assert all(c["reference_summary"] and c["record"]["remarks"] for c in cases)
    assert all(c['source_only'] and c['reference_summary'] == ' '.join(c['reference_sentences']) for c in cases)
    assert all(sentence in summarizer.split_sentences(c['record']['remarks']) for c in cases for sentence in c['reference_sentences'])
    result = evaluate(cases, summarizer)
    assert result["reference_count"] == 20
    assert all(0 <= score <= 1 for group in result["scores"].values() for score in group.values())
    if any(c["review_status"] != "manual_verified" for c in cases):
        assert result["evaluation_status"] == "provisional_draft_references"


def test_test_and_development_listing_ids_are_disjoint():
    root = Path(__file__).resolve().parents[1] / 'tests/fixtures'
    development = json.loads(REFERENCE_PATH.read_text())['cases']
    test = json.loads(TEST_REFERENCE_PATH.read_text())['cases']
    old_test = json.loads((root / 'week8_test_references.json').read_text())['cases']
    assert len(test) == 20
    assert {c['listing_id'] for c in development}.isdisjoint(c['listing_id'] for c in test)
    assert {c['listing_id'] for c in old_test}.isdisjoint(c['listing_id'] for c in test)
    assert all(c['split'] == 'test' for c in test)


def review_frame():
    cases = [{'listing_id': 'a', 'reference_summary': 'A home.', 'review_status': 'needs_human_review'}]
    frame = pd.DataFrame([{'Listing ID': 'a', 'Reviewer': '', 'Factual accuracy (1-5)': None,
                           'Completeness (1-5)': None, 'Readability (1-5)': None,
                           'Approve reference': '', 'Notes': '', 'Reference summary': 'A home.'}])
    return frame, cases


def test_blank_reviews_remain_incomplete():
    frame, cases = review_frame()
    updated, ratings = parse_review_rows(frame, cases)
    assert ratings[0]['status'] == 'incomplete'
    assert updated[0]['review_status'] == 'needs_human_review'


def test_rating_completion_does_not_imply_reference_approval():
    frame, cases = review_frame()
    frame.loc[0, ['Reviewer', 'Factual accuracy (1-5)', 'Completeness (1-5)', 'Readability (1-5)']] = ['Test fixture reviewer', 5, 4, 3]
    updated, ratings = parse_review_rows(frame, cases)
    assert ratings[0]['status'] == 'complete'
    assert updated[0]['review_status'] == 'needs_human_review'


def test_reference_approval_requires_named_reviewer():
    frame, cases = review_frame()
    frame.loc[0, 'Approve reference'] = 'yes'
    with pytest.raises(ValueError, match='reviewer'):
        parse_review_rows(frame, cases)
    frame.loc[0, 'Reviewer'] = 'Test fixture reviewer'
    updated, _ = parse_review_rows(frame, cases)
    assert updated[0]['review_status'] == 'manual_verified'
    assert cases[0]['review_status'] == 'needs_human_review'


@pytest.mark.parametrize('invalid', [0, 6, 2.5, 'excellent'])
def test_invalid_human_rating_is_rejected(invalid):
    frame, cases = review_frame()
    frame.loc[0, 'Factual accuracy (1-5)'] = invalid
    with pytest.raises(ValueError, match='integer'):
        parse_review_rows(frame, cases)


def test_mismatched_review_id_is_rejected():
    frame, cases = review_frame()
    frame.loc[0, 'Listing ID'] = 'wrong'
    with pytest.raises(ValueError, match='IDs'):
        parse_review_rows(frame, cases)


def test_sale_price_cents_are_preserved(summarizer, listing):
    listing['price'] = 999999.99
    result = summarizer.summarize_listing(listing)
    assert result['facts']['price'] == 999999.99
    assert '$999,999.99' in result['template_summary']
    assert '$999,999.99' not in result['summary']


@pytest.mark.parametrize('text,kind,expected', [
    ('There are two beautiful bedrooms.', 'beds', [2]),
    ('Enjoy three large sunny bedrooms.', 'beds', [3]),
    ('It has 2.5 bathrooms.', 'baths', [2.5]),
    ('This home has 2 1/2 bathrooms.', 'baths', [2.5]),
    ('A 2-car garage and spacious bedrooms.', 'beds', []),
])
def test_sentence_scoring_understands_room_evidence(summarizer, text, kind, expected):
    assert summarizer._room_counts(text, kind) == expected


def test_room_counts_and_distinct_features_beat_repeated_decor(summarizer):
    text = ('An elegant kitchen has granite countertops and stainless steel appliances. '
            'This Irvine home offers three beautiful bedrooms and two bathrooms. '
            'Outside, enjoy a private pool and covered patio. '
            'The kitchen has more granite countertops and stainless steel appliances.')
    result = summarizer.extractive_details(text, {'city': 'Irvine'}, 2)
    assert 'three beautiful bedrooms and two bathrooms' in result['summary']
    assert 'private pool' in result['summary']
    assert result['covered_roles']


def test_material_repair_warning_is_retained_from_source(summarizer):
    text = ('This home has three bedrooms and two bathrooms. '
            'The kitchen offers quartz countertops and stainless steel appliances. '
            'Bathroom requires repair including plumbing and drywall.')
    result = summarizer.summarize_listing({'remarks': text})
    assert 'Bathroom requires repair' in result['summary']


def test_template_facts_cannot_leak_into_extraction_when_remarks_empty(summarizer, listing):
    listing['remarks'] = ''
    result = summarizer.summarize_listing(listing)
    assert result['summary'] == '' and not result['selected_sentences']
    assert result['facts']['price'] == 950000


def test_stale_template_review_cannot_be_imported_as_extractive_review():
    frame, cases = review_frame()
    frame['Summary'] = 'A template with fields absent from remarks.'
    with pytest.raises(ValueError, match='Summary changed'):
        parse_review_rows(frame, cases, expected_summaries={'a': 'A home.'})
    frame['Summary'] = 'A home.'
    _, ratings = parse_review_rows(frame, cases, expected_summaries={'a': 'A home.'})
    assert ratings[0]['summary_sha256'] == summary_hash('A home.')


def test_fractional_bath_conflict_does_not_read_denominator_as_count(summarizer, listing):
    listing['remarks'] = 'This home has 2 1/2 bathrooms and a pool.'
    assert not summarizer.summarize_listing(listing)['warnings']


def test_mysql_executor_keeps_remarks_for_summary(monkeypatch, checker, listing):
    from scripts import search_listings as module
    connection, cursor = Mock(), Mock()
    connection.cursor.return_value = cursor
    cursor.fetchall.return_value = [listing]
    monkeypatch.setattr(module.mysql.connector, 'connect', Mock(return_value=connection))
    frame, filters, sql, params = module.search_listings('homes in Irvine', parser=checker.parser)
    assert frame.iloc[0]['remarks'] == listing['remarks']
    assert '%s' in sql and 'Irvine' not in sql
    assert 'irvine' in params
