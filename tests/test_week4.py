from pathlib import Path
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.query_parser import QueryFilters, QueryParser, QueryValidationError, SafeSQLGenerator


VALID_CITIES = {"Irvine", "Pasadena", "Los Angeles", "San Diego", "Beverly Hills", "Upland", "Rancho Cucamonga"}


@pytest.fixture(scope="module")
def parser():
    return QueryParser(valid_cities=VALID_CITIES)


@pytest.fixture(scope="module")
def generator(parser):
    return SafeSQLGenerator(parser)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Homes in Irvine under $800k", {"city": "Irvine", "price_max": 800000}),
        ("properties above 1.2m in Pasadena", {"city": "Pasadena", "price_min": 1200000}),
        ("between $700k and $1 million", {"price_min": 700000, "price_max": 1000000}),
        ("from 500000 to 750000", {"price_min": 500000, "price_max": 750000}),
        ("at most $950,000", {"price_max": 950000}),
        ("Need a house lower than 1.2M in Rancho Cucamonga", {"price_max": 1200000}),
        ("3+ bedrooms", {"beds_min": 3}),
        ("at least four beds", {"beds_min": 4}),
        ("between 2 and 4 bedrooms", {"beds_min": 2, "beds_max": 4}),
        ("3 bedroom homes", {"beds_min": 3, "beds_max": 3}),
        ("no more than 2 beds", {"beds_max": 2}),
        ("2.5 baths", {"baths_min": 2.5, "baths_max": 2.5}),
        ("at least 3 bathrooms", {"baths_min": 3}),
        ("between one and three baths", {"baths_min": 1, "baths_max": 3}),
        ("homes with a pool and spa", {"amenities_include": ["pool", "spa"]}),
        ("condos without a pool", {"amenities_exclude": ["pool"]}),
        ("no gym or sauna", {"amenities_exclude": ["gym", "sauna"]}),
        ("with EV charging", {"amenities_include": ["ev charging"]}),
        ("with a pool but no spa", {"amenities_include": ["pool"], "amenities_exclude": ["spa"]}),
    ],
)
def test_query_patterns(parser, query, expected):
    filters = parser.parse(query)
    for key, value in expected.items():
        assert getattr(filters, key) == value


def test_room_counts_are_not_mistaken_for_prices(parser):
    filters = parser.parse(
        "Find 2 to 5 bedroom homes with at least 3 bathrooms in Upland with no pool"
    )
    assert filters.price_min is None
    assert filters.price_max is None
    assert filters.beds_min == 2
    assert filters.beds_max == 5
    assert filters.baths_min == 3
    assert filters.city == "Upland"


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        (QueryFilters(city="Mars City"), "Unknown city"),
        (QueryFilters(price_max=-1), "price_max"),
        (QueryFilters(beds_min=21), "beds_min"),
        (QueryFilters(baths_max=21), "baths_max"),
        (QueryFilters(price_min=900000, price_max=800000), "price minimum"),
        (QueryFilters(beds_min=4, beds_max=2), "beds minimum"),
        (QueryFilters(amenities_include=["pool"], amenities_exclude=["pool"]), "Cannot both"),
    ],
)
def test_schema_validation(parser, filters, message):
    with pytest.raises(QueryValidationError, match=message):
        parser.validate(filters)


def test_invalid_city_in_query_is_rejected(parser):
    with pytest.raises(QueryValidationError, match="Unknown city"):
        parser.parse("Find homes in Mars City")


def test_empty_query_is_rejected(parser):
    with pytest.raises(QueryValidationError, match="non-empty"):
        parser.parse("   ")


def test_sql_generator_uses_placeholders_and_separate_params(parser, generator):
    filters = parser.parse("3+ bedroom homes in Irvine under $1m with a pool but no spa")
    sql, params = generator.build(filters)
    assert "Irvine" not in sql
    assert "1m" not in sql
    assert "%s" in sql
    assert params == ["irvine", 1000000, 3, "%pool%", "%spa%", 50]
    assert "LOWER(COALESCE(L_Remarks, '')) NOT LIKE %s" in sql


def test_sql_injection_is_not_concatenated(parser, generator):
    malicious = "Irvine'; DROP TABLE rets_property; --"
    with pytest.raises(QueryValidationError, match="Unknown city"):
        parser.validate(QueryFilters(city=malicious))
    safe_filters = parser.parse("homes in Irvine with a pool")
    sql, params = generator.build(safe_filters)
    assert "DROP TABLE" not in sql
    assert "DROP TABLE" not in " ".join(map(str, params))
    assert "Irvine" not in sql


SAMPLE_QUERY_PATH = Path(__file__).resolve().parents[1] / "data/processed/sample_user_queries.csv"
SAMPLE_ROWS = pd.read_csv(SAMPLE_QUERY_PATH).to_dict("records")
EXPECTATIONS_PATH = Path(__file__).resolve().parent / "fixtures/week4_expected_filters.json"
EXPECTED_SAMPLE_FILTERS = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))

FILTER_FIELDS = (
    "city", "price_min", "price_max", "beds_min", "beds_max", "baths_min", "baths_max",
    "amenities_include", "amenities_exclude", "terms_include", "terms_exclude",
)
EMPTY_FILTERS = {
    "city": None,
    "price_min": None,
    "price_max": None,
    "beds_min": None,
    "beds_max": None,
    "baths_min": None,
    "baths_max": None,
    "amenities_include": [],
    "amenities_exclude": [],
    "terms_include": [],
    "terms_exclude": [],
}


def expected_filters(query_id: str) -> dict[str, object]:
    """Return the complete expected structured output for one Week 1 query."""
    return EMPTY_FILTERS | EXPECTED_SAMPLE_FILTERS[query_id]


def parsed_filter_values(parser: QueryParser, query: str) -> dict[str, object]:
    """Exclude diagnostic categories; compare the filters that drive SQL instead."""
    filters = parser.parse(query)
    return {field: getattr(filters, field) for field in FILTER_FIELDS}


def test_every_week1_query_has_an_expected_filter_fixture():
    query_ids = {row["query_id"] for row in SAMPLE_ROWS}
    assert set(EXPECTED_SAMPLE_FILTERS) == query_ids


@pytest.mark.parametrize("row", SAMPLE_ROWS, ids=lambda row: row["query_id"])
def test_week1_sample_queries_extract_expected_filters(row, parser):
    """Each of the 64 examples must produce the SQL-driving filter values.

    `matched_categories` is diagnostic metadata.  It cannot prove that a term
    such as ``townhouse`` or ``pool`` was placed in the correct include/exclude
    list, so it is deliberately not used as the correctness criterion here.
    """
    assert parsed_filter_values(parser, row["query"]) == expected_filters(row["query_id"])


def test_week1_structured_filter_accuracy_is_at_least_90_percent(parser):
    correct = sum(
        parsed_filter_values(parser, row["query"]) == expected_filters(row["query_id"])
        for row in SAMPLE_ROWS
    )
    assert correct / len(SAMPLE_ROWS) >= 0.90
