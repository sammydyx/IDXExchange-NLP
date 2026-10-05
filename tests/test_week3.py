from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.entity_extraction import EntityExtractor


@pytest.fixture(scope="module")
def extractor():
    return EntityExtractor()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A 3 bedroom home", 3),
        ("3 bed, 2 bath", 3),
        ("4BR residence", 4),
        ("Modern 3-bedroom condo", 3),
        ("A two-bedroom cottage", 2),
        ("2 beautiful bedrooms", 2),
        ("Three spacious bedrooms", 3),
        ("Bedrooms, 2 in total", 2),
        ("A secondary bedroom", None),
        ("An oversized primary suite", None),
        ("Beautiful bedrooms and 2 bathrooms", None),
        ("Two bedrooms", 2),
        ("No bedroom count stated", None),
    ],
)
def test_extract_bedrooms(extractor, text, expected):
    assert extractor.extract_bedrooms(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A 2 bathroom home", 2),
        ("2.5 baths", 2.5),
        ("3BA townhouse", 3),
        ("Modern 2-bath condo", 2),
        ("Three full bathrooms", 3),
        ("2 luxurious full bathrooms", 2),
        ("Bathrooms, three in total", 3),
        ("A renovated bathroom", None),
        ("No bath count stated", None),
    ],
)
def test_extract_bathrooms(extractor, text, expected):
    assert extractor.extract_bathrooms(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,500 square feet", 1500),
        ("900 sq ft", 900),
        ("750 sqft", 750),
        ("1,200 SF", 1200),
        ("11,545-square-foot home", 11545),
        ("1,800-square-feet", 1800),
        ("No size disclosed", None),
    ],
)
def test_extract_square_feet(extractor, text, expected):
    assert extractor.extract_square_feet(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Listed at $850,000", 850000),
        ("Price: $850k", 850000),
        ("Offered at $1.2M", 1200000),
        ("asking 975k", 975000),
        ("No price is stated", None),
    ],
)
def test_extract_price(extractor, text, expected):
    assert extractor.extract_price(text) == expected


def test_taxonomy_feature_matching(extractor):
    entities = extractor.extract_entities("Updated kitchen, pool, solar panels, and a fireplace.")
    assert "pool" in entities["amenities"]
    assert "kitchen" in entities["interior_features"]
    assert "solar panels" in entities["exterior_features"]


def test_word_boundaries_prevent_false_matches(extractor):
    features = extractor.extract_taxonomy_features("The spool is stored inside.")
    assert "pool" not in [item["term"] for item in features]


def test_full_entity_record(extractor):
    entities = extractor.extract_entities("A 3BR, 2.5 bath home with 1,800 sqft, a pool, and a fireplace.")
    assert entities["bedrooms"] == 3
    assert entities["bathrooms"] == 2.5
    assert entities["square_feet"] == 1800
    assert "pool" in entities["amenities"]


def test_scalar_entity_metric_counts_wrong_values_as_fp_and_fn():
    import pandas as pd
    from scripts.evaluate_entities import score_scalar

    dataframe = pd.DataFrame({
        "predicted": [3, 2, None, 1],
        "expected": [3, 3, 2, None],
    })
    reviewed, tp, fp, fn, precision, recall, f1 = score_scalar(dataframe, "predicted", "expected")
    assert (reviewed, tp, fp, fn) == (4, 1, 2, 2)
    assert precision == 1 / 3
    assert recall == 1 / 3
    assert f1 == 1 / 3


def test_bathroom_full_decimal_and_written_half_counts(extractor):
    assert extractor.extract_bathrooms("The home has 3 full bathrooms.") == 3
    assert extractor.extract_bathrooms("It includes 2.75 bathrooms.") == 2.75
    assert extractor.extract_bathrooms("Features two-and-a-half bathrooms.") == 2.5


def test_square_feet_prefers_primary_living_area_over_lot_or_outdoor_area(extractor):
    text = (
        "Set on a 6,600 square foot lot, this home offers 1,320 square feet "
        "of comfortable living space and a 240 square feet outdoor deck."
    )
    assert extractor.extract_square_feet(text) == 1320


def test_square_feet_prefers_main_residence_over_total_multi_unit_space(extractor):
    text = (
        "Offering 2,600 square feet of total living space, the 4-bedroom main "
        "residence has approximately 2100 square feet plus a 500 square feet guest house."
    )
    assert extractor.extract_square_feet(text) == 2100
