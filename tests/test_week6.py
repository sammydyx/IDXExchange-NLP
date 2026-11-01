from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.signal_extraction import SignalExtractor


@pytest.fixture(scope="module")
def extractor():
    return SignalExtractor()


def test_signal_schema_combines_source_fields_entities_and_taxonomy(extractor):
    listing = {
        "L_ListingID": "WK6-001",
        "L_City": "Irvine",
        "L_SystemPrice": 850000,
        "L_Keyword2": 3,
        "LM_Dec_3": 2.5,
        "LM_Int2_3": 1800,
        "L_Type_": "Townhouse",
        "L_Remarks": (
            "Move-in ready two-story townhouse with 3 bedrooms, 2.5 bathrooms, "
            "1,800 square feet, and listed at $850,000. Enjoy a pool and fireplace "
            "near dining with seller financing, but without a spa."
        ),
    }

    signals = extractor.extract_signals(listing)

    assert signals["listing_id"] == "WK6-001"
    assert signals["listing_attributes"] == {
        "city": "Irvine", "listed_price": 850000, "listed_bedrooms": 3,
        "listed_bathrooms": 2.5, "listed_square_feet": 1800, "property_type_field": "Townhouse",
    }
    assert signals["entities"] == {"bedrooms": 3, "bathrooms": 2.5, "square_feet": 1800, "price": 850000}
    assert signals["amenities"] == ["pool"]
    assert "spa" in signals["negated_terms"]
    assert "move-in ready" in signals["keywords"]["condition_keywords"]
    assert "seller financing" in signals["keywords"]["financing_terms"]
    assert "near dining" in signals["keywords"]["location_features"]
    assert "townhouse" in signals["keywords"]["property_types"]
    assert "two-story" in signals["keywords"]["layout_features"]
    assert "fireplace" in signals["keywords"]["interior_features"]


def test_taxonomy_matching_handles_plural_and_hyphen_variants(extractor):
    signals = extractor.extract_signals({
        "listing_id": "WK6-002",
        "remarks": "Ocean-view condo with pools, a private backyard, and updated finishes.",
    })
    assert "pool" in signals["amenities"]
    assert "ocean views" in signals["keywords"]["location_features"]
    assert "condo" in signals["keywords"]["property_types"]
    assert "private backyard" in signals["keywords"]["exterior_features"]
    assert "updated" in signals["keywords"]["condition_keywords"]


def test_missing_remarks_produces_empty_signals(extractor):
    signals = extractor.extract_signals({"listing_id": 12345, "remarks": None})
    assert signals["listing_id"] == "12345"
    assert signals["entities"] == {"bedrooms": None, "bathrooms": None, "square_feet": None, "price": None}
    assert signals["amenities"] == []
    assert all(values == [] for values in signals["keywords"].values())


def test_listing_id_is_required(extractor):
    with pytest.raises(ValueError, match="listing_id"):
        extractor.extract_signals({"remarks": "A home with a pool"})


def test_input_must_be_a_mapping(extractor):
    with pytest.raises(TypeError, match="mapping"):
        extractor.extract_signals("not a listing")
