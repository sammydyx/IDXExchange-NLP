"""Week 6: combine entity extraction and taxonomy matching into listing signals."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any

from scripts.entity_extraction import EntityExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SignalExtractor:
    """Extract JSON-serializable search and filtering signals from one listing.

    ``listing_attributes`` are source fields supplied by the MLS record, while
    ``entities`` are values extracted from the listing remarks.  Keeping the
    two groups separate makes the output safe to use for both auditing and
    search indexing.
    """

    KEYWORD_FIELDS = {
        "condition": "condition_keywords",
        "financing": "financing_terms",
        "location_feature": "location_features",
        "property_type": "property_types",
        "layout": "layout_features",
        "interior_feature": "interior_features",
        "exterior_feature": "exterior_features",
    }

    def __init__(self, taxonomy_path: str | Path | None = None, entity_extractor: EntityExtractor | None = None):
        self.taxonomy_path = Path(taxonomy_path or PROJECT_ROOT / "data/processed/taxonomy.json")
        self.entity_extractor = entity_extractor or EntityExtractor(self.taxonomy_path)
        self.taxonomy_terms = self._load_taxonomy_terms()

    @staticmethod
    def _normalize(text: object) -> str:
        return re.sub(r"\s+", " ", str(text or "").lower().replace("-", " ")).strip()

    @staticmethod
    def _term_pattern(term: str) -> re.Pattern[str]:
        """Match a taxonomy phrase with optional pluralization of its last word."""
        words = term.split()
        last_word = words[-1]
        singular_last = last_word[:-1] if last_word.endswith("s") else last_word
        escaped_words = [re.escape(word) for word in words[:-1]] + [re.escape(singular_last) + r"s?"]
        return re.compile(r"(?<!\w)" + r"\s+".join(escaped_words) + r"(?!\w)")

    def _load_taxonomy_terms(self) -> list[dict[str, Any]]:
        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        terms = []
        for item in taxonomy["terms"]:
            if not item.get("active", True):
                continue
            normalized = self._normalize(item["term"])
            terms.append({
                "term": str(item["term"]).lower().strip(),
                "category": item["category"],
                "pattern": self._term_pattern(normalized),
            })
        return sorted(terms, key=lambda item: (-len(item["term"]), item["term"]))

    @staticmethod
    def _is_negated(text: str, start: int) -> bool:
        """Treat phrases such as ``no pool`` and ``without a spa`` as exclusions."""
        context = text[max(0, start - 36):start]
        return bool(re.search(r"\b(?:no|without|exclude|excluding|not)\b(?:\s+\w+){0,3}\s*$", context))

    def match_taxonomy(self, remarks: object) -> tuple[dict[str, list[str]], list[str]]:
        """Return positive category matches and explicitly negated taxonomy terms."""
        normalized = self._normalize(remarks)
        matches: dict[str, list[str]] = {category: [] for category in set(self.KEYWORD_FIELDS) | {"amenity"}}
        negated_terms: list[str] = []
        for item in self.taxonomy_terms:
            match = item["pattern"].search(normalized)
            if not match:
                continue
            if self._is_negated(normalized, match.start()):
                negated_terms.append(item["term"])
                continue
            category_terms = matches.setdefault(item["category"], [])
            if item["term"] not in category_terms:
                category_terms.append(item["term"])
        return matches, sorted(set(negated_terms))

    @staticmethod
    def _first(record: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in record and record[key] is not None:
                return record[key]
        return None

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, Decimal):
            return int(value) if value == value.to_integral_value() else float(value)
        return value

    def extract_signals(self, listing_record: Mapping[str, Any]) -> dict[str, Any]:
        """Create the Week 6 JSON record for a single MLS listing."""
        if not isinstance(listing_record, Mapping):
            raise TypeError("listing_record must be a mapping")

        listing_id = self._first(listing_record, "listing_id", "L_ListingID")
        if listing_id is None:
            raise ValueError("listing_record must contain listing_id or L_ListingID")
        remarks = self._first(listing_record, "remarks", "L_Remarks") or ""
        taxonomy_matches, negated_terms = self.match_taxonomy(remarks)
        # Week 6 has its own taxonomy pass below, covering every category.
        # Calling ``extract_entities`` here would scan the amenity/interior/
        # exterior taxonomy a second time only to discard those feature lists.
        # Reuse just the four Week 3 structured-entity methods instead.
        extracted_entities = {
            "bedrooms": self.entity_extractor.extract_bedrooms(remarks),
            "bathrooms": self.entity_extractor.extract_bathrooms(remarks),
            "square_feet": self.entity_extractor.extract_square_feet(remarks),
            "price": self.entity_extractor.extract_price(remarks),
        }

        keywords = {
            field: taxonomy_matches.get(category, [])
            for category, field in self.KEYWORD_FIELDS.items()
        }
        return {
            "listing_id": str(listing_id),
            "listing_attributes": {
                "city": self._json_value(self._first(listing_record, "city", "L_City")),
                "listed_price": self._json_value(self._first(listing_record, "listed_price", "price", "L_SystemPrice")),
                "listed_bedrooms": self._json_value(self._first(listing_record, "listed_bedrooms", "beds", "L_Keyword2")),
                "listed_bathrooms": self._json_value(self._first(listing_record, "listed_bathrooms", "baths", "LM_Dec_3")),
                "listed_square_feet": self._json_value(self._first(listing_record, "listed_square_feet", "square_feet", "LM_Int2_3")),
                "property_type_field": self._json_value(self._first(listing_record, "property_type", "L_Type_")),
            },
            "entities": {
                "bedrooms": extracted_entities["bedrooms"],
                "bathrooms": extracted_entities["bathrooms"],
                "square_feet": extracted_entities["square_feet"],
                "price": extracted_entities["price"],
            },
            "amenities": taxonomy_matches.get("amenity", []),
            "keywords": keywords,
            "negated_terms": negated_terms,
        }
