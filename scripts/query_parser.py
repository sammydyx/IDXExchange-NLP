from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QueryValidationError(ValueError):
    """Raised when parsed filters are invalid for the real-estate schema."""


@dataclass
class QueryFilters:
    """Structured constraints extracted from one user property-search query."""

    city: str | None = None
    price_min: int | None = None
    price_max: int | None = None
    beds_min: float | None = None
    beds_max: float | None = None
    baths_min: float | None = None
    baths_max: float | None = None
    amenities_include: list[str] = field(default_factory=list)
    amenities_exclude: list[str] = field(default_factory=list)
    terms_include: list[str] = field(default_factory=list)
    terms_exclude: list[str] = field(default_factory=list)
    matched_categories: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation for notebooks and debugging."""
        return {
            "city": self.city,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "beds_min": self.beds_min,
            "beds_max": self.beds_max,
            "baths_min": self.baths_min,
            "baths_max": self.baths_max,
            "amenities_include": self.amenities_include,
            "amenities_exclude": self.amenities_exclude,
            "terms_include": self.terms_include,
            "terms_exclude": self.terms_exclude,
            "matched_categories": sorted(self.matched_categories),
        }


class QueryParser:
    """Parse constrained natural-language property searches into safe filters."""

    MAX_PRICE = 100_000_000
    MAX_BEDS = 20
    MAX_BATHS = 20
    NUMBER_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    CATEGORY_ALIASES = {
        "loft": ("loft", "property_type"),
        "good condition": ("well maintained", "condition"),
        "financing options": ("financing", "financing"),
        "hiking trails": ("nearby trails", "location_feature"),
        "close to restaurants": ("near dining", "location_feature"),
        "beach": ("beachfront", "location_feature"),
    }

    def __init__(
        self,
        taxonomy_path: str | Path | None = None,
        valid_cities: Iterable[str] | None = None,
    ) -> None:
        taxonomy_path = Path(taxonomy_path or PROJECT_ROOT / "data/processed/taxonomy.json")
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        self.taxonomy_terms = [
            (self._normalize(item["term"]), item["term"].lower(), item["category"])
            for item in taxonomy["terms"]
            if item.get("active", True)
        ]
        self.taxonomy_terms.sort(key=lambda item: len(item[0]), reverse=True)
        cities = valid_cities if valid_cities is not None else self._sample_cities()
        self.valid_cities = {str(city).strip().lower(): str(city).strip() for city in cities if str(city).strip()}

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).lower().replace("-", " ")).strip()

    def _sample_cities(self) -> set[str]:
        """Load local valid cities without requiring a live Docker connection."""
        for filename in ("listing_sample.csv", "listing_sample_cleaned.csv"):
            path = PROJECT_ROOT / "data/processed" / filename
            if path.exists():
                values = pd.read_csv(path, usecols=["L_City"])["L_City"].dropna().astype(str)
                return set(values.str.strip())
        return set()

    @classmethod
    def _number(cls, token: str) -> float:
        token = token.lower().replace(",", "").strip()
        if token in cls.NUMBER_WORDS:
            return float(cls.NUMBER_WORDS[token])
        return float(token)

    @staticmethod
    def _clean_number(value: float) -> int | float:
        return int(value) if value.is_integer() else value

    @staticmethod
    def _money_value(number: str, suffix: str | None) -> int:
        value = float(number.replace(",", ""))
        suffix = (suffix or "").lower().replace(" ", "")
        if suffix in {"k", "thousand"}:
            value *= 1_000
        elif suffix in {"m", "million"}:
            value *= 1_000_000
        return int(value)

    @staticmethod
    def _is_negated(text: str, start: int) -> bool:
        context = text[max(0, start - 30):start]
        return bool(re.search(r"\b(?:no|without|exclude|excluding|not)\b[^.]{0,24}$", context))

    def _extract_city(self, original: str, normalized: str) -> str | None:
        for normalized_city, display_city in sorted(self.valid_cities.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(r"(?<!\w)" + re.escape(normalized_city) + r"(?!\w)", normalized):
                return display_city

        # A capitalized phrase following "in" is likely intended as a city. Reject it
        # rather than silently returning an unfiltered search.
        candidate = re.search(r"\b(?:in|around)\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,2})", original)
        if candidate:
            raise QueryValidationError(f"Unknown city: {candidate.group(1)}")
        return None

    @staticmethod
    def _has_money_signal(match: re.Match[str], number_group: int, suffix_group: int) -> bool:
        """Reject small counts such as 'at least 3 bathrooms' as prices."""
        value = float(match.group(number_group).replace(",", ""))
        return "$" in match.group(0) or bool(match.group(suffix_group)) or value >= 100_000

    def _extract_price(self, text: str, filters: QueryFilters) -> None:
        money = r"\$?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(k|m|thousand|million)?"
        range_match = re.search(rf"\b(?:between|from)\s+{money}\s+(?:and|to|-)\s+{money}\b", text)
        if range_match and self._has_money_signal(range_match, 1, 2) and self._has_money_signal(range_match, 3, 4):
            filters.price_min = self._money_value(range_match.group(1), range_match.group(2))
            filters.price_max = self._money_value(range_match.group(3), range_match.group(4))
            return
        maximum = re.search(rf"\b(?:under|below|lower than|less than|at most|up to|maximum|max)\s+{money}\b", text)
        if maximum and self._has_money_signal(maximum, 1, 2):
            filters.price_max = self._money_value(maximum.group(1), maximum.group(2))
        # Do not read the 'more than' inside 'no more than 2 beds' as a price.
        minimum = re.search(rf"(?<!no )\b(?:over|above|more than|at least|minimum|min)\s+{money}\b", text)
        if minimum and self._has_money_signal(minimum, 1, 2):
            filters.price_min = self._money_value(minimum.group(1), minimum.group(2))

    def _extract_room_bounds(self, text: str, filters: QueryFilters, field: str) -> None:
        labels = r"(?:bedrooms?|beds?|br)" if field == "beds" else r"(?:bathrooms?|baths?|ba)"
        number = r"([0-9]+(?:\.5)?|one|two|three|four|five|six|seven|eight|nine|ten)"
        range_match = re.search(rf"\bbetween\s+{number}\s+and\s+{number}\s+{labels}\b", text)
        if not range_match:
            range_match = re.search(rf"\b{number}\s+to\s+{number}\s+{labels}\b", text)
        if range_match:
            minimum, maximum = self._number(range_match.group(1)), self._number(range_match.group(2))
            setattr(filters, f"{field}_min", self._clean_number(minimum))
            setattr(filters, f"{field}_max", self._clean_number(maximum))
            return
        minimum = re.search(rf"(?<!no )\b(?:at least|minimum|min|more than)\s+{number}\s*(?:\+|or more)?\s*{labels}\b", text)
        plus = re.search(rf"\b{number}\s*\+\s*{labels}\b", text)
        maximum = re.search(rf"\b(?:at most|maximum|max|no more than|under)\s+{number}\s*{labels}\b", text)
        exact = re.search(rf"\b{number}\s*{labels}\b", text)
        if minimum or plus:
            match = minimum or plus
            setattr(filters, f"{field}_min", self._clean_number(self._number(match.group(1))))
        elif maximum:
            setattr(filters, f"{field}_max", self._clean_number(self._number(maximum.group(1))))
        elif exact:
            value = self._clean_number(self._number(exact.group(1)))
            setattr(filters, f"{field}_min", value)
            setattr(filters, f"{field}_max", value)

    def _extract_taxonomy_terms(self, normalized: str, filters: QueryFilters) -> None:
        matched: set[tuple[str, str]] = set()
        for normalized_term, canonical_term, category in self.taxonomy_terms:
            plural = normalized_term + "s" if not normalized_term.endswith("s") else normalized_term
            if re.search(r"(?<!\w)(?:" + re.escape(normalized_term) + "|" + re.escape(plural) + r")(?!\w)", normalized):
                matched.add((canonical_term, category))
        for phrase, (canonical_term, category) in self.CATEGORY_ALIASES.items():
            if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", normalized):
                matched.add((canonical_term, category))

        for term, category in sorted(matched):
            first_position = normalized.find(self._normalize(term))
            negated = self._is_negated(normalized, first_position) if first_position >= 0 else False
            filters.matched_categories.add(category)
            if category == "amenity":
                target = filters.amenities_exclude if negated else filters.amenities_include
            else:
                target = filters.terms_exclude if negated else filters.terms_include
            if term not in target:
                target.append(term)

    def validate(self, filters: QueryFilters) -> None:
        if filters.city and filters.city.lower() not in self.valid_cities:
            raise QueryValidationError(f"Unknown city: {filters.city}")
        for name, value, maximum in (
            ("price_min", filters.price_min, self.MAX_PRICE),
            ("price_max", filters.price_max, self.MAX_PRICE),
            ("beds_min", filters.beds_min, self.MAX_BEDS),
            ("beds_max", filters.beds_max, self.MAX_BEDS),
            ("baths_min", filters.baths_min, self.MAX_BATHS),
            ("baths_max", filters.baths_max, self.MAX_BATHS),
        ):
            if value is not None and not 0 < value <= maximum:
                raise QueryValidationError(f"{name} must be between 0 and {maximum}")
        for minimum, maximum, label in (
            (filters.price_min, filters.price_max, "price"),
            (filters.beds_min, filters.beds_max, "beds"),
            (filters.baths_min, filters.baths_max, "baths"),
        ):
            if minimum is not None and maximum is not None and minimum > maximum:
                raise QueryValidationError(f"{label} minimum cannot exceed maximum")
        overlap = set(filters.amenities_include) & set(filters.amenities_exclude)
        if overlap:
            raise QueryValidationError(f"Cannot both require and exclude: {sorted(overlap)}")

    def parse(self, query: str) -> QueryFilters:
        if not isinstance(query, str) or not query.strip():
            raise QueryValidationError("Query must be a non-empty string")
        normalized = self._normalize(query)
        filters = QueryFilters(city=self._extract_city(query, normalized))
        self._extract_price(normalized, filters)
        self._extract_room_bounds(normalized, filters, "beds")
        self._extract_room_bounds(normalized, filters, "baths")
        self._extract_taxonomy_terms(normalized, filters)
        self.validate(filters)
        return filters


class SafeSQLGenerator:
    """Generate MySQL queries with placeholders and a separate parameter list."""

    BASE_QUERY = """
    SELECT L_ListingID, L_Address, L_City, L_Keyword2 AS beds,
        LM_Dec_3 AS baths, L_SystemPrice AS price, LM_Int2_3 AS square_feet,
        L_Type_ AS property_type, L_Remarks AS remarks
    FROM rets_property
    """.strip()

    def __init__(self, parser: QueryParser) -> None:
        self.parser = parser

    def build(self, filters: QueryFilters, limit: int = 50) -> tuple[str, list[object]]:
        self.parser.validate(filters)
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise QueryValidationError("limit must be an integer from 1 to 100")
        where: list[str] = []
        params: list[object] = []
        scalar_columns = {
            "city": "LOWER(TRIM(L_City))",
            "price_min": "L_SystemPrice >= %s",
            "price_max": "L_SystemPrice <= %s",
            "beds_min": "L_Keyword2 >= %s",
            "beds_max": "L_Keyword2 <= %s",
            "baths_min": "LM_Dec_3 >= %s",
            "baths_max": "LM_Dec_3 <= %s",
        }
        if filters.city:
            where.append(scalar_columns["city"] + " = %s")
            params.append(filters.city.lower())
        for name in ("price_min", "price_max", "beds_min", "beds_max", "baths_min", "baths_max"):
            value = getattr(filters, name)
            if value is not None:
                where.append(scalar_columns[name])
                params.append(value)
        for term in filters.amenities_include + filters.terms_include:
            where.append("LOWER(COALESCE(L_Remarks, '')) LIKE %s")
            params.append(f"%{term.lower()}%")
        for term in filters.amenities_exclude + filters.terms_exclude:
            where.append("LOWER(COALESCE(L_Remarks, '')) NOT LIKE %s")
            params.append(f"%{term.lower()}%")
        sql = self.BASE_QUERY
        if where:
            sql += "\nWHERE " + "\n  AND ".join(where)
        sql += "\nORDER BY L_SystemPrice ASC\nLIMIT %s"
        params.append(limit)
        return sql, params


def parse_and_build(query: str, parser: QueryParser | None = None, limit: int = 50) -> tuple[QueryFilters, str, list[object]]:
    """Convenience entry point used by the Week 4 notebook and future API layer."""
    parser = parser or QueryParser()
    filters = parser.parse(query)
    sql, params = SafeSQLGenerator(parser).build(filters, limit=limit)
    return filters, sql, params
