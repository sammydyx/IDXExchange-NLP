from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EntityExtractor:
    """Extract structured real-estate entities from listing remarks."""

    NUMBER_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    BEDROOM_PATTERN = re.compile(
        r"\b(\d+(?:\.5)?)\s*-?\s*(?:bedrooms?|beds?|br|bdrms?|bd)\b", re.IGNORECASE
    )
    BEDROOM_WORD_PATTERN = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*-?\s*(?:bedrooms?|beds?)\b", re.IGNORECASE
    )
    BATHROOM_PATTERN = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*-?\s*(?:full\s+|half\s+)?(?:bathrooms?|baths?|ba|bth)\b", re.IGNORECASE
    )
    BATHROOM_WORD_PATTERN = re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)(?:(?:\s*-\s*|\s+)and(?:\s*-\s*|\s+)a(?:\s*-\s*|\s+)half)?\s+(?:full\s+)?(?:bathrooms?|baths?)\b", re.IGNORECASE
    )
    SQFT_PATTERN = re.compile(
        r"\b([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*(?:-?\s*)(?:square\s*-?\s*(?:foot|feet)|sq\.?\s*-?\s*ft\.?|sqft|sf)\b",
        re.IGNORECASE,
    )
    DOLLAR_PRICE_PATTERN = re.compile(
        r"\$\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.(\d+))?\s*([km])?\b",
        re.IGNORECASE,
    )
    COMPACT_PRICE_PATTERN = re.compile(
        r"\b(?:price(?:d)?|offered at|listed at|asking)\s*(?:at|for)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([km])\b",
        re.IGNORECASE,
    )

    def __init__(self, taxonomy_path: str | Path | None = None):
        self.taxonomy_path = Path(taxonomy_path or PROJECT_ROOT / "data/processed/taxonomy.json")
        self.taxonomy_terms = self._load_taxonomy_terms()

    def _load_taxonomy_terms(self) -> list[dict[str, str]]:
        """Load only active terms that describe amenities or physical features."""
        with self.taxonomy_path.open(encoding="utf-8") as file:
            taxonomy = json.load(file)

        entity_categories = {"amenity", "interior_feature", "exterior_feature"}
        terms = [
            {"term": item["term"].lower().strip(), "category": item["category"]}
            for item in taxonomy["terms"]
            if item.get("active", True) and item.get("category") in entity_categories
        ]
        return sorted(terms, key=lambda item: (-len(item["term"]), item["term"]))

    @staticmethod
    def _number(match: str) -> int | float:
        value = float(match)
        return int(value) if value.is_integer() else value

    @staticmethod
    def _first_match(pattern: re.Pattern, text: object) -> str | None:
        match = pattern.search(str(text or ""))
        return match.group(1) if match else None

    @classmethod
    def _token_to_number(cls, token: str) -> int | float | None:
        if token.lower() in cls.NUMBER_WORDS:
            return cls.NUMBER_WORDS[token.lower()]
        # "a/an bathroom" identifies one room, but does not establish the
        # total number of bathrooms in the property. Scalar fields therefore
        # require an explicit numeric or written cardinal value.
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return cls._number(token)
        return None

    def _nearby_count(
        self,
        text: str,
        target_nouns: set[str],
        other_entity_nouns: set[str],
        window: int = 3,
    ) -> int | float | None:
        """Find a number within three tokens of a bedroom/bathroom noun."""
        tokens = re.findall(r"\d+(?:\.\d+)?|[A-Za-z]+", text.lower())
        candidates: list[tuple[int, int, int, int | float]] = []
        for noun_index, token in enumerate(tokens):
            if token not in target_nouns:
                continue
            for index in range(max(0, noun_index - window), min(len(tokens), noun_index + window + 1)):
                if index == noun_index:
                    continue
                value = self._token_to_number(tokens[index])
                if value is None:
                    continue
                # Avoid assigning the count in "bedrooms and 2 bathrooms" to bedrooms.
                if index > noun_index and any(
                    following in other_entity_nouns
                    for following in tokens[index + 1 : min(len(tokens), index + window + 1)]
                ):
                    continue
                distance = abs(noun_index - index)
                direction = 0 if index < noun_index else 1
                candidates.append((noun_index, distance, direction, value))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0][3]

    def _extract_count(
        self,
        text: object,
        numeric_pattern: re.Pattern,
        word_pattern: re.Pattern,
        target_nouns: set[str],
        other_entity_nouns: set[str],
    ) -> int | float | None:
        text = str(text or "")
        numeric_match = numeric_pattern.search(text)
        word_match = word_pattern.search(text)
        if numeric_match and (not word_match or numeric_match.start() <= word_match.start()):
            return self._number(numeric_match.group(1))
        if word_match:
            base_value = self.NUMBER_WORDS[word_match.group(1).lower()]
            return base_value + 0.5 if "half" in word_match.group(0).lower() else base_value
        return self._nearby_count(text, target_nouns, other_entity_nouns)

    def extract_bedrooms(self, text: object) -> int | float | None:
        return self._extract_count(
            text,
            self.BEDROOM_PATTERN,
            self.BEDROOM_WORD_PATTERN,
            {"bedroom", "bedrooms", "bed", "beds", "br", "bdrm", "bdrms", "bd", "suite"},
            {"bath", "baths", "bathroom", "bathrooms", "ba", "bth"},
        )

    def extract_bathrooms(self, text: object) -> int | float | None:
        return self._extract_count(
            text,
            self.BATHROOM_PATTERN,
            self.BATHROOM_WORD_PATTERN,
            {"bath", "baths", "bathroom", "bathrooms", "ba", "bth"},
            {"bed", "beds", "bedroom", "bedrooms", "br", "bdrm", "bdrms", "bd"},
        )

    @staticmethod
    def _square_feet_candidate_score(text: str, match: re.Match, order: int) -> tuple[int, int]:
        """Rank likely primary-property square-footage mentions by local context."""
        before = text[max(0, match.start() - 50):match.start()].lower()
        after = text[match.end():match.end() + 50].lower()
        score = 0
        if re.search(r"\bmain residence\b", before):
            score += 15
        if re.search(r"\b(?:interior|comfortable) living\b|\bliving (?:space|area)\b", after):
            score += 12
        if re.search(r"\b(?:residence|home|house)\s+(?:offers|has|totaling|includes)\b", before):
            score += 8
        if re.search(r"\b(?:structure|building)\b", before):
            score += 6
        if re.search(r"^\s*(?:of\s+)?(?:a\s+)?(?:lot|parcel|acre|land|yard|garage|deck|patio|outdoor)\b", after):
            score -= 7
        if re.search(r"\b(?:ranging|range|from)\b", before[-45:]) or re.search(r"^\s*(?:-|to)\b", after[:10]):
            score -= 5
        # Earlier mentions win exact ties, preserving the original stable behavior.
        return score, -order

    def extract_square_feet(self, text: object) -> int | None:
        """Extract the most likely primary-property square-footage value."""
        normalized = str(text or "")
        matches = list(self.SQFT_PATTERN.finditer(normalized))
        if not matches:
            return None
        best_match = max(
            enumerate(matches),
            key=lambda item: self._square_feet_candidate_score(normalized, item[1], item[0]),
        )[1]
        return int(best_match.group(1).replace(",", ""))

    def extract_price(self, text: object) -> int | None:
        """Extract the first explicitly dollar-denominated or labelled compact price."""
        text = str(text or "")
        match = self.DOLLAR_PRICE_PATTERN.search(text)
        if match:
            number = float(match.group(1).replace(",", "") + ("." + match.group(2) if match.group(2) else ""))
            suffix = (match.group(3) or "").lower()
            multiplier = 1_000 if suffix == "k" else 1_000_000 if suffix == "m" else 1
            return int(number * multiplier)

        compact = self.COMPACT_PRICE_PATTERN.search(text)
        if compact:
            multiplier = 1_000 if compact.group(2).lower() == "k" else 1_000_000
            return int(float(compact.group(1)) * multiplier)
        return None

    def extract_taxonomy_features(self, text: object) -> list[dict[str, str]]:
        normalized = str(text or "").lower()
        matches = []
        for item in self.taxonomy_terms:
            if re.search(r"(?<!\w)" + re.escape(item["term"]) + r"(?!\w)", normalized):
                matches.append(item)
        return matches

    def extract_entities(self, text: object) -> dict[str, Any]:
        features = self.extract_taxonomy_features(text)
        return {
            "bedrooms": self.extract_bedrooms(text),
            "bathrooms": self.extract_bathrooms(text),
            "square_feet": self.extract_square_feet(text),
            "price": self.extract_price(text),
            "amenities": [item["term"] for item in features if item["category"] == "amenity"],
            "interior_features": [item["term"] for item in features if item["category"] == "interior_feature"],
            "exterior_features": [item["term"] for item in features if item["category"] == "exterior_feature"],
        }


def main() -> None:
    source = PROJECT_ROOT / "data/processed/listing_sample_cleaned.csv"
    output = PROJECT_ROOT / "data/processed/listing_sample_entities.csv"
    dataframe = pd.read_csv(source)
    extractor = EntityExtractor()

    extracted = dataframe.apply(
        lambda row: extractor.extract_entities(row["remarks_cleaned"]), axis=1
    )
    dataframe["bedrooms_extracted"] = extracted.apply(lambda item: item["bedrooms"])
    dataframe["bathrooms_extracted"] = extracted.apply(lambda item: item["bathrooms"])
    dataframe["square_feet_extracted"] = extracted.apply(lambda item: item["square_feet"])
    # Use raw remarks for price because cleaning intentionally removes compact-price suffixes.
    dataframe["price_mentioned_extracted"] = dataframe["remarks"].apply(extractor.extract_price)
    for column, key in [
        ("amenities_extracted", "amenities"),
        ("interior_features_extracted", "interior_features"),
        ("exterior_features_extracted", "exterior_features"),
    ]:
        dataframe[column] = extracted.apply(lambda item: json.dumps(item[key]))

    dataframe.to_csv(output, index=False)
    print(f"Saved {len(dataframe):,} listings with extracted entities to {output}")
    print(f"Bedroom mentions: {dataframe['bedrooms_extracted'].notna().sum():,}")
    print(f"Bathroom mentions: {dataframe['bathrooms_extracted'].notna().sum():,}")
    print(f"Square-footage mentions: {dataframe['square_feet_extracted'].notna().sum():,}")
    print(f"Price mentions: {dataframe['price_mentioned_extracted'].notna().sum():,}")


if __name__ == "__main__":
    main()
