"""Create a transparent, balanced Week 7 seed annotation dataset.

The generated labels are a structured starting point for human review.  They
are deliberately marked ``needs_human_review`` and must not be described as
manually verified labels until a reviewer changes that status.
"""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data/processed/week7_intent_seed_draft.csv"

CITIES = [
    "Irvine", "San Diego", "Beverly Hills", "Los Angeles", "Pasadena",
    "Long Beach", "Sacramento", "Santa Monica", "Anaheim", "Riverside",
]
PROPERTY_TYPES = ["homes", "condos", "townhomes", "single-family homes", "duplexes"]
AMENITIES = ["pools", "fireplaces", "large yards", "home offices", "solar panels"]

BROWSING_TEMPLATES = [
    "show me {property_type} in {city}",
    "find {property_type} in {city}",
    "browse {property_type} around {city}",
    "list {property_type} in {city}",
    "what {property_type} are for sale in {city}",
    "show {property_type} with {amenity} in {city}",
    "find {property_type} near {city}",
    "I am looking for {property_type} in {city}",
]

RESEARCHING_TEMPLATES = [
    "which neighborhoods in {city} have lower property taxes",
    "compare HOA fees for {property_type} in {city}",
    "what should I know about buying {property_type} in {city}",
    "which areas near {city} have good schools",
    "are {property_type} in {city} expensive to maintain",
    "how do property taxes compare around {city}",
    "what is the typical HOA for {property_type} in {city}",
    "which {city} neighborhoods are best for families",
]

HIGH_INTENT_TEMPLATES = [
    "schedule a tour of {property_type} in {city} this week",
    "show open houses in {city} this weekend",
    "I need to see move-in ready {property_type} in {city} today",
    "contact an agent about {property_type} in {city}",
    "I am ready to make an offer on a {property_type} in {city}",
    "find newly listed {property_type} in {city} available now",
    "book a viewing for {property_type} with {amenity} in {city}",
    "urgent: {property_type} in {city} I can tour tomorrow",
]


def _queries(templates: list[str]) -> list[str]:
    """Create 80 distinct wording examples from eight intent templates."""
    queries = []
    for index in range(80):
        template_index = index % len(templates)
        city_index = index // len(templates)
        queries.append(
            templates[template_index].format(
                city=CITIES[city_index],
                property_type=PROPERTY_TYPES[(city_index + template_index) % len(PROPERTY_TYPES)],
                amenity=AMENITIES[(city_index + template_index) % len(AMENITIES)],
            )
        )
    return queries


def build_dataset(output_path: Path = OUTPUT_PATH) -> int:
    rows = []
    for intent, templates in [
        ("browsing", BROWSING_TEMPLATES),
        ("researching", RESEARCHING_TEMPLATES),
        ("high_intent_inquiry", HIGH_INTENT_TEMPLATES),
    ]:
        rows.extend((query, intent) for query in _queries(templates))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "query", "intent", "label_source", "review_status", "notes"],
        )
        writer.writeheader()
        for index, (query, intent) in enumerate(rows, start=1):
            writer.writerow({
                "query_id": f"W7-{index:03d}",
                "query": query,
                "intent": intent,
                "label_source": "curated_seed_draft",
                "review_status": "needs_human_review",
                "notes": "",
            })
    return len(rows)


if __name__ == "__main__":
    count = build_dataset()
    print(f"Saved {count} draft-labeled queries to {OUTPUT_PATH}")
