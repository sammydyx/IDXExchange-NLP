from __future__ import annotations

import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.query_parser import QueryParser


QUERY_PATH = PROJECT_ROOT / "data/processed/sample_user_queries.csv"
REPORT_PATH = PROJECT_ROOT / "data/processed/week4_query_parser_report.md"
EXPECTATIONS_PATH = PROJECT_ROOT / "tests/fixtures/week4_expected_filters.json"
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


def structured_filter_values(parser: QueryParser, query: str) -> dict[str, object]:
    filters = parser.parse(query)
    return {field: getattr(filters, field) for field in FILTER_FIELDS}


def main() -> None:
    dataframe = pd.read_csv(QUERY_PATH)
    expected_by_id = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    query_ids = set(dataframe["query_id"])
    if set(expected_by_id) != query_ids:
        raise ValueError("The expected-filter fixture must cover every Week 1 query exactly once.")
    parser = QueryParser()
    dataframe["expected_filters"] = dataframe["query_id"].map(
        lambda query_id: EMPTY_FILTERS | expected_by_id[query_id]
    )
    dataframe["parsed_filters"] = dataframe["query"].map(
        lambda query: structured_filter_values(parser, query)
    )
    dataframe["correct"] = dataframe["parsed_filters"] == dataframe["expected_filters"]
    accuracy = dataframe["correct"].mean()
    failures = dataframe.loc[
        ~dataframe["correct"], ["query_id", "query", "expected_filters", "parsed_filters"]
    ]
    lines = [
        "# Week 4 Query Parser Evaluation",
        "",
        f"- Week 1 queries evaluated: {len(dataframe)}",
        f"- Structured-filter parsing accuracy: {accuracy:.1%}",
        f"- Target: 90.0%+",
        "- Correctness compares city, price, beds, baths, amenity include/exclude, and taxonomy-term include/exclude fields.",
        "",
        "## Unmatched queries",
        "",
    ]
    if failures.empty:
        lines.append("All Week 1 sample queries matched their expected structured filters.")
    else:
        lines.extend(
            f"- {row.query_id}: expected `{row.expected_filters}`, got `{row.parsed_filters}`"
            for row in failures.itertuples()
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
