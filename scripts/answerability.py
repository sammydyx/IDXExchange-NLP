"""Week 8: explain unsupported searches before SQL and empty results after SQL."""
from __future__ import annotations

import re
from typing import Callable

import mysql.connector
import pandas as pd

from scripts.query_parser import QueryParser, QueryValidationError
from scripts.listing_summarizer import ListingSummarizer


class AnswerabilityChecker:
    """
    Rule-based scope checking, reusing Week 4 parsing and validation.

    Before searching, it reuses the Week 4 parser and validation
    to identify invalid cities, invalid filters, and unsupported questions.

    After searching, it distinguishes between
    no matching listings, empty results, and database connection problems.
    """

    UNSUPPORTED = [
        (r"\b(?:weather|recipe|football|stock market)\b", "This search service contains property listings, not that information."),
        (r"\b(?:predict|forecast|appreciat\w*|next year|future price|investment return|guaranteed return)\b", "The listing database cannot predict future prices or investment returns."),
        (r"\b(?:school ratings?|school rankings?|crime rates?|property tax rates?|hoa fees?|commute times?)\b", "The current search schema does not provide verified school ratings, crime rates, taxes, HOA fees, or commute estimates."),
        (r"\b(?:how many|average price|median price|price history)\b", "The Week 4 search supports individual listing filters, not aggregate or historical analysis."),
        (r"\b(?:\d+(?:\.\d+)?\s*(?:sqft|sq\.?\s*ft\.?|square feet|miles?|acres?)|built after|built before|year built)\b", "Area, distance, acreage, and construction-year filters are not supported by the current Week 4 parser."),
    ]

    def __init__(self, taxonomy=None, schema_validator=None, *, parser=None):
        self.parser = parser or QueryParser()
        self.validator = schema_validator or self.parser

    def assess_query(self, query) -> dict:
        def reject(reason, message):
            return {"answerable": False, "reason": reason, "message": message, "filters": None}
        if not isinstance(query, str) or not query.strip():
            return reject("empty_query", "Please enter a property-search query.")
        query = query.strip()
        if re.search(r";|--|/\*|\*/|\b(?:drop|delete|insert|update)\s+(?:table|from|into)\b|\bor\s+1\s*=\s*1", query, re.I):
            return reject("unsafe_query", "Please enter a property search without SQL commands or comments.")
        for pattern, message in self.UNSUPPORTED:
            if re.search(pattern, query, re.I):
                return reject("unsupported_question", message)
        if re.search(r"(?<!\w)-\s*\$?\s*\d", query):
            return reject("invalid_value", "Price and room-count filters cannot be negative.")
        if re.search(r"\bor\b(?!\s+(?:more|less)\b)", query, re.I):
            return reject("unsupported_logic", "This parser supports combined AND filters; OR alternatives need separate searches.")
        # Week 4 detects unknown capitalized cities. Also catch lowercase city
        # names rather than letting an unrecognized location become no filter.
        location = re.search(r"\b(?:in|around|near)\s+([a-z][a-z ]*?)(?=\s+(?:under|below|with|without|over|above|between|for|at least|at most)\b|[.!?]|$)", query, re.I)
        if location:
            city = location.group(1).strip().lower()
            if city and not re.match(r"(?:a|an|the)\s", city) and city not in self.parser.valid_cities:
                return reject("invalid_city", f"City '{location.group(1).strip()}' is not in the available city list.")
        try:
            filters = self.parser.parse(query)
            if hasattr(self.validator, "validate_query"):
                valid, errors = self.validator.validate_query(filters)
                if not valid:
                    return reject("invalid_filters", "; ".join(errors))
            else:
                self.validator.validate(filters)
        except QueryValidationError as error:
            return reject("invalid_filters", str(error))
        real_terms = re.search(r"\b(?:house|houses|home|homes|property|properties|listing|listings|condos?|townhomes?|townhouses?|bedrooms?|bathrooms?|beds?|baths?|pool|garage|sqft|price)\b", query, re.I)
        if not real_terms and not filters.city and not filters.matched_categories:
            return reject("out_of_domain", "This does not appear to be a supported real-estate listing search.")
        return {"answerable": True, "reason": "supported_search", "message": "The query can be searched using the available listing fields.", "filters": filters.as_dict()}

    def check_pre_query(self, query) -> tuple[bool, str]:
        result = self.assess_query(query)
        return result["answerable"], result["message"]

    def assess_results(self, results_df) -> dict:
        if results_df is None:
            return {"answerable": False, "reason": "data_unavailable", "message": "No database response is available."}
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError("results_df must be a pandas DataFrame")
        if results_df.empty:
            return {"answerable": False, "reason": "no_matches", "message": "No listings match your criteria. Try relaxing a filter."}
        content = results_df.drop(columns=["listing_id", "L_ListingID", "id"], errors="ignore")
        meaningful = content.applymap(lambda value: bool(ListingSummarizer.text(value))).any().any()
        if not meaningful:
            return {"answerable": False, "reason": "empty_content", "message": "The query returned listings without meaningful property details."}
        return {"answerable": True, "reason": "results_found", "message": f"Found {len(results_df)} matching listings."}

    def check_post_query(self, query, results_df) -> tuple[bool, str]:
        result = self.assess_results(results_df)
        return result["answerable"], result["message"]


def search_and_summarize(query: str, *, limit=5, checker=None, summarizer=None,
                         search_function: Callable | None = None) -> dict:
    """Inject a local executor for offline demos; default to Week 4 MySQL SQL.

    Unsupported requests never reach SQL. Database connection failures remain
    distinguishable from successful searches with zero matching properties.
    """
    checker = checker or AnswerabilityChecker()
    before = checker.assess_query(query)
    if not before["answerable"]:
        return {**before, "results": []}
    if search_function is None:
        from scripts.search_listings import search_listings
        search_function = search_listings
    try:
        result = search_function(query, limit=limit, parser=checker.parser)
    except (mysql.connector.Error, ConnectionError, TimeoutError, OSError):
        return {"answerable": False, "reason": "database_unavailable", "message": "The database is unavailable. Check Docker/MySQL and retry.", "results": []}
    rows = result[0] if isinstance(result, tuple) else result
    after = checker.assess_results(rows)
    if not after["answerable"]:
        return {**after, "filters": before["filters"], "results": []}
    summarizer = summarizer or ListingSummarizer()
    return {**after, "filters": before["filters"], "results": [summarizer.summarize_listing(row) for row in rows.to_dict("records")]}
