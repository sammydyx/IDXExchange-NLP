"""Execute safe Week 4 property searches against the local MySQL database."""

from __future__ import annotations

import os

import mysql.connector
import pandas as pd

from scripts.query_parser import QueryFilters, QueryParser, SafeSQLGenerator


DISPLAY_COLUMNS = [
    "L_ListingID",
    "L_Address",
    "L_City",
    "beds",
    "baths",
    "price",
    "square_feet",
    "property_type",
]


def search_listings(
    query: str, *, limit: int = 10, parser: QueryParser | None = None
) -> tuple[pd.DataFrame, QueryFilters, str, list[object]]:
    """Parse a natural-language query and execute its parameterized SQL safely.

    The SQL text contains fixed schema names and ``%s`` placeholders only. Query
    values remain in ``params`` and are passed separately to MySQL Connector.
    """
    parser = parser or QueryParser()
    filters = parser.parse(query)
    sql, params = SafeSQLGenerator(parser).build(filters, limit=limit)

    connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
        database=os.getenv("MYSQL_DATABASE", "real_estate"),
    )
    try:
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        connection.close()

    results = pd.DataFrame(rows, columns=DISPLAY_COLUMNS)
    return results, filters, sql, params


if __name__ == "__main__":
    example = "Find 3+ bedroom homes in Irvine under $1M with a pool"
    results, filters, _, _ = search_listings(example, limit=5)
    print(filters.as_dict())
    print(results.to_string(index=False))
