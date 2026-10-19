"""Export and clean listing remarks from the Docker MySQL database."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import mysql.connector
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.text_cleaning import TextCleaner


def export_corpus(limit: int | None = None) -> pd.DataFrame:
    sql = """
        SELECT L_ListingID AS listing_id, L_Address AS address, L_City AS city,
               L_SystemPrice AS price, L_Keyword2 AS beds, LM_Dec_3 AS baths,
               L_Remarks AS remarks
        FROM rets_property
        WHERE L_Remarks IS NOT NULL AND CHAR_LENGTH(TRIM(L_Remarks)) > 50
        ORDER BY L_ListingID
    """
    params: tuple[int, ...] = ()
    if limit is not None:
        sql += " LIMIT %s"
        params = (limit,)
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
    dataframe = pd.DataFrame(rows)
    cleaner = TextCleaner()
    dataframe["remarks_cleaned"] = dataframe["remarks"].map(cleaner.clean_text)
    return dataframe[dataframe["remarks_cleaned"].str.len() > 0].drop_duplicates("listing_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Use 10,000 for a quick development corpus")
    parser.add_argument("--output", default=ROOT / "data/processed/semantic_listings.csv")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    corpus = export_corpus(args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(output, index=False)
    print(f"Saved {len(corpus):,} cleaned listings to {output}")


if __name__ == "__main__":
    main()
