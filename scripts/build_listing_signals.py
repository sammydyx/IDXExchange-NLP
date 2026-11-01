"""Build a Week 6 JSONL signal file from every row in ``rets_property``."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import mysql.connector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.signal_extraction import SignalExtractor


SQL = """
    SELECT L_ListingID AS listing_id, L_City AS city,
           L_SystemPrice AS listed_price, L_Keyword2 AS listed_bedrooms,
           LM_Dec_3 AS listed_bathrooms, LM_Int2_3 AS listed_square_feet,
           L_Type_ AS property_type, L_Remarks AS remarks
    FROM rets_property
    ORDER BY id
"""

ENTITY_REFRESH_SQL = """
    SELECT L_ListingID AS listing_id, L_Remarks AS remarks
    FROM rets_property
    ORDER BY id
"""


def connect():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
        database=os.getenv("MYSQL_DATABASE", "real_estate"),
    )


def build_signals(
    output_path: Path,
    batch_size: int = 500,
    limit: int | None = None,
    offset: int = 0,
    append: bool = False,
) -> dict[str, int]:
    """Stream database listings into JSONL without holding the corpus in memory.

    ``offset`` and ``append`` allow a long full-table build to resume safely
    after an interrupted local process. They are intended for batch recovery,
    not for normal first runs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = SignalExtractor()
    counts = {"listings": 0, "remarks_with_text": 0, "amenity_matches": 0, "keyword_matches": 0}
    connection = connect()
    try:
        cursor = connection.cursor(dictionary=True)
        try:
            if limit is not None:
                statement = SQL + " LIMIT %s OFFSET %s"
                parameters = (limit, offset)
            elif offset:
                statement = SQL + " LIMIT 18446744073709551615 OFFSET %s"
                parameters = (offset,)
            else:
                statement = SQL
                parameters = ()
            cursor.execute(statement, parameters)
            with output_path.open("a" if append else "w", encoding="utf-8") as output:
                while rows := cursor.fetchmany(batch_size):
                    for row in rows:
                        signal = extractor.extract_signals(row)
                        output.write(json.dumps(signal, ensure_ascii=False) + "\n")
                        counts["listings"] += 1
                        counts["remarks_with_text"] += int(bool(str(row.get("remarks") or "").strip()))
                        counts["amenity_matches"] += len(signal["amenities"])
                        counts["keyword_matches"] += sum(len(values) for values in signal["keywords"].values())
        finally:
            cursor.close()
    finally:
        connection.close()
    return counts


def refresh_entities(output_path: Path, batch_size: int = 500) -> dict[str, int]:
    """Refresh only derived numeric entities in an existing complete JSONL file.

    This keeps an already-built taxonomy output intact after Week 3 entity
    rules change. Source rows and JSONL rows are both ordered by database id;
    their listing IDs are checked before replacing the completed file.
    """
    if not output_path.is_file():
        raise FileNotFoundError(f"Cannot refresh a missing signal file: {output_path}")

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    extractor = SignalExtractor()
    counts = {"listings": 0, "entity_mentions": 0}
    connection = connect()
    try:
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(ENTITY_REFRESH_SQL)
            with output_path.open(encoding="utf-8") as source, temporary_path.open("w", encoding="utf-8") as destination:
                while rows := cursor.fetchmany(batch_size):
                    for row in rows:
                        try:
                            signal = json.loads(next(source))
                        except StopIteration as error:
                            raise ValueError("Signal file has fewer rows than rets_property") from error
                        if str(row["listing_id"]) != str(signal["listing_id"]):
                            raise ValueError("Signal file order does not match rets_property; rebuild instead of refreshing")
                        remarks = row.get("remarks") or ""
                        signal["entities"] = {
                            "bedrooms": extractor.entity_extractor.extract_bedrooms(remarks),
                            "bathrooms": extractor.entity_extractor.extract_bathrooms(remarks),
                            "square_feet": extractor.entity_extractor.extract_square_feet(remarks),
                            "price": extractor.entity_extractor.extract_price(remarks),
                        }
                        destination.write(json.dumps(signal, ensure_ascii=False) + "\n")
                        counts["listings"] += 1
                        counts["entity_mentions"] += sum(value is not None for value in signal["entities"].values())
                if next(source, None) is not None:
                    raise ValueError("Signal file has more rows than rets_property")
        finally:
            cursor.close()
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    temporary_path.replace(output_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Week 6 listing signals as JSONL.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/listing_signals.jsonl")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="Optional development limit; omit for every listing.")
    parser.add_argument("--offset", type=int, default=0, help="Rows to skip; use only when resuming an interrupted build.")
    parser.add_argument("--append", action="store_true", help="Append output instead of replacing it; use with --offset.")
    parser.add_argument("--refresh-entities", action="store_true", help="Update numeric entities in an existing complete JSONL file.")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1) or args.offset < 0:
        parser.error("--batch-size, --limit, and --offset must be non-negative; batch-size and limit must be positive")
    if args.append and not args.offset:
        parser.error("--append requires a positive --offset so output is not duplicated")
    if args.refresh_entities:
        if args.limit is not None or args.offset or args.append:
            parser.error("--refresh-entities cannot be combined with --limit, --offset, or --append")
        counts = refresh_entities(args.output, args.batch_size)
        print(f"Refreshed entities in {counts['listings']:,} listing signals at {args.output}")
        print(f"Structured entity mentions: {counts['entity_mentions']:,}")
        return
    counts = build_signals(args.output, args.batch_size, args.limit, args.offset, args.append)
    print(f"Saved {counts['listings']:,} listing signals to {args.output}")
    print(f"Remarks with text: {counts['remarks_with_text']:,}")
    print(f"Amenity matches: {counts['amenity_matches']:,}; other keyword matches: {counts['keyword_matches']:,}")


if __name__ == "__main__":
    main()
