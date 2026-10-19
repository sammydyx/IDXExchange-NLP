"""Create a balanced semantic-vs-BM25 relevance review template."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bm25_search import BM25Searcher
from scripts.semantic_search import SemanticSearcher

DEFAULT_QUERIES = [
    "quiet family friendly home with outdoor space",
    "modern chef kitchen",
    "home near restaurants and shopping",
    "move in ready property",
    "ocean view condo",
    "private backyard",
    "home office",
    "resort style amenities",
    "newly renovated home",
    "property with a pool",
]


def existing_annotations(output: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    """Preserve prior reviews when regenerating the same query-result rows."""
    if not output.exists():
        return {}

    previous = pd.read_csv(output, dtype={"listing_id": str}).fillna("")
    required = {"query", "search_method", "listing_id", "relevant", "notes"}
    if not required.issubset(previous.columns):
        return {}

    annotations = {}
    for row in previous.itertuples(index=False):
        key = (str(row.query), str(row.search_method), int(str(row.listing_id)))
        annotations[key] = {
            "relevant": str(row.relevant),
            "notes": str(row.notes),
            "annotation_source": str(getattr(row, "annotation_source", "")),
        }
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=ROOT / "data/models/listing_metadata.csv")
    parser.add_argument("--index", default=ROOT / "data/models/listings.faiss")
    parser.add_argument("--output", default=ROOT / "data/processed/week5_relevance_review.csv")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Results per query for each method (default: 5).")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be a positive integer")

    output = Path(args.output)
    annotations = existing_annotations(output)
    metadata = pd.read_csv(args.metadata)
    semantic = SemanticSearcher.load(args.index, args.metadata)
    bm25 = BM25Searcher(metadata)
    rows = []
    for query in DEFAULT_QUERIES:
        for method, results, score_column in (
            ("semantic", semantic.search(query, top_k=args.top_k), "similarity_score"),
            ("bm25", bm25.search(query, top_k=args.top_k), "bm25_score"),
        ):
            for row in results.itertuples(index=False):
                remarks = str(getattr(row, "remarks_cleaned", ""))
                annotation = annotations.get(
                    (query, method, int(getattr(row, "listing_id"))), {}
                )
                rows.append({
                    "query": query, "search_method": method, "rank": row.rank,
                    "listing_id": getattr(row, "listing_id"), "address": getattr(row, "address", ""),
                    "city": getattr(row, "city", ""), "price": getattr(row, "price", ""),
                    "score": round(float(getattr(row, score_column)), 4),
                    "remarks_preview": remarks[:300],
                    "relevant": annotation.get("relevant", ""),
                    "notes": annotation.get("notes", ""),
                    "annotation_source": annotation.get("annotation_source", ""),
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    labelled = sum(row["relevant"] in {"yes", "no"} for row in rows)
    print(f"Saved {len(rows)} query-result pairs to {output} ({labelled} preserved labels)")


if __name__ == "__main__":
    main()
