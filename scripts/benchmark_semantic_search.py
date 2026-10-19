"""Benchmark query embedding plus FAISS retrieval on the local listing index."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_search import SemanticSearcher

QUERIES = [
    "quiet family home", "modern kitchen", "private backyard", "ocean view condo",
    "move in ready property", "home office", "pool and spa", "near restaurants",
    "newly renovated home", "resort style amenities",
]


def main() -> None:
    models = ROOT / "data/models"
    searcher = SemanticSearcher.load(models / "listings.faiss", models / "listing_metadata.csv")
    # Warm up model loading and the first inference; latency excludes one-time setup.
    searcher.search(QUERIES[0], top_k=10)
    metrics = searcher.benchmark(QUERIES, top_k=10)
    metrics["listing_count"] = int(searcher.index.ntotal)
    metrics["target_under_100ms"] = metrics["median_ms"] < 100
    output = ROOT / "data/processed/week5_latency_report.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
