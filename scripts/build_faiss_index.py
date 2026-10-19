"""Build and save a FAISS index from Week 5 listing embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_search import SemanticSearcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default=ROOT / "data/models/listing_embeddings.npy")
    parser.add_argument("--metadata", default=ROOT / "data/models/listing_metadata.csv")
    parser.add_argument("--index-output", default=ROOT / "data/models/listings.faiss")
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata)
    embeddings = np.load(args.embeddings)
    searcher = SemanticSearcher()
    searcher.build_index(metadata, embeddings)
    searcher.save(args.index_output, args.metadata)
    print(f"Saved FAISS index with {searcher.index.ntotal:,} listings to {args.index_output}")


if __name__ == "__main__":
    main()
