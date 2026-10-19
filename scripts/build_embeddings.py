"""Create normalized MiniLM embeddings for a cleaned listing corpus."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=ROOT / "data/processed/semantic_listings.csv")
    parser.add_argument("--output-dir", default=ROOT / "data/models")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Install sentence-transformers before creating embeddings.") from error

    listings = pd.read_csv(args.input)
    if listings.empty or "remarks_cleaned" not in listings:
        raise SystemExit("Input must contain one or more cleaned listing remarks.")
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        listings["remarks_cleaned"].fillna("").tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "listing_embeddings.npy", embeddings)
    listings.to_csv(output_dir / "listing_metadata.csv", index=False)
    print(f"Saved {embeddings.shape[0]:,} embeddings with {embeddings.shape[1]} dimensions")


if __name__ == "__main__":
    main()
