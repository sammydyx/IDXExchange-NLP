"""Keyword-search baseline used to compare against semantic retrieval."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi


def tokenize(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


class BM25Searcher:
    def __init__(self, metadata: pd.DataFrame, text_column: str = "remarks_cleaned") -> None:
        if metadata.empty:
            raise ValueError("metadata must not be empty")
        if text_column not in metadata.columns:
            raise ValueError(f"metadata must contain {text_column!r}")
        self.metadata = metadata.reset_index(drop=True).copy()
        self.text_column = text_column
        self.bm25 = BM25Okapi([tokenize(value) for value in self.metadata[text_column]])

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores = self.bm25.get_scores(tokenize(query))
        positions = sorted(range(len(scores)), key=lambda position: (-scores[position], position))[:top_k]
        results = self.metadata.iloc[positions].copy().reset_index(drop=True)
        results.insert(0, "rank", range(1, len(results) + 1))
        results["bm25_score"] = [float(scores[position]) for position in positions]
        return results
