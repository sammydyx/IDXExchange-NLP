"""FAISS-backed semantic retrieval for real-estate listing remarks."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

# On this macOS environment, PyTorch must initialize before FAISS.
# Reversing this order can cause an OpenMP segmentation fault during encoding.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # Keep module importable for environments that only run BM25/tests.
    SentenceTransformer = None

import faiss
import numpy as np
import pandas as pd


class SemanticSearcher:
    """Embed a query and retrieve cosine-similar listings from a FAISS index."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model
        self.index: faiss.Index | None = None
        self.metadata: pd.DataFrame | None = None

    def _load_model(self) -> Any:
        if self.model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is required. Run: pip install sentence-transformers"
                )
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.model

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray, listing_count: int) -> np.ndarray:
        values = np.ascontiguousarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        if values.shape[0] != listing_count:
            raise ValueError("embedding count must equal metadata row count")
        if not np.isfinite(values).all():
            raise ValueError("embeddings must contain only finite values")

        # Avoid faiss.normalize_L2 here: its macOS AVX2 wrapper can segfault on
        # the full corpus. NumPy produces the same unit vectors for cosine search.
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(values / norms, dtype=np.float32)

    def build_index(self, metadata: pd.DataFrame, embeddings: np.ndarray) -> None:
        if metadata.empty:
            raise ValueError("metadata must not be empty")
        vectors = self._validate_embeddings(embeddings.copy(), len(metadata))
        index = faiss.IndexFlatIP(vectors.shape[1])  # normalized inner product = cosine similarity
        index.add(vectors)
        self.index = index
        self.metadata = metadata.reset_index(drop=True).copy()

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        if self.index is None or self.metadata is None:
            raise RuntimeError("Build or load an index before saving it")
        index_path, metadata_path = Path(index_path), Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        self.metadata.to_csv(metadata_path, index=False)

    @classmethod
    def load(
        cls, index_path: str | Path, metadata_path: str | Path, model: Any | None = None
    ) -> "SemanticSearcher":
        searcher = cls(model=model)
        searcher.index = faiss.read_index(str(index_path))
        searcher.metadata = pd.read_csv(metadata_path)
        if searcher.index.ntotal != len(searcher.metadata):
            raise ValueError("FAISS index and metadata have different listing counts")
        return searcher

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.index is None or self.metadata is None:
            raise RuntimeError("Build or load an index before searching")

        model = self._load_model()
        query_vector = self._validate_embeddings(
            model.encode([query], normalize_embeddings=True), listing_count=1
        )
        top_k = min(top_k, len(self.metadata))
        scores, positions = self.index.search(query_vector, top_k)
        valid = positions[0] >= 0
        results = self.metadata.iloc[positions[0][valid]].copy().reset_index(drop=True)
        results.insert(0, "rank", range(1, len(results) + 1))
        results["similarity_score"] = scores[0][valid]
        return results

    def benchmark(self, queries: list[str], top_k: int = 10) -> dict[str, float]:
        if not queries:
            raise ValueError("queries must not be empty")
        timings = []
        for query in queries:
            start = time.perf_counter()
            self.search(query, top_k=top_k)
            timings.append((time.perf_counter() - start) * 1000)
        return {
            "query_count": len(timings),
            "average_ms": round(float(np.mean(timings)), 2),
            "median_ms": round(float(np.median(timings)), 2),
            "p95_ms": round(float(np.percentile(timings, 95)), 2),
        }
