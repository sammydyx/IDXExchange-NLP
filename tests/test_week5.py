from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bm25_search import BM25Searcher, tokenize
from scripts.semantic_search import SemanticSearcher


class FakeModel:
    vectors = {
        "modern kitchen": [1.0, 0.0],
        "quiet yard": [0.0, 1.0],
        "pool": [0.5, 0.5],
    }

    def encode(self, queries, normalize_embeddings=True):
        return np.asarray([self.vectors[query] for query in queries], dtype=np.float32)


@pytest.fixture()
def metadata():
    return pd.DataFrame({
        "listing_id": [101, 102, 103],
        "address": ["1 A St", "2 B St", "3 C St"],
        "remarks_cleaned": ["modern chef kitchen", "quiet private yard", "pool and patio"],
    })


@pytest.fixture()
def semantic(metadata):
    searcher = SemanticSearcher(model=FakeModel())
    searcher.build_index(metadata, np.asarray([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32))
    return searcher


def test_semantic_search_returns_ranked_results(semantic):
    results = semantic.search("modern kitchen", top_k=2)
    assert results["listing_id"].tolist()[0] == 101
    assert results["rank"].tolist() == [1, 2]
    assert results["similarity_score"].is_monotonic_decreasing


def test_semantic_search_limits_top_k_to_corpus_size(semantic):
    assert len(semantic.search("quiet yard", top_k=99)) == 3


@pytest.mark.parametrize("query", ["", "   ", None])
def test_semantic_search_rejects_empty_query(semantic, query):
    with pytest.raises(ValueError, match="non-empty"):
        semantic.search(query)


def test_semantic_search_rejects_invalid_top_k(semantic):
    with pytest.raises(ValueError, match="positive"):
        semantic.search("pool", top_k=0)


def test_build_index_rejects_embedding_count_mismatch(metadata):
    with pytest.raises(ValueError, match="embedding count"):
        SemanticSearcher(model=FakeModel()).build_index(metadata, np.asarray([[1, 0]], dtype=np.float32))


def test_bm25_returns_keyword_ranked_results(metadata):
    results = BM25Searcher(metadata).search("chef kitchen", top_k=2)
    assert results["listing_id"].tolist()[0] == 101
    assert "bm25_score" in results


def test_tokenize_normalizes_case_and_punctuation():
    assert tokenize("Chef's Kitchen, 2-Bed!") == ["chef", "s", "kitchen", "2", "bed"]


@pytest.mark.parametrize("query", ["", "  "])
def test_bm25_rejects_empty_query(metadata, query):
    with pytest.raises(ValueError, match="non-empty"):
        BM25Searcher(metadata).search(query)
