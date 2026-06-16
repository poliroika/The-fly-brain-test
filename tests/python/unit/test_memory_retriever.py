"""Unit tests for `EpisodicMemory`, `VectorMemory`, and `BM25Retriever`."""

from __future__ import annotations

import numpy as np

from flybrain.runtime.memory import EpisodicMemory, VectorMemory
from flybrain.runtime.retriever import BM25Retriever

# ---------------------------------------------------------------- episodic


def test_episodic_append_and_lookup() -> None:
    m = EpisodicMemory()
    m.append("plan", "step 1", step_id=0)
    m.append("plan", "step 2", step_id=1)
    m.append("code", "x = 1", step_id=2)
    assert len(m) == 3
    assert [e.content for e in m.by_tag("plan")] == ["step 1", "step 2"]
    assert m.latest("plan").content == "step 2"
    assert m.latest("ghost") is None


# ---------------------------------------------------------------- vector


def test_vector_search_returns_nearest_neighbour() -> None:
    m = VectorMemory()
    m.add("a", np.array([1.0, 0.0, 0.0]), payload="a-doc")
    m.add("b", np.array([0.9, 0.1, 0.0]), payload="b-doc")
    m.add("c", np.array([0.0, 1.0, 0.0]), payload="c-doc")
    results = m.search(np.array([1.0, 0.0, 0.0]), k=2)
    assert results[0][1].key == "a"
    assert results[1][1].key == "b"


def test_vector_search_handles_empty_store() -> None:
    m = VectorMemory()
    assert m.search(np.array([1.0, 0.0]), k=3) == []


def test_vector_rejects_non_1d_inputs() -> None:
    import pytest

    m = VectorMemory()
    with pytest.raises(ValueError):
        m.add("bad", np.zeros((2, 3)))


# ---------------------------------------------------------------- BM25


def test_bm25_orders_by_relevance() -> None:
    r = BM25Retriever(
        corpus=[
            "The quick brown fox jumps over the lazy dog.",
            "Drosophila connectome graph compression methods.",
            "Functional graph priors for multi-agent systems.",
        ]
    )
    top = r.top_k("graph compression", k=2)
    # The connectome doc must rank first, fox doc must not be in top-2.
    assert top[0][1] == 1
    assert all(idx != 0 for _, idx, _ in top)


def test_bm25_empty_corpus_returns_no_results() -> None:
    r = BM25Retriever()
    assert r.top_k("anything") == []


def test_bm25_handles_empty_query() -> None:
    r = BM25Retriever(corpus=["a b c"])
    assert r.top_k("") == [(0.0, 0, "a b c")]
