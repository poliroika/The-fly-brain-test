"""Unit tests for the Phase-4 embedding-client layer.

Covers:

* `MockEmbeddingClient` — determinism, mode separation, unit-norm,
  configurable dim;
* `EmbeddingCache` — round-trip + miss + size + dim mismatch;
* `cache_key` — stability + mode separation.
"""

from __future__ import annotations

import asyncio

import numpy as np

from flybrain.embeddings.base import EmbeddingMode, EmbeddingResponse
from flybrain.embeddings.cache import EmbeddingCache, cache_key
from flybrain.embeddings.mock_client import MockEmbeddingClient


def _embed(client: MockEmbeddingClient, text: str, mode: EmbeddingMode) -> EmbeddingResponse:
    return asyncio.run(client.embed(text, mode=mode))


def test_mock_client_returns_correct_dim_and_dtype() -> None:
    client = MockEmbeddingClient(output_dim=64)
    resp = _embed(client, "hello world", EmbeddingMode.DOC)
    assert resp.dim == 64
    assert resp.vector.shape == (64,)
    assert resp.vector.dtype == np.float32
    assert client.dim == 64


def test_mock_client_is_deterministic() -> None:
    client = MockEmbeddingClient(output_dim=32)
    a = _embed(client, "the quick brown fox", EmbeddingMode.DOC)
    b = _embed(client, "the quick brown fox", EmbeddingMode.DOC)
    np.testing.assert_array_equal(a.vector, b.vector)


def test_mock_client_doc_and_query_differ() -> None:
    client = MockEmbeddingClient(output_dim=32)
    doc = _embed(client, "same text", EmbeddingMode.DOC)
    qry = _embed(client, "same text", EmbeddingMode.QUERY)
    assert not np.allclose(doc.vector, qry.vector)
    # Different mode tags propagated into the model field.
    assert doc.model.endswith("doc")
    assert qry.model.endswith("query")


def test_mock_client_vectors_are_unit_norm() -> None:
    client = MockEmbeddingClient(output_dim=128)
    resp = _embed(client, "another text", EmbeddingMode.DOC)
    assert abs(float(np.linalg.norm(resp.vector)) - 1.0) < 1e-5


def test_mock_client_embed_many_runs_each_input_through_embed() -> None:
    client = MockEmbeddingClient(output_dim=16)
    out = asyncio.run(client.embed_many(["a", "b", "c"], mode=EmbeddingMode.DOC))
    assert len(out) == 3
    assert all(r.vector.shape == (16,) for r in out)
    # Different inputs → different vectors.
    assert not np.allclose(out[0].vector, out[1].vector)


def test_cache_key_is_stable_and_mode_aware() -> None:
    k1 = cache_key("model-a", EmbeddingMode.DOC, "hello")
    k2 = cache_key("model-a", EmbeddingMode.DOC, "hello")
    k3 = cache_key("model-a", EmbeddingMode.QUERY, "hello")
    k4 = cache_key("model-b", EmbeddingMode.DOC, "hello")
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4


def test_cache_round_trips_vector(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    response = EmbeddingResponse(
        vector=vec.copy(),
        dim=4,
        model="test-model",
        tokens=2,
        latency_ms=0,
        cost_rub=0.0,
        cached=False,
    )
    key = cache_key("test-model", EmbeddingMode.DOC, "abc")
    assert cache.get(key) is None
    cache.put(key, EmbeddingMode.DOC, response)
    hit = cache.get(key)
    assert hit is not None
    assert hit.cached is True
    np.testing.assert_array_equal(hit.vector, vec)
    assert hit.dim == 4
    assert cache.size() == 1


def test_cache_returns_none_on_dim_mismatch(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    vec = np.zeros(8, dtype=np.float32)
    response = EmbeddingResponse(vector=vec, dim=8, model="m")
    key = cache_key("m", EmbeddingMode.DOC, "x")
    cache.put(key, EmbeddingMode.DOC, response)
    # Tamper: mark the row as if it had dim=999 to simulate schema drift.
    with cache._connect() as conn:
        conn.execute("UPDATE emb_cache SET dim = ? WHERE cache_key = ?", (999, key))
    assert cache.get(key) is None
