"""SQLite-backed cache for text embeddings.

Mirrors the design of [`flybrain.llm.cache.SQLiteCache`](../llm/cache.py)
but stores the dense float32 vector as a raw blob instead of a JSON
payload. Vectors are typically 256–1024 floats, so JSON would balloon
the file 4–6× for no gain.

Cache key = `sha256(model || mode || text)`.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from flybrain.embeddings.base import EmbeddingMode, EmbeddingResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS emb_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    mode TEXT NOT NULL,
    dim INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    cost_rub REAL NOT NULL,
    vector BLOB NOT NULL,
    created_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS emb_cache_model_idx ON emb_cache(model);
"""


def cache_key(model: str, mode: EmbeddingMode, text: str) -> str:
    """Stable key. `mode` is part of the key because doc/query embeddings
    of identical text are, by design, *not* interchangeable on the
    Yandex side."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(mode.value.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbeddingCache:
    """Thread-safe SQLite cache for embedding vectors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    def get(self, key: str) -> EmbeddingResponse | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT model, mode, dim, tokens, cost_rub, vector "
                "FROM emb_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        model, _mode, dim, tokens, cost_rub, blob = row
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        if vec.size != dim:
            # Schema drift from an older cache; treat as miss.
            return None
        return EmbeddingResponse(
            vector=vec,
            dim=int(dim),
            model=model,
            tokens=int(tokens),
            latency_ms=0,
            cost_rub=float(cost_rub),
            cached=True,
            raw=None,
        )

    def put(self, key: str, mode: EmbeddingMode, response: EmbeddingResponse) -> None:
        import time

        vec = np.ascontiguousarray(response.vector, dtype=np.float32)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO emb_cache "
                "(cache_key, model, mode, dim, tokens, cost_rub, vector, created_unix_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    response.model,
                    mode.value,
                    response.dim,
                    response.tokens,
                    response.cost_rub,
                    vec.tobytes(),
                    int(time.time() * 1000),
                ),
            )

    def size(self) -> int:
        with self._lock, self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM emb_cache").fetchone()
        return int(n)
