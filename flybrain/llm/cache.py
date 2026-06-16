"""SQLite-backed LLM response cache.

The cache key is `sha256(model + temperature + messages_json)`. We store the
raw response payload so retries reproduce identical outputs without a real
LLM call. This is critical given the 2000 ₽ total budget.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from flybrain.llm.base import LLMResponse, Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    payload TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_rub REAL NOT NULL,
    created_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_cache_model_idx ON llm_cache(model);
"""


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"unserializable type: {type(obj).__name__}")


def cache_key(model: str, temperature: float, messages: list[Message]) -> str:
    payload = json.dumps(
        {
            "model": model,
            "temperature": round(temperature, 4),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SQLiteCache:
    """Thread-safe SQLite cache. Multiple processes can share the same file."""

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

    def get(self, key: str) -> LLMResponse | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload, tokens_in, tokens_out, cost_rub FROM llm_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        payload, tokens_in, tokens_out, cost_rub = row
        data = json.loads(payload)
        return LLMResponse(
            content=data["content"],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=data.get("latency_ms", 0),
            cost_rub=cost_rub,
            model=data.get("model", ""),
            cached=True,
            raw=data.get("raw"),
        )

    def put(self, key: str, response: LLMResponse) -> None:
        payload = json.dumps(
            {
                "content": response.content,
                "latency_ms": response.latency_ms,
                "model": response.model,
                "raw": response.raw,
            },
            ensure_ascii=False,
        )
        import time

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(cache_key, model, payload, tokens_in, tokens_out, cost_rub, created_unix_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    response.model,
                    payload,
                    response.tokens_in,
                    response.tokens_out,
                    response.cost_rub,
                    int(time.time() * 1000),
                ),
            )

    def size(self) -> int:
        with self._lock, self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return int(n)
