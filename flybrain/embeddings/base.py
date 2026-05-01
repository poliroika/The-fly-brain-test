"""Embedding-client abstraction shared by `task_emb`, `agent_emb`, etc.

Phase 4 introduces this layer for the same reasons we introduced
`flybrain.llm.base.LLMClient` in Phase 0: keep a stable async interface
that the rest of the codebase can talk to, with concrete clients
(`MockEmbeddingClient`, `YandexEmbeddingClient`) plugging in behind it.

Two operating modes match the Yandex AI Studio embedding surface:

* ``EmbeddingMode.DOC``  → `text-search-doc/latest` (used to embed
  long-lived role descriptions / documents).
* ``EmbeddingMode.QUERY`` → `text-search-query/latest` (used to embed
  the user task at the start of every run).

Both modes return fixed-dimensional ``np.float32`` vectors. Concrete
clients SHOULD honour ``EmbeddingClient.dim`` so callers can preallocate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class EmbeddingMode(str, Enum):
    """Which Yandex embedding endpoint to mimic.

    Hash-based mocks ignore the mode but use it as part of the cache key
    so the doc/query halves stay isolated even when the same text is
    embedded on both sides.
    """

    DOC = "doc"
    QUERY = "query"


@dataclass(slots=True)
class EmbeddingResponse:
    """A single embedding result.

    `cached=True` indicates the vector came from the SQLite cache (see
    `flybrain.embeddings.cache.EmbeddingCache`); the latency / cost
    fields are still meaningful for accounting even when we hit the
    cache.
    """

    vector: np.ndarray
    """Float32 1-D array of length `dim`."""

    dim: int
    model: str = ""
    tokens: int = 0
    latency_ms: int = 0
    cost_rub: float = 0.0
    cached: bool = False
    raw: dict | None = field(default=None)


class EmbeddingClient(ABC):
    """Async-capable text-embedding interface."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output dimensionality. Stable for the lifetime of the client."""

    @abstractmethod
    async def embed(
        self,
        text: str,
        *,
        mode: EmbeddingMode = EmbeddingMode.DOC,
    ) -> EmbeddingResponse:
        """Return the embedding of `text` under the given `mode`."""

    async def embed_many(
        self,
        texts: list[str],
        *,
        mode: EmbeddingMode = EmbeddingMode.DOC,
    ) -> list[EmbeddingResponse]:
        """Default sequential implementation. Concrete clients may override
        to batch / parallelise; controllers SHOULD use this entry point so
        they can benefit transparently from faster backends."""
        out: list[EmbeddingResponse] = []
        for t in texts:
            out.append(await self.embed(t, mode=mode))
        return out

    @abstractmethod
    async def aclose(self) -> None:
        """Release connections / flush caches."""
