"""Deterministic hash-based embedding client.

Used in unit tests + in any environment where the live Yandex API is
not reachable (CI runs do not have `YANDEX_API_KEY`). The vectors are:

* fully deterministic for a given (mode, text) pair,
* approximately unit-norm,
* mode-aware (doc vs query produce *different* vectors even for the
  same text — same as the real Yandex endpoints), and
* numerically stable across NumPy versions because we seed a
  ``numpy.random.Generator`` with a hash digest.

The vectors are not semantically meaningful. They exist so the rest
of the embedding stack (cache, controller state, GCN, downstream
training plumbing) can be wired up and tested without burning the
budget on real API calls.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from flybrain.embeddings.base import EmbeddingClient, EmbeddingMode, EmbeddingResponse


def _seed_from(text: str, mode: EmbeddingMode, model: str) -> int:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(mode.value.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    digest = h.digest()
    # Take the first 8 bytes as a uint64 seed; np.random.default_rng
    # accepts any non-negative int <= 2**64 - 1.
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass(slots=True)
class MockEmbeddingClient(EmbeddingClient):
    """Hash-seeded Gaussian embeddings, normalised to unit-norm."""

    output_dim: int = 256
    model: str = "mock/text-search"
    fixed_latency_ms: int = 0
    cost_per_call_rub: float = 0.0

    @property
    def dim(self) -> int:
        return self.output_dim

    async def embed(
        self,
        text: str,
        *,
        mode: EmbeddingMode = EmbeddingMode.DOC,
    ) -> EmbeddingResponse:
        if self.fixed_latency_ms:
            time.sleep(self.fixed_latency_ms / 1000.0)
        seed = _seed_from(text, mode, self.model)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.output_dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return EmbeddingResponse(
            vector=vec,
            dim=self.output_dim,
            model=f"{self.model}/{mode.value}",
            tokens=max(1, len(text) // 4),
            latency_ms=self.fixed_latency_ms,
            cost_rub=self.cost_per_call_rub,
            cached=False,
            raw=None,
        )

    async def aclose(self) -> None:  # pragma: no cover - nothing to release
        return None
