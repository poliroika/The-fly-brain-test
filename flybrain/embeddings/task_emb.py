"""`TaskEmbedder` — wraps an `EmbeddingClient` in QUERY mode.

Per `PLAN.md` Phase 4 §569–§571: the controller embeds the user task
once at the start of each run via `text-search-query`. This class is
the thin wrapper that does that, with optional caching baked in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.embeddings.base import EmbeddingClient, EmbeddingMode


@dataclass(slots=True)
class TaskEmbedder:
    client: EmbeddingClient

    @property
    def dim(self) -> int:
        return self.client.dim

    async def embed(self, prompt: str, *, task_type: str = "") -> np.ndarray:
        """Return the float32 embedding of `(task_type || prompt)`.

        Concatenating `task_type` is a cheap, deterministic way to keep
        the vector aware of the routing label without spending a second
        embedding call. For the mock client the seed includes the
        prefix; for Yandex the prefix becomes part of the prompt.
        """
        text = f"[{task_type}] {prompt}" if task_type else prompt
        resp = await self.client.embed(text, mode=EmbeddingMode.QUERY)
        return np.asarray(resp.vector, dtype=np.float32)
