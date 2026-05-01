"""`AgentEmbedder` — pre-computes a static embedding per `AgentSpec`.

Agent role descriptions only change on Phase boundaries (we ship one
canonical ``minimal_15`` / ``extended_25`` set per release), so we
embed every spec once at startup and look the vectors up by name. The
DOC mode mirrors what `flybrain.controller` will eventually use to
score agents against the task embedding.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from flybrain.embeddings.base import EmbeddingClient, EmbeddingMode
from flybrain.runtime.agent import AgentSpec


def _agent_text(spec: AgentSpec) -> str:
    """Canonical text representation used for embedding.

    We include the role and the system prompt because role alone is too
    sparse (multiple agents share roles like ``coder``) and the prompt
    alone hides the role label that the router needs."""
    parts = [
        f"name={spec.name}",
        f"role={spec.role}",
        f"tier={spec.model_tier}",
        f"prompt={spec.system_prompt}",
    ]
    if spec.tools:
        parts.append("tools=" + ",".join(sorted(spec.tools)))
    return " | ".join(parts)


@dataclass(slots=True)
class AgentEmbedder:
    client: EmbeddingClient
    _table: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def dim(self) -> int:
        return self.client.dim

    async def precompute(self, specs: list[AgentSpec]) -> None:
        """Embed every spec once and stash the vectors by `spec.name`."""
        texts = [_agent_text(s) for s in specs]
        responses = await self.client.embed_many(texts, mode=EmbeddingMode.DOC)
        self._table = {
            spec.name: np.asarray(resp.vector, dtype=np.float32)
            for spec, resp in zip(specs, responses, strict=True)
        }

    def precompute_sync(self, specs: list[AgentSpec]) -> None:
        """Convenience wrapper that drives `precompute` in a fresh event
        loop. Useful for unit tests and synchronous bootstrap code."""
        asyncio.run(self.precompute(specs))

    def get(self, name: str) -> np.ndarray:
        try:
            return self._table[name]
        except KeyError as e:
            raise KeyError(
                f"agent {name!r} has no precomputed embedding; "
                "call `AgentEmbedder.precompute(specs)` first"
            ) from e

    def stack(self, names: list[str]) -> np.ndarray:
        """Return a `(len(names), dim)` matrix in the requested order.

        Missing names get a zero row instead of raising; the controller
        treats zero-norm rows as "unknown agent" via the GCN message
        passing layer."""
        if not names:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[np.ndarray] = []
        for name in names:
            vec = self._table.get(name)
            if vec is None:
                rows.append(np.zeros(self.dim, dtype=np.float32))
            else:
                rows.append(vec)
        return np.stack(rows, axis=0)

    @property
    def known_names(self) -> list[str]:
        return sorted(self._table.keys())
