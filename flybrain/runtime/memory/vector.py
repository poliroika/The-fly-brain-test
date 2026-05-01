"""Trivial in-memory vector store.

Embeddings are numpy arrays; "search" is a brute-force cosine ranking.
This is intentional: Phase 2 needs a *runtime hook* the controller can
invoke (`call_memory`), not a production vector DB. Plug-in indices land
later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class VectorEntry:
    key: str
    vector: np.ndarray
    payload: Any


@dataclass(slots=True)
class VectorMemory:
    entries: list[VectorEntry] = field(default_factory=list)

    def add(self, key: str, vector: np.ndarray, payload: Any = None) -> None:
        v = np.asarray(vector, dtype=np.float32)
        if v.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {v.shape}")
        self.entries.append(VectorEntry(key=key, vector=v, payload=payload))

    def search(self, query: np.ndarray, k: int = 3) -> list[tuple[float, VectorEntry]]:
        if not self.entries:
            return []
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {q.shape}")
        qn = q / (np.linalg.norm(q) + 1e-9)
        scored = []
        for e in self.entries:
            v = e.vector
            sim = float(np.dot(qn, v / (np.linalg.norm(v) + 1e-9)))
            scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self.entries)
