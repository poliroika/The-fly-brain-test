"""`TraceEmbedder` — pool the run-trace into a fixed-length vector.

Per `PLAN.md` Phase 4 §572: pooling over step-embeddings + handcrafted
features. For Phase 4 the "step-embedding" of a step is the embedding
of its `output_summary` (the most informative free-form text we have
on the trace); the handcrafted features cover the structural side
(latency, tokens, errors, verifier scores).

The embedder works on either:

* a **finalised** trace dict (the shape `TraceWriter.finalize`
  returns), in which case we have access to `final_answer` and
  cumulative totals;

* a **partial** trace as observed by the runner mid-loop (a list of
  step dicts produced by `MAS._make_step`).

Both code paths share the same handcrafted-feature extractor so the
controller sees a consistent feature layout no matter when it asks.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.embeddings.base import EmbeddingClient, EmbeddingMode

_HANDCRAFTED_FEATURES = (
    "num_steps",
    "num_agent_runs",
    "num_tool_calls",
    "num_errors",
    "num_verifier_calls",
    "mean_verifier_score",
    "min_verifier_score",
    "total_tokens_in",
    "total_tokens_out",
    "total_cost_rub",
    "mean_latency_ms",
    "max_latency_ms",
    "graph_hash_changes",
)


def _safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, x))


def extract_features(steps: list[dict[str, Any]]) -> np.ndarray:
    """Compute the handcrafted feature vector for a list of trace steps.

    Returns a `(len(_HANDCRAFTED_FEATURES),)` float32 array. The order
    is fixed by `_HANDCRAFTED_FEATURES`; downstream code can inspect
    that constant if it needs feature names."""
    n = len(steps)
    if n == 0:
        return np.zeros(len(_HANDCRAFTED_FEATURES), dtype=np.float32)

    num_agent_runs = sum(
        1 for s in steps if (s.get("graph_action", {}) or {}).get("kind") == "run_agent"
    )
    num_tool_calls = sum(len(s.get("tool_calls") or []) for s in steps)
    num_errors = sum(len(s.get("errors") or []) for s in steps)

    verifier_scores = [
        float(s["verifier_score"]) for s in steps if s.get("verifier_score") is not None
    ]
    num_verifier_calls = len(verifier_scores)
    mean_v = float(np.mean(verifier_scores)) if verifier_scores else 0.0
    min_v = float(np.min(verifier_scores)) if verifier_scores else 0.0

    total_tokens_in = sum(int(s.get("tokens_in") or 0) for s in steps)
    total_tokens_out = sum(int(s.get("tokens_out") or 0) for s in steps)
    total_cost = sum(float(s.get("cost_rub") or 0.0) for s in steps)
    latencies = [int(s.get("latency_ms") or 0) for s in steps]
    mean_l = float(np.mean(latencies)) if latencies else 0.0
    max_l = float(np.max(latencies)) if latencies else 0.0

    seen_hashes: set[str] = set()
    hash_changes = 0
    for s in steps:
        h = s.get("current_graph_hash")
        if h is None:
            continue
        if h not in seen_hashes:
            seen_hashes.add(h)
            hash_changes += 1
    # First observation is the baseline, not a "change".
    hash_changes = max(0, hash_changes - 1)

    feats = np.array(
        [
            float(n),
            float(num_agent_runs),
            float(num_tool_calls),
            float(num_errors),
            float(num_verifier_calls),
            mean_v,
            min_v,
            _safe_log1p(total_tokens_in),
            _safe_log1p(total_tokens_out),
            _safe_log1p(total_cost),
            _safe_log1p(mean_l),
            _safe_log1p(max_l),
            float(hash_changes),
        ],
        dtype=np.float32,
    )
    assert feats.shape == (len(_HANDCRAFTED_FEATURES),)
    return feats


@dataclass(slots=True)
class TraceEmbedder:
    """Pool a trace into a single dense vector.

    Layout of the returned vector:

        [0 : dim)                       — mean-pooled step embedding
        [dim : dim + len(features))     — handcrafted features
    """

    client: EmbeddingClient
    pool_kind: str = "mean"
    """Currently only `"mean"` is implemented; `"attention"` lands with
    Phase 5 once the controller knows what to attend to."""

    _empty_vec: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._empty_vec = np.zeros(self.client.dim, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return len(_HANDCRAFTED_FEATURES)

    @property
    def output_dim(self) -> int:
        return self.client.dim + self.feature_dim

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _HANDCRAFTED_FEATURES

    async def _pool_step_texts(self, steps: list[dict[str, Any]]) -> np.ndarray:
        texts = [
            (s.get("output_summary") or "").strip()
            for s in steps
            if (s.get("output_summary") or "").strip()
        ]
        if not texts:
            return self._empty_vec.copy()
        responses = await self.client.embed_many(texts, mode=EmbeddingMode.DOC)
        stacked = np.stack([np.asarray(r.vector, dtype=np.float32) for r in responses], axis=0)
        if self.pool_kind == "mean":
            pooled = stacked.mean(axis=0)
        else:
            raise ValueError(f"unknown pool_kind: {self.pool_kind!r}")
        return pooled.astype(np.float32, copy=False)

    async def embed(self, steps: list[dict[str, Any]]) -> np.ndarray:
        pooled = await self._pool_step_texts(steps)
        feats = extract_features(steps)
        return np.concatenate([pooled, feats], axis=0).astype(np.float32, copy=False)

    def embed_sync(self, steps: list[dict[str, Any]]) -> np.ndarray:
        return asyncio.run(self.embed(steps))
