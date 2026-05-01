"""Task / agent / trace / graph / fly-connectome embeddings.

Phase 4 of `PLAN.md`. Re-exports the public surface so importing
``flybrain.embeddings`` brings the whole stack into scope.

The clients (mock + Yandex) live behind a small ABC mirroring the
shape of `flybrain.llm.base.LLMClient`; the wrappers
(``TaskEmbedder``, ``AgentEmbedder``, ``TraceEmbedder``,
``AgentGraphEmbedder``, ``FlyGraphEmbedder``) plug in via that ABC
so we can swap the deterministic mock for the real Yandex API
without touching the controller code.

The Phase-4 exit deliverable is :class:`ControllerStateBuilder`,
which assembles a :class:`ControllerState` from a
``flybrain.runtime.state.RuntimeState`` in <50 ms on CPU. The
Phase-5 GNN / RNN / learned-router controllers will consume this
state.
"""

from flybrain.embeddings.agent_emb import AgentEmbedder
from flybrain.embeddings.base import (
    EmbeddingClient,
    EmbeddingMode,
    EmbeddingResponse,
)
from flybrain.embeddings.cache import EmbeddingCache, cache_key
from flybrain.embeddings.fly_emb import FlyGraphEmbedder
from flybrain.embeddings.graph_emb import AgentGraphEmbedder
from flybrain.embeddings.mock_client import MockEmbeddingClient
from flybrain.embeddings.state import ControllerState, ControllerStateBuilder
from flybrain.embeddings.task_emb import TaskEmbedder
from flybrain.embeddings.trace_emb import TraceEmbedder, extract_features
from flybrain.embeddings.yandex_client import YandexEmbeddingClient, YandexEmbeddingConfig

__all__ = [
    "AgentEmbedder",
    "AgentGraphEmbedder",
    "ControllerState",
    "ControllerStateBuilder",
    "EmbeddingCache",
    "EmbeddingClient",
    "EmbeddingMode",
    "EmbeddingResponse",
    "FlyGraphEmbedder",
    "MockEmbeddingClient",
    "TaskEmbedder",
    "TraceEmbedder",
    "YandexEmbeddingClient",
    "YandexEmbeddingConfig",
    "cache_key",
    "extract_features",
]
