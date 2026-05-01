"""`ControllerState` — the bag of features the Phase-5 controller sees.

This is the Phase-4 exit deliverable. ``ControllerState.from_runtime``
must assemble the full feature bundle in **<50 ms on CPU** (per
`PLAN.md` §575). To keep the path that fast we:

1. Pre-embed agents *once* at MAS startup (``AgentEmbedder``),
2. Pre-embed the fly graph *once* at MAS startup (``FlyGraphEmbedder``),
3. Re-run only the cheap parts on every tick: the trace pooler (one
   embedding call per new-step output), the agent-graph GCN
   (numpy matmul over a tiny K×K matrix), and the handcrafted
   per-step features.

`from_runtime_sync` exists because the runner's tight loop is async —
controllers that only need the state at task boundaries can stay
synchronous and call the sync helper.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.embeddings.agent_emb import AgentEmbedder
from flybrain.embeddings.fly_emb import FlyGraphEmbedder
from flybrain.embeddings.graph_emb import AgentGraphEmbedder
from flybrain.embeddings.task_emb import TaskEmbedder
from flybrain.embeddings.trace_emb import TraceEmbedder
from flybrain.graph.dataclasses import FlyGraph
from flybrain.runtime.state import RuntimeState


@dataclass(slots=True)
class ControllerState:
    """Tensor-friendly snapshot for the Phase-5 controller.

    Field shapes (all float32 unless stated):
      - `task_vec`          : `(task_dim,)`
      - `agent_node_vecs`   : `(num_agents, agent_dim)`
      - `agent_graph_vec`   : `(graph_dim,)`
      - `agent_node_emb`    : `(num_agents, graph_out_dim)` after GCN
      - `trace_vec`         : `(trace_dim,)` (pooled steps + handcrafted)
      - `fly_vec`           : `(fly_dim,)` (graph-level fly prior)
      - `inbox_vec`         : `(num_agents,)` pending-message counts
      - `produced_mask`     : `(C,)` 0/1 over the runtime's component tags
      - `step_id`, `task_type`, `agent_names` : metadata mirrors of the
        underlying `RuntimeState` so the controller can index back into
        the runtime without holding a reference to it.
    """

    task_vec: np.ndarray
    agent_node_vecs: np.ndarray
    agent_graph_vec: np.ndarray
    agent_node_emb: np.ndarray
    trace_vec: np.ndarray
    fly_vec: np.ndarray
    inbox_vec: np.ndarray
    produced_mask: np.ndarray

    step_id: int
    task_type: str
    agent_names: list[str]
    component_tags: list[str]
    build_ms: float = 0.0

    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "task_vec": tuple(self.task_vec.shape),
            "agent_node_vecs": tuple(self.agent_node_vecs.shape),
            "agent_graph_vec": tuple(self.agent_graph_vec.shape),
            "agent_node_emb": tuple(self.agent_node_emb.shape),
            "trace_vec": tuple(self.trace_vec.shape),
            "fly_vec": tuple(self.fly_vec.shape),
            "inbox_vec": tuple(self.inbox_vec.shape),
            "produced_mask": tuple(self.produced_mask.shape),
        }


_DEFAULT_COMPONENT_TAGS: tuple[str, ...] = (
    "plan",
    "code",
    "tests_run",
    "final_answer",
    "tool_used",
    "verifier_called",
)


@dataclass(slots=True)
class ControllerStateBuilder:
    """Hold the per-MAS prebuilt artefacts (agent + fly embeddings) and
    assemble a `ControllerState` per tick."""

    task: TaskEmbedder
    agents: AgentEmbedder
    trace: TraceEmbedder
    fly: FlyGraphEmbedder
    agent_graph: AgentGraphEmbedder

    fly_graph: FlyGraph | None = None
    component_tags: tuple[str, ...] = _DEFAULT_COMPONENT_TAGS

    _fly_vec_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def _fly_vector(self) -> np.ndarray:
        if self.fly_graph is None:
            return np.zeros(self.fly.dim, dtype=np.float32)
        if self._fly_vec_cache is None:
            self._fly_vec_cache = self.fly.graph_vector(self.fly_graph)
        return self._fly_vec_cache

    def _produced_mask(self, produced: set[str]) -> np.ndarray:
        return np.array(
            [1.0 if tag in produced else 0.0 for tag in self.component_tags],
            dtype=np.float32,
        )

    def _inbox_vec(self, names: list[str], pending: dict[str, int]) -> np.ndarray:
        return np.array(
            [float(pending.get(n, 0)) for n in names],
            dtype=np.float32,
        )

    async def from_runtime(
        self,
        runtime: RuntimeState,
        *,
        agent_graph: dict[str, Any] | None = None,
        trace_steps: list[dict[str, Any]] | None = None,
    ) -> ControllerState:
        """Async assembly path. Suitable for the runner's main loop."""
        import time as _t

        t0 = _t.perf_counter()
        names = list(runtime.available_agents)

        # Cheap, deterministic Yandex-mocked task embedding.
        task_vec = await self.task.embed(runtime.prompt, task_type=runtime.task_type)

        agent_node_vecs = self.agents.stack(names)

        if agent_graph is None:
            agent_graph = {"nodes": names, "edges": {}}
        graph_vec, node_emb = self.agent_graph.embed(agent_graph, names, agent_node_vecs)

        trace_vec = (
            await self.trace.embed(trace_steps)
            if trace_steps
            else np.concatenate(
                [
                    np.zeros(self.trace.client.dim, dtype=np.float32),
                    np.zeros(self.trace.feature_dim, dtype=np.float32),
                ],
                axis=0,
            )
        )

        fly_vec = self._fly_vector()

        state = ControllerState(
            task_vec=task_vec.astype(np.float32, copy=False),
            agent_node_vecs=agent_node_vecs,
            agent_graph_vec=graph_vec,
            agent_node_emb=node_emb,
            trace_vec=trace_vec.astype(np.float32, copy=False),
            fly_vec=fly_vec,
            inbox_vec=self._inbox_vec(names, runtime.pending_inbox),
            produced_mask=self._produced_mask(runtime.produced_components),
            step_id=runtime.step_id,
            task_type=runtime.task_type,
            agent_names=names,
            component_tags=list(self.component_tags),
        )
        state.build_ms = (_t.perf_counter() - t0) * 1000.0
        return state

    def from_runtime_sync(
        self,
        runtime: RuntimeState,
        *,
        agent_graph: dict[str, Any] | None = None,
        trace_steps: list[dict[str, Any]] | None = None,
    ) -> ControllerState:
        """Synchronous wrapper for tests + non-async controllers."""
        return asyncio.run(
            self.from_runtime(
                runtime,
                agent_graph=agent_graph,
                trace_steps=trace_steps,
            )
        )
