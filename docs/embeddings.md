# Embeddings runbook (Phase 4)

Operator-facing notes for the embedding layer that lands with Phase 4 of
[`PLAN.md`](../PLAN.md). Phase 5 (the GNN / RNN / learned-router controllers)
consumes the artefacts described here.

## TL;DR

The embedding stack is a stateful **builder** (`ControllerStateBuilder`) that
turns a `RuntimeState` into a tensor-friendly `ControllerState` snapshot. It
must build that snapshot in **< 50 ms on CPU** (Phase-4 exit criterion). Five
embedders feed into the builder:

| Module | Backs | Default backend (CI) | Production backend |
|---|---|---|---|
| `task_emb.py` | `text-search-query` for the user prompt | `MockEmbeddingClient` | `YandexEmbeddingClient` |
| `agent_emb.py` | `text-search-doc` for each agent's role + system prompt | `MockEmbeddingClient` | `YandexEmbeddingClient` |
| `trace_emb.py` | mean-pool over step `output_summary` + 13 handcrafted features | `MockEmbeddingClient` | `YandexEmbeddingClient` |
| `graph_emb.py` | tiny 2-layer GCN over the live `AgentGraph` | numpy | numpy (Phase 5 swaps in torch GCN) |
| `fly_emb.py`  | spectral / node2vec embedding of the fly connectome | spectral (numpy) | node2vec (`pip install -e .[ml]`) |

All clients implement the `EmbeddingClient` ABC in `flybrain/embeddings/base.py`
— same shape as `flybrain.llm.base.LLMClient`.

## Quick start

```python
import asyncio
from flybrain.agents.specs import MINIMAL_15
from flybrain.embeddings import (
    AgentEmbedder, AgentGraphEmbedder, ControllerStateBuilder, FlyGraphEmbedder,
    MockEmbeddingClient, TaskEmbedder, TraceEmbedder,
)
from flybrain.graph.dataclasses import FlyGraph
from flybrain.runtime.state import RuntimeState
import flybrain.flybrain_native as native

async def main() -> None:
    client = MockEmbeddingClient(output_dim=256)
    agents = AgentEmbedder(client=client)
    await agents.precompute(MINIMAL_15)

    builder = ControllerStateBuilder(
        task=TaskEmbedder(client=client),
        agents=agents,
        trace=TraceEmbedder(client=client),
        fly=FlyGraphEmbedder(dim=64),
        agent_graph=AgentGraphEmbedder(in_dim=256, hidden_dim=64, out_dim=64),
    )
    builder.fly_graph = FlyGraph.from_dict(native.build_synthetic_fly_graph(64, 42))

    rt = RuntimeState(
        task_id="t1", task_type="coding", prompt="implement fizzbuzz",
        step_id=0,
        available_agents=[s.name for s in MINIMAL_15],
        pending_inbox={s.name: 0 for s in MINIMAL_15},
        last_active_agent=None,
    )
    state = await builder.from_runtime(rt)
    print(state.shapes, "build_ms=", round(state.build_ms, 2))

asyncio.run(main())
```

## Switching from the mock to live Yandex embeddings

The Yandex client expects `YANDEX_API_KEY` + `YANDEX_FOLDER_ID` (or
`folder_id`) in the environment.

```python
from flybrain.embeddings import (
    EmbeddingCache, YandexEmbeddingClient, YandexEmbeddingConfig,
)

cfg = YandexEmbeddingConfig.from_env()
cache = EmbeddingCache("data/cache/emb.sqlite")
client = YandexEmbeddingClient(config=cfg, cache=cache)
```

Live runs MUST attach an `EmbeddingCache` — `text-search-doc` calls for the
agent table only need to happen once per release, and caching the user-prompt
embedding lets the controller re-evaluate routing within a single run for free.

## Vector layouts

`ControllerState.shapes` returns the canonical layout. With the defaults used
in CI (`emb_dim=64`, `fly_dim=32`, 15 minimal agents):

```text
task_vec         : (64,)
agent_node_vecs  : (15, 64)        — per-agent text embedding
agent_graph_vec  : (32,)           — mean-pooled GCN output
agent_node_emb   : (15, 32)        — per-agent GCN output
trace_vec        : (64 + 13,)      — pooled steps ‖ handcrafted features
fly_vec          : (32,)           — graph-level fly prior
inbox_vec        : (15,)           — pending message counts per agent
produced_mask    : (6,)            — 0/1 over component_tags
```

The 13 handcrafted trace features are exposed via
`TraceEmbedder.feature_names`. Their order is fixed and documented inside
`flybrain/embeddings/trace_emb.py::_HANDCRAFTED_FEATURES`.

## Caching

* Mock client never caches (it's already deterministic and free).
* Yandex client caches in SQLite via `EmbeddingCache`; rows store the raw
  float32 vector as a blob (4× smaller than JSON, no precision loss).
* Cache key = `sha256(model_uri || mode || text)`. The mode is part of the key
  because doc / query embeddings of the same text are distinct.

The `EmbeddingCache` is thread-safe (single SQLite connection guarded by a
lock) and writes use `WAL` so concurrent readers don't block.

## Testing

```bash
make test                         # 116 baseline + 33 Phase-4 = 149 Python tests
pytest tests/python/unit/test_embeddings_clients.py -v
pytest tests/python/unit/test_embeddings_wrappers.py -v
pytest tests/python/unit/test_controller_state.py -v
```

The `test_controller_state_meets_50ms_warm_budget` test is the Phase-4 exit
criterion enforced in CI.

## What's *not* in Phase 4 (planned)

* **GNN controller** — Phase 5 lifts `AgentGraphEmbedder` into a torch
  `nn.Module` with learned weights instead of seeded random projections.
* **node2vec / GraphSAGE** — `FlyGraphEmbedder` already has the `node2vec`
  backend wired up; it just isn't exercised in CI because `torch_geometric`
  isn't a CI dependency. Production training jobs flip
  `FlyGraphEmbedder(backend="node2vec")` after `pip install -e .[ml]`.
* **Embedding budget tracking** — the Yandex client wires through
  `BudgetTracker` but the per-token tariff for embedding endpoints isn't in
  `flybrain/llm/pricing.py` yet (it lands with Phase 8 once we pull the live
  pricing table).
