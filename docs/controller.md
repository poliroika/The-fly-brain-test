# Phase 5 — Controller (PLAN.md §577-583)

## What ships

Phase 5 adds three torch-based controllers on top of the Phase-4
`ControllerState`. They all satisfy the `flybrain.controller.Controller`
Protocol, so `MAS.run` can drop them in next to `ManualController` /
`RandomController` without changes.

| Variant | Class                       | What's special                                                                 |
|---------|-----------------------------|-------------------------------------------------------------------------------|
| A       | `FlyBrainGNNController`     | 2-layer Kipf GCN over the live AgentGraph; pooled into the global state       |
| B       | `FlyBrainRNNController`     | `GRUCell` over time; per-agent linear weight initialised from `A_fly`         |
| C       | `LearnedRouterController`   | Cross-attention from state → agents; `fly_regularizer_loss()` pulls weights toward `A_fly` |

All three share the same encoder (`StateEncoder`) and head bundle
(`PolicyHeads`):

* `kind`     → 9-way logits over `GraphAction` discriminants.
* `agent`    → K-way logits, used by `activate_agent`.
* `edge`     → two K-way heads (`from`, `to`) + one continuous scalar
  (used by `add_edge` / `remove_edge` / `scale_edge`).
* `value`    → scalar V(s) for value-based and PPO training.
* `aux`      → predicted verifier score in `[0, 1]` (auxiliary loss).

## Action space + masking

`flybrain.controller.action_space.ActionSpace` decodes the head outputs
into the `GraphAction` JSON dict that the runner consumes:

```python
from flybrain.controller import ActionSpace, KIND_NAMES
space = ActionSpace(agent_names=["Planner", "Coder", "TestRunner"])

mask = space.legal_mask()
# mask.kind_mask : (9,) — illegal kinds zeroed out (e.g. activate_agent
#                          when no agents are available, edge ops when
#                          K < 2).
# mask.agent_mask : (K,)
# mask.edge_mask  : (K, K) without self-loops

space.decode(0, agent_id=1)
# -> {"kind": "activate_agent", "agent": "Coder"}

space.decode(1, edge_from_id=0, edge_to_id=2, edge_weight=0.7)
# -> {"kind": "add_edge", "from": "Planner", "to": "TestRunner",
#     "weight": 0.7}
```

Discriminants mirror the Rust `GraphAction` enum
(`crates/flybrain-core/src/action.rs`):

| ID | kind                 |
|----|----------------------|
| 0  | `activate_agent`     |
| 1  | `add_edge`           |
| 2  | `remove_edge`        |
| 3  | `scale_edge`         |
| 4  | `call_memory`        |
| 5  | `call_retriever`     |
| 6  | `call_tool_executor` |
| 7  | `call_verifier`      |
| 8  | `terminate`          |

## Quickstart

```python
import asyncio
from flybrain.agents.specs import MINIMAL_15
from flybrain.controller import FlyBrainGNNController
from flybrain.embeddings import (
    AgentEmbedder, AgentGraphEmbedder, ControllerStateBuilder,
    FlyGraphEmbedder, MockEmbeddingClient, TaskEmbedder, TraceEmbedder,
)
from flybrain.runtime.state import RuntimeState

client = MockEmbeddingClient(output_dim=32)
agents = AgentEmbedder(client)
asyncio.run(agents.precompute(MINIMAL_15))

builder = ControllerStateBuilder(
    task=TaskEmbedder(client),
    agents=agents,
    trace=TraceEmbedder(client),
    fly=FlyGraphEmbedder(dim=8),
    agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
)

ctrl = FlyBrainGNNController(
    builder=builder,
    task_dim=client.dim, agent_dim=client.dim,
    graph_dim=32, trace_dim=client.dim + 13,
    fly_dim=8, produced_dim=6,
)

rs = RuntimeState(
    task_id="t1", task_type="coding", prompt="hello",
    step_id=0, available_agents=["Planner", "Coder", "TestRunner"],
    pending_inbox={"Planner": 1}, last_active_agent=None,
)
print(ctrl.select_action(rs))  # {"kind": "activate_agent", "agent": ...}
```

## Fly-graph initialisation

Variant B and C consume the fly graph at construction so they start
respecting the fly prior:

```python
from flybrain.controller import FlyBrainRNNController, LearnedRouterController

rnn = FlyBrainRNNController(builder=builder, ...)
rnn.init_from_fly_graph(fly_graph, num_agents=len(agent_names))

router = LearnedRouterController(builder=builder, ...)
router.init_from_fly_graph(fly_graph, num_agents=len(agent_names))

# Phase-7/8 training: add the regulariser to the policy loss.
loss = policy_loss + 1e-3 * router.fly_regularizer_loss()
```

The projection from the fly-graph node count `n` to the agent count
`k` is deterministic (seeded with `42`) so the same `(fly_graph, k)`
pair produces the same prior across runs.

## Training-time vs inference-time

* `controller(controller_state) -> HeadOutputs` — used during training.
  No masking, raw logits + value + aux. Gradients flow through every
  parameter.
* `controller.select_action(runtime_state) -> dict` — used by
  `MAS.run`. Wraps the masker, decodes via `ActionSpace`, returns the
  JSON-serialised `GraphAction`.

## Smoke tests (PLAN.md §583)

`tests/python/unit/test_controllers_torch.py` covers:

* All three controllers: `forward(...)` shape + `loss.backward()` grad
  propagation across every learnable parameter.
* Fly-graph initialisation modifies the relevant weights (RNN's
  `_AFlyLinear`; LearnedRouter's `_fly_prior`).
* `select_action(...)` returns a dict with `kind` ∈ `KIND_NAMES`.
* Self-loop avoidance for edge ops.
* Determinism across two instances seeded identically.
