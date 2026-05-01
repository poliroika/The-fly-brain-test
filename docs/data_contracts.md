# Data contracts

Every shared structure between Rust and Python is defined ONCE in
`crates/flybrain-core/src/`. The Python side never re-declares these types;
it round-trips them through JSON via `flybrain_native.*_round_trip(...)` and
trusts the Rust definitions.

## Type index

| Concept | Rust source | Python access |
|---|---|---|
| `AgentSpec` | `crates/flybrain-core/src/agent.rs` | `flybrain_native.agent_spec_round_trip` |
| `ModelTier` | same | enum string `"lite"` / `"pro"` |
| `GraphAction` | `crates/flybrain-core/src/action.rs` | `flybrain_native.graph_action_round_trip` |
| `VerificationResult` | `crates/flybrain-core/src/verify.rs` | `flybrain_native.verification_result_round_trip` |
| `TraceStep` / `Trace` / `Totals` / `ToolCall` | `crates/flybrain-core/src/trace.rs` | `flybrain_native.trace_round_trip` |
| `FlyGraph` / `NodeMetadata` | `crates/flybrain-core/src/graph.rs` | `flybrain_native.fly_graph_round_trip` |
| `AgentGraph` | same | `flybrain_native.agent_graph_round_trip`, `agent_graph_hash` |
| `TaskSpec` / `TaskBudget` / `TaskType` | `crates/flybrain-core/src/task.rs` | (Phase 2) |

## GraphAction discriminants

Stable IDs used by the controller's action head. Once trained on these, do
NOT renumber them.

| ID | Variant |
|---:|---|
| 0 | `activate_agent { agent }` |
| 1 | `add_edge { from, to, weight }` |
| 2 | `remove_edge { from, to }` |
| 3 | `scale_edge { from, to, factor }` |
| 4 | `call_memory` |
| 5 | `call_retriever` |
| 6 | `call_tool_executor` |
| 7 | `call_verifier` |
| 8 | `terminate` |

## VerificationResult shape

```json
{
  "passed": true,
  "score": 0.97,
  "errors": [],
  "warnings": [],
  "failed_component": null,
  "suggested_next_agent": null,
  "reward_delta": 0.0
}
```

`reward_delta` is consumed directly by the trainer. `passed` is the headline
pass/fail. `score` is graded in [0, 1] for partial-credit reward shaping.

## AgentGraph hash

`flybrain_native.agent_graph_hash({nodes, edges})` returns a 16-hex-char FNV-1a
hash of the canonicalized JSON. The hash is stable under insert order and
captures node + edge weight changes. Used as `current_graph_hash` on each
`TraceStep` so traces can reference snapshots cheaply.

## Phase 0 status

Phase 0 only ships the type round-trip helpers. Strongly-typed PyO3 wrapper
classes (`AgentGraphHandle`, `RuntimeHandle`, `BudgetVerifierHandle`) arrive
with their respective phases.
