# Runtime architecture (Phase 2)

The MAS runtime is a **Rust core** that owns mutable state plus a thin
**Python layer** that drives the per-task event loop, talks to the LLM,
and runs side-effect tools. This document describes how a single task
flows through the system end-to-end.

## High-level picture

```text
                ┌────────────────────────────────────────────────────┐
                │                Python (flybrain.runtime)           │
                │                                                    │
   Task ──────► │   ┌──── Controller.select_action(state) ──┐         │
                │   ▼                                        │        │
                │   GraphAction                              │        │
                │   │                                        │        │
                │   ▼                       Rust (flybrain_native)    │
                │ ┌─────────────────────────────────────────────────┐ │
                │ │  Scheduler.apply(action) ──► SchedulerStatus    │ │
                │ │     ├ AgentGraph.apply(...)                     │ │
                │ │     └ step_id, last_active, terminated          │ │
                │ │                                                 │ │
                │ │  MessageBus.send / pop                          │ │
                │ │  TraceWriter.record_step (JSONL on disk)        │ │
                │ └─────────────────────────────────────────────────┘ │
                │   │                                        ▲        │
                │   ▼                                        │        │
                │   Dispatch on status:                      │        │
                │     • run_agent  → Agent.step (LLM + tool) │        │
                │     • call_*     → memory/retriever/verify │        │
                │     • terminate  → exit loop               │        │
                │                                            │        │
                │   build_state(task, sched, bus, mem, ret) ─┘        │
                └────────────────────────────────────────────────────┘
```

## The two halves of the boundary

| Concern              | Rust (`flybrain-runtime`)                  | Python (`flybrain.runtime`) |
|----------------------|--------------------------------------------|-----------------------------|
| `AgentGraph` mutate  | `Scheduler::apply` (deterministic hash)    | —                           |
| Step counter         | `Scheduler::advance_step`                  | —                           |
| Inter-agent routing  | `MessageBus.send/pop` (FIFO, monotonic id) | —                           |
| Trace persistence    | `TraceWriter` (JSONL + finalised dict)     | builds `TraceStep` payloads |
| Agent.step (LLM)     | —                                          | `Agent` class               |
| Tools                | —                                          | `tools/` package            |
| Episodic / vector mem| —                                          | `memory/` package           |
| BM25 retriever       | —                                          | `retriever/bm25.py`         |
| Controller           | —                                          | `controller/` package       |
| Verifier (Phase 3)   | `flybrain-verify::BudgetVerifier`          | `flybrain.verification.*`   |

The Python side never owns mutable graph state — every action goes
through `Scheduler.apply`. This keeps the `current_graph_hash` in the
trace consistent with whatever the controller sees on the next tick,
which matters for deterministic replay.

## One tick of the loop

`flybrain.runtime.runner.MAS.run` is the canonical implementation. The
loop body, condensed:

```python
state  = build_state(task, sched, bus, last_step, last_verifier)
action = controller.select_action(state)        # Python decision
status = sched.apply(action)                     # Rust state mutation

if status["kind"] == "run_agent":
    inbox  = drain(bus, status["agent"])
    result = await agent[status["agent"]].step(state, inbox)
    forward(result.output, edges_out_of(status["agent"]))   # via bus
    memory.append(...)
    retriever.add(result.output_text)

elif status["kind"] == "call_verifier":
    verdict = verifier.check(state)              # rule-based stub in Phase 2
                                                 # full pipeline in Phase 3

elif status["kind"] in {"call_memory", "call_retriever", "call_tool_executor"}:
    # Phase 2 records the call as a trace step; full handlers in Phase 3+

trace.record_step(make_step(...))                # JSONL line on disk
sched.advance_step()
if status["kind"] == "terminated":
    break
```

## What gets persisted

For every task two artefacts are written under `trace_dir/`:

| File                           | Content                                |
|--------------------------------|----------------------------------------|
| `<task_id>.steps.jsonl`        | one [`TraceStep`](../crates/flybrain-core/src/trace.rs) per line, flushed on every `record_step` |
| `<task_id>.trace.json` *(opt)* | the full [`Trace`](../crates/flybrain-core/src/trace.rs) document the caller persists after `finalize` |

The JSONL is intentionally append-only and flushed each step so a
crashing or long-running agent leaves a usable partial trace on disk.

## Action dispatch table

| `GraphAction` discriminant | `SchedulerStatus`        | Python side-effect                                |
|----------------------------|--------------------------|---------------------------------------------------|
| `activate_agent`           | `RunAgent(name)`         | `Agent.step` + message routing                    |
| `add_edge` / `remove_edge` | `GraphMutation`          | none (next tick the message bus uses new edges)   |
| `scale_edge`               | `GraphMutation`          | none                                              |
| `call_memory`              | `CallMemory`             | `EpisodicMemory.latest(...)` (Phase 3 expands)    |
| `call_retriever`           | `CallRetriever`          | `BM25Retriever.top_k(...)`                        |
| `call_tool_executor`       | `CallToolExecutor`       | `ToolRegistry.call(name, args)`                   |
| `call_verifier`            | `CallVerifier`           | rule-based verifier in Phase 2; LLM-judges in P3  |
| `terminate`                | `Terminated`             | runner exits the loop                             |

## Agents shipped in Phase 2

`flybrain.agents.specs` ships **15 minimal + 10 extended = 25** named
`AgentSpec`s, each with `system_prompt`, `tools`, and a `model_tier`
matching `configs/llm/yandex.yaml::agent_to_model`. They round-trip
through `flybrain_native.agent_spec_round_trip` in
`tests/python/unit/test_agent_specs.py`.

| Tier    | Agents                                                                 |
|---------|------------------------------------------------------------------------|
| `pro`   | `Planner`, `Verifier`, `Critic`, `Judge`, `ProofChecker`              |
| `lite`  | the other 20 (mechanical / fast roles)                                |

## Controllers in Phase 2

| Class                | When used                                  |
|----------------------|--------------------------------------------|
| `ManualController`   | hand-tuned plan per `task_type` (baseline) |
| `RandomController`   | random valid action (smoke test, §15.C)    |

`Controller` is a Protocol; Phase-5 controllers (`LearnedRouter`,
`FlyBrainGNN`, `FlyBrainRNN`) plug into the same shape.

## Tools

| Tool             | Purpose                                       | Real impl in Phase 2? |
|------------------|-----------------------------------------------|------------------------|
| `python_exec`    | run a Python snippet in a fresh subprocess    | yes (subprocess + timeout + denylist) |
| `unit_tester`    | run code + assert-based tests                 | yes (built on `python_exec`)          |
| `file_tool`      | read / list files inside a sandbox root       | yes (path-escape rejection)           |
| `web_search`     | look up canned snippets by substring          | stubbed; HTTP impl ships later        |

Each tool returns a `ToolResult` that projects to the `ToolCall` JSON
schema used by `TraceStep.tool_calls`.

## Determinism guarantees

* `MessageBus` ids are monotonic per bus instance.
* `Scheduler` always returns the same `SchedulerStatus` for the same
  `(graph, action, history)` triple (`cargo test scheduler::`).
* `RandomController` re-seeds itself per task from
  `seed ^ hash(task_id)` so the same `(seed, task_id)` always produces
  the same action sequence.
* `MockLLMClient` is deterministic given the rule list.
* The integration test
  [`tests/python/integration/test_mas_runtime_mock.py`](../tests/python/integration/test_mas_runtime_mock.py)
  pins traces for three task types (coding / math / research) plus a
  random-controller smoke run. Re-run after any runtime change.

## Extending the runtime

* New action kind — add a `GraphAction` variant in
  `flybrain-core::action`, a `SchedulerStatus` variant + dispatch arm in
  `flybrain-runtime::scheduler`, the matching `runtime_py.rs` mapping,
  and the Python dispatch arm in `runner.MAS.run`. Update the table in
  this document.
* New tool — implement the `Tool` protocol, register in
  `default_tool_registry()`, add a unit test in
  `tests/python/unit/test_tools.py`.
* New controller — implement `Controller.select_action`, drop it under
  `flybrain.controller`, and exercise it via the integration test.

## Phase 2 exit criteria checklist

- [x] `cargo test -p flybrain-runtime` passes (14 tests).
- [x] `cargo clippy --workspace -- -D warnings` clean.
- [x] PyO3 bindings expose `Scheduler`, `MessageBus`, `TraceWriter`.
- [x] `flybrain.runtime` ships `Agent`, `MAS`, `MASConfig`, `Task`,
      `RuntimeState`.
- [x] 4 deterministic tools (`python_exec`, `unit_tester`, `file_tool`,
      `web_search`) registered by default.
- [x] `EpisodicMemory`, `VectorMemory`, `BM25Retriever`.
- [x] 25 named `AgentSpec`s with prompts + tools + tiers.
- [x] `ManualController`, `RandomController`.
- [x] `tests/python/integration/test_mas_runtime_mock.py` runs three
      task types end-to-end on the mock LLM and asserts trace shape.
- [x] `ruff check`, `ruff format --check`, `mypy flybrain` all clean.
