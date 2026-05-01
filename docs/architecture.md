# Architecture

This document is the high-level map of the FlyBrain Optimizer system. The
authoritative narrative is `PLAN.md`; this file is the quick reference.

## System overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                         FlyBrain Optimizer                            │
│                                                                        │
│   FlyWire ──► flybrain-graph ──► fly_graph_K (compressed connectome)   │
│                                          │                             │
│                                          ▼                             │
│   Task ──► embeddings ──► FlyBrain controller ──► GraphAction          │
│                                          │            │                │
│                                          ▼            ▼                │
│                                    flybrain-runtime  agent_graph       │
│                                          │            │                │
│                                          ▼            ▼                │
│                                    LLM agents (Yandex AI Studio)       │
│                                          │                             │
│                                          ▼                             │
│                                    flybrain-verify (Rust)              │
│                                    flybrain.verification.llm (Python)  │
│                                          │                             │
│                                          ▼                             │
│                                       Trace + metrics                  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Data flow per task (Phase 2+)

1. **Task in.** `TaskSpec` is loaded from a benchmark or synthetic source.
2. **Embed.** `flybrain.embeddings` produces task / agent / trace / graph /
   fly embeddings.
3. **Controller tick.** `FlyBrainController` (GNN / RNN / router) reads the
   current `AgentGraph` snapshot + embeddings and emits a `GraphAction`.
4. **Runtime apply.** `flybrain_native.runtime` mutates the agent graph (or
   activates an agent / calls a verifier / terminates).
5. **Agent step.** Active agent talks to YandexGPT through `flybrain.llm`.
6. **Verify.** Deterministic Rust verifiers fire on every step; LLM-judge
   verifiers fire at task end.
7. **Persist.** A `TraceStep` is written to JSONL; `Totals` are updated.
8. **Repeat** until `terminate()` or budget hits the hard cap.

## Module boundaries

| Layer | Crate / package | What lives there |
|---|---|---|
| Data contracts | `flybrain-core` | `AgentSpec`, `Trace*`, `GraphAction`, `VerificationResult`, `AgentGraph`, `FlyGraph`, `TaskSpec` |
| Connectome | `flybrain-graph` | Zenodo / Codex loaders, Louvain / Leiden / spectral / region / cell-type compressors |
| Runtime | `flybrain-runtime` | Message bus, scheduler, trace writer (Phase 2) |
| Determinism | `flybrain-verify` | Schema, tool_use, budget, trace verifiers |
| FFI | `flybrain-py` | PyO3 bindings exposed as `flybrain_native` |
| CLI | `flybrain-cli` | `flybrain` binary (build / sim / bench / report) |
| LLM | `flybrain.llm` | `YandexClient`, `MockLLMClient`, SQLite cache, BudgetTracker |
| Agents | `flybrain.agents` | Agent specs + prompts (Phase 2) |
| Controller | `flybrain.controller` | GNN / RNN / Learned Router (Phase 5) |
| Embeddings | `flybrain.embeddings` | Task / agent / trace / graph / fly (Phase 4) |
| Training | `flybrain.training` | SSL, sim, IL, RL / bandit / PPO (Phases 6–8) |
| Baselines | `flybrain.baselines` | 9 baselines (Phase 9) |
| Benchmarks | `flybrain.benchmarks` | HumanEval, GSM8K, BBH-mini, synthetic routing (Phase 10) |
| Eval | `flybrain.eval` | Metrics, ablation tables, report (Phases 10–11) |

## Why Rust + Python

Rust hosts the deterministic, hot-path code (graph compression, runtime
scheduler, deterministic verifiers, trace writer). Python hosts the ML stack
(controllers, embeddings, training) and everything that talks to YandexGPT.
Maturin glues them together as a mixed-layout project; the Rust core ships as
a single PyO3 module `flybrain_native`. See
[`docs/rust_python_boundary.md`](rust_python_boundary.md) for the design
contract between the two halves.
