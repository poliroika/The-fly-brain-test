# Phase 9 — Baselines (PLAN.md §603-605, README §15)

## What ships

| File | What it does |
|------|--------------|
| `flybrain/baselines/graphs.py` | Static AgentGraph builders: `empty_graph`, `fully_connected_graph`, `random_sparse_graph` (Erdos-Renyi), `degree_preserving_random_graph`. Each returns the JSON dict the runtime ingests via `MAS.run(initial_graph=...)`. |
| `flybrain/baselines/round_robin.py` | `RoundRobinController` floor baseline (cycles through `available_agents`, calls verifier once, terminates). |
| `flybrain/baselines/registry.py` | `BaselineSpec` + 9 builtin baselines + suite registry (`full_min`, `static`, `learned`, `smoke`). |
| `scripts/run_baselines.py` | CLI that runs a chosen suite over the same task set with the same MAS, emits `comparison.json` + `comparison.md`. |
| `tests/python/unit/test_baselines.py` | 18 smoke tests covering graph builders, registry, suites, round-robin. |

## The 9 baselines

| # | Name | Type | Notes |
|---|------|------|-------|
| 1 | `manual_graph` | static graph | Empty initial graph + `ManualController` plan |
| 2 | `fully_connected` | static graph | All-to-all edges + `ManualController` |
| 3 | `random_sparse` | static graph | Erdos-Renyi p=0.2 + `RandomController` |
| 4 | `degree_preserving` | static graph | Random rewire matching target out-degree + `ManualController` |
| 5 | `learned_router_no_prior` | learned, untrained | Phase-5 `LearnedRouter` without `init_from_fly_graph` call |
| 6 | `flybrain_prior_untrained` | learned, untrained | Phase-5 GNN with fly-prior init, no training |
| 7 | `flybrain_sim_pretrain` | learned, trained | Phase-6 sim-pretrain checkpoint loaded via `FLYBRAIN_BASELINE_SIM_PRETRAIN` env var |
| 8 | `flybrain_imitation` | learned, trained | Phase-7 imitation checkpoint via `FLYBRAIN_BASELINE_IMITATION` |
| 9 | `flybrain_rl` | learned, trained | Phase-8 RL/bandit checkpoint via `FLYBRAIN_BASELINE_RL` |

Baselines 7-9 fall back to the *untrained* variant when the
corresponding `FLYBRAIN_BASELINE_*` env var is unset or the
checkpoint file is missing — so `run_baselines.py` is always runnable
even before later phases land.

## Quickstart

### Smoke run with mock LLM (no API)

```bash
python scripts/run_baselines.py \
    --suite full_min --backend mock --tasks 12 \
    --output runs/baselines_smoke
```

Per the PLAN.md §605 exit criterion this command runs as a single
invocation and produces a `comparison.md` table.

Sample output (12 tasks, mock LLM):

| Baseline | Tasks | Success | Steps/task | Tokens/task | Cost/task |
|----------|-------|---------|------------|-------------|-----------|
| manual_graph | 12 | 1.00 | 5.2 | 220 | 0.13 RUB |
| fully_connected | 12 | 1.00 | 5.2 | 261 | 0.15 RUB |
| random_sparse | 12 | 0.42 | 7.0 | 380 | 0.21 RUB |
| degree_preserving | 12 | 1.00 | 5.2 | 220 | 0.13 RUB |
| learned_router_no_prior | 12 | 0.00 | 6.2 | 138 | 0.06 RUB |
| flybrain_prior_untrained | 12 | 0.00 | 8.0 | 0 | 0.00 RUB |
| flybrain_sim_pretrain | 12 | 0.00 | 8.0 | 0 | 0.00 RUB |
| flybrain_imitation | 12 | 0.00 | 8.0 | 0 | 0.00 RUB |
| flybrain_rl | 12 | 0.00 | 8.0 | 0 | 0.00 RUB |

The trained-flybrain rows hit 0.00 success in the smoke run because
no checkpoints are loaded. That's expected — the comparison is
meaningful once Phase-7/8 checkpoints are pointed at via
`FLYBRAIN_BASELINE_*`.

### Live run with checkpoints

```bash
export FLYBRAIN_BASELINE_SIM_PRETRAIN=runs/sim_pretrain/gnn.pt
export FLYBRAIN_BASELINE_IMITATION=runs/il/gnn.pt

YANDEX_API_KEY=... folder_id=... \
python scripts/run_baselines.py \
    --suite full_min --backend yandex \
    --tasks 40 --budget-rub 300 \
    --output data/baselines/v1
```

### Subsetting

`--only manual_graph fully_connected` runs a hand-picked subset.

`--suite static` runs the four no-LLM-controller baselines.

`--suite smoke` runs `manual_graph` + `random_sparse` only — useful
for quick CI sanity checks.

## Bug fix: `from_runtime_sync` from inside an event loop

While wiring the learned baselines through `MAS.run` we hit a real
bug: the controller's `select_action` is sync but its internals
call `ControllerStateBuilder.from_runtime_sync`, which in turn calls
`asyncio.run` — and that crashes with
``RuntimeError: asyncio.run() cannot be called from a running event loop``
when invoked from inside `MAS.run` (which is itself async).

`from_runtime_sync` now detects the running loop with
`asyncio.get_running_loop()` and dispatches to a `ThreadPoolExecutor`
worker (which has its own event loop context) when needed. Outside
event loops the fast `asyncio.run` path stays unchanged.

This unblocks the full Phase-9 suite from `run_baselines.py` and
will be load-bearing for Phase 10 too (benchmarks driving learned
controllers through the live MAS).

## Out of scope

* Real benchmark datasets (HumanEval / GSM8K / BBH-mini) — Phase 10.
* Final results table for PLAN.md §17 — Phase 11 will populate it
  using the per-baseline `comparison.json` files emitted here.
* Statistical significance / variance across seeds — Phase 11.
