# Phase 7 — Expert Traces + Imitation Learning (PLAN.md §592-595)

## What ships

| Module | What it does |
|--------|--------------|
| `flybrain/training/expert_dataset.py` | Loads `*.trace.json` files emitted by `MAS.run`, replays each step into a `RuntimeState`, and pairs it with the expert's action — yields `ImitationExample`\\s. |
| `flybrain/training/imitation.py` | Supervised cloning loop: cross-entropy on `kind` + `agent` + edge heads, BCE on the auxiliary verifier prediction, AdamW. |
| `scripts/collect_expert_traces.py` | Runs the expert (`ManualController`) over a synthetic task stream against either YandexGPT or the deterministic `MockLLMClient`. Persists per-task trace JSONs + a `summary.json` budget report. Halts cleanly on `BudgetExceededError`. |
| `scripts/run_imitation.py` | CLI: warm-start from a Phase-6 sim-pretrain checkpoint, then imitate the expert. |

## Quickstart

### 1. Dry-run (no API calls, no cost)

```bash
python scripts/collect_expert_traces.py \
    --output runs/expert_dry --backend mock --tasks 8
```

### 2. Live collection on YandexGPT Lite

```bash
YANDEX_API_KEY=... folder_id=... \
python scripts/collect_expert_traces.py \
    --output data/traces/expert/v1 \
    --backend yandex --tier lite \
    --tasks 100 --budget-rub 200
```

The collector:

* Generates `--tasks` synthetic tasks balanced across the four task
  types (Phase-6 generator).
* Runs each task through `MAS.from_specs(MINIMAL_15, …)` with
  `ManualController` as the expert.
* Persists `<task_id>.trace.json` with the full Phase-3 verification
  result (`{passed, score, errors}`).
* Tracks running cost via `BudgetTracker(hard_cap_rub=…)`. When the
  cap would be exceeded by the next call the collector halts cleanly
  and writes a `summary.json` describing how far it got.

The Yandex client uses `SQLiteCache` keyed on
`hash(messages, model, temperature)` so re-running the script cheaply
resumes — cached responses cost 0 ₽ + 0 tokens.

### 3. Imitate

```bash
python scripts/run_imitation.py \
    --controller gnn \
    --traces data/traces/expert/v1 \
    --warm-from runs/sim_pretrain/gnn.pt \
    --epochs 8 --batch-size 16 \
    --output runs/il/gnn.pt
```

`--warm-from` is optional but recommended — the synthetic-pretrain
checkpoint gives the IL loop a strong starting prior, which is what
PLAN.md §514 means by "simulation pretraining компенсирует
маленькую expert dataset на Phase 7".

## Trace format

Per-task trace JSON (mirrors `flybrain_core::Trace`):

```jsonc
{
  "task_id": "sim-coding-12345",
  "task_type": "coding",
  "steps": [
    {
      "step_id": 0,
      "active_agent": null,
      "graph_action": {"kind": "activate_agent", "agent": "Planner"},
      "verifier_score": null,
      "tokens_in": 0,
      "tokens_out": 0,
      "cost_rub": 0.0,
      "current_graph_hash": "h0"
    },
    /* ... */
  ],
  "verification": {"passed": true, "score": 0.92},
  "totals": {"tokens_in": 750, "tokens_out": 80, "cost_rub": 0.5}
}
```

The replay logic in `expert_dataset.py` is intentionally lossy — we
only reconstruct the fields the controller actually looks at (task
descriptors, `produced_components`, `last_active_agent`, recent
verifier score). Full state replay is unnecessary because the
controller's `ControllerStateBuilder` re-derives embeddings + the
agent graph from the runtime state on every tick.

## Budget plan (1000 ₽)

Per PLAN.md §666 ("буджетный режим"):

* ~150 expert traces × 4 task types ⇒ ~600 traces. With ~1 ₽ / trace
  on YandexGPT Lite (≈ 5 LLM calls × 0.2 ₽), ~600 ₽ total.
* Reserve 400 ₽ for Phase-10 benchmarks + retries.

The Yandex SDK is gated behind `flybrain[ml]`; `MockLLMClient`
covers the CI smoke path.

## Exit criterion (PLAN.md §595)

> на held-out subset IL-controller бьёт sim-only по success rate и/или по cost.

The CI smoke test only asserts the loop reduces loss across epochs.
The actual ≥ sim-only metric is exercised by the standalone
`scripts/run_imitation.py` invocation, where `epoch_accuracy[-1]`
should beat the Phase-6 sim-pretrain baseline on the same held-out
trace split.

## Out of scope

* RL / bandit finetuning — Phase 8.
* Baselines + benchmarks — Phase 9 / Phase 10.
* Real benchmark datasets (HumanEval / GSM8K / BBH-mini) — overlay
  on top of Phase-7 traces in Phase 10.
