# Phase 6 — Simulation Pretraining (PLAN.md §585-590)

## What ships

| Module | What it does |
|--------|--------------|
| `flybrain/sim/optimal_routes.py` | Ground-truth optimal route per task type (README §12.1) + the agent → component-tag map the runtime uses to track progress. |
| `flybrain/sim/task_generator.py` | Deterministic synthetic task templates for `coding` / `math` / `research` / `tool_use`. Cheap (microsecond per sample), no LLM calls. |
| `flybrain/sim/synthetic_mas.py` | LLM-free MAS simulator: per-agent `success_prob[task_type]`, `cost`, `error_rate`. Walks any `Controller` through a task and emits a per-step trace + `RuntimeState` snapshots. |
| `flybrain/training/simulation_pretrain.py` | Supervised pretraining loop on synthetic `(state → optimal_action)` pairs. Returns `PretrainResult` with per-step losses + per-epoch held-out accuracy. |
| `scripts/run_simulation_pretrain.py` | CLI entrypoint: pretrain any of the three Phase-5 controllers, optionally evaluate on `SyntheticMAS`, save a checkpoint + JSON sidecar. |

## Quickstart

```bash
# 30-epoch pretrain on the GNN controller, ~2 min on CPU.
python scripts/run_simulation_pretrain.py \
    --controller gnn --epochs 30 --n-per-type 64 --batch-size 64 \
    --lr 3e-3 --hidden-dim 64 --evaluate-on-sim \
    --output runs/sim_pretrain/gnn.pt
```

The script logs:

```
[pretrain] controller=gnn cfg=PretrainConfig(...)
[pretrain] done examples=… loss[first/last]=… final_acc=…
[eval] tasks=32 success_rate=… mean_final_score=…
[pretrain] saved checkpoint to runs/sim_pretrain/gnn.pt
```

## Optimal routes (README §12.1)

```python
OPTIMAL_ROUTES = {
    "coding":   ["Planner", "Coder", "TestRunner", "Debugger", "Verifier"],
    "math":     ["Planner", "MathSolver", "Critic", "Verifier"],
    "research": ["Planner", "Researcher", "Retriever", "CitationChecker", "Finalizer"],
    "tool_use": ["Planner", "ToolExecutor", "SchemaValidator", "Verifier"],
}
```

The expert dataset walks each route step-by-step, snapshots the
`RuntimeState` the runtime would have at that point (with
`produced_components` reflecting the agents that already fired),
and labels it with the next agent in the route. After the last
agent the label is `terminate`.

## Library use

```python
from flybrain.sim import TaskGenerator
from flybrain.training import PretrainConfig, simulation_pretrain
from flybrain.controller import FlyBrainGNNController
# (build a ControllerStateBuilder + controller as in docs/controller.md)

cfg = PretrainConfig(n_per_type=64, epochs=30, batch_size=64, learning_rate=3e-3)
tasks = TaskGenerator(seed=0).balanced_dataset(n_per_type=cfg.n_per_type)

result = simulation_pretrain(
    controller, agent_names=[a.name for a in MINIMAL_15],
    config=cfg, tasks=tasks,
)
print(result.final_accuracy, result.epoch_accuracy)
```

## Synthetic MAS evaluation

`SyntheticMAS.run(controller, task)` produces a `SyntheticOutcome`
with:

* `success`     — `final_score >= 0.6` (i.e. the controller recovered
  most of the optimal route's components).
* `final_score` — fraction of optimal-route components that ended up
  in `produced_components`.
* `actions`, `states`, `rewards`, `cost`, `steps`.

This is what the Phase-8 RL/bandit finetuning will use as the env
loop.

## Exit criterion (PLAN.md §590)

> controller сходится за <10 минут на CPU и решает sim-задачи на ≥0.85 success.

The CI smoke test only asserts the *training loop reduces loss
across epochs* — the actual ≥0.85 metric is exercised by the
standalone `scripts/run_simulation_pretrain.py` invocation, which
runs for several minutes and prints the final accuracy + the
synthetic-MAS success rate.

## Out of scope

* Imitation learning from real expert traces — Phase 7.
* RL / bandit finetuning + reward shaping — Phase 8.
* Baselines + benchmarks — Phase 9 / Phase 10.
