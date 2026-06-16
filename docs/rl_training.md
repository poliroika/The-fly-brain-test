# Phase 8 — RL / Bandit Finetuning (PLAN.md §597-601, README §12.3-12.4)

## What ships

| File | What it does |
|------|--------------|
| `flybrain/training/rl/rewards.py` | `RewardConfig` + `compute_reward(trace)` — the README §12.3 reward formula (success + verifier - tokens - calls - latency - failed tools - graph density). Robust to missing fields. |
| `flybrain/training/rl/bandit.py` | `LinUCBBandit` (linear UCB, Sherman-Morrison rank-1 updates) and `ThompsonBandit` (Bayesian linreg + Thompson sampling). Pure-numpy, picklable, action-mask aware. |
| `flybrain/training/rl/reinforce.py` | Vanilla policy gradient (REINFORCE) over the Phase-5 controllers; uses the existing value head as a state-dependent baseline + small entropy bonus. |
| `flybrain/training/rl/ppo.py` | Clipped-objective PPO with cached old-policy log-probs and configurable inner-loop epochs per collected batch. |
| `flybrain/training/rl/offline_rl.py` | Thin wrapper that loads `*.trace.json` from disk and runs REINFORCE — README §12.4 offline-RL-from-traces flavour. Zero LLM calls. |
| `scripts/run_rl.py` | CLI: `reinforce` and `ppo` subcommands; supports `--warm-from <Phase-7 imitation .pt>` and `--only-passed`. |
| `tests/python/unit/test_rl.py` | 15 smoke tests covering rewards, bandits (LinUCB learns best arm in 200 steps), REINFORCE / PPO / offline RL on synthetic traces. |

## Reward formula

```
reward = (
    success_weight * success_score                          # 1 if verification.passed else 0
  + verifier_weight * mean_verifier_score                   # 0..1 across steps
  - alpha_tokens * (tokens_in + tokens_out)
  - beta_llm_calls * num_steps
  - gamma_latency_s * wall_seconds
  - delta_failed_tool_calls * num_failed_tools
  - eta_graph_density * mean_per_step_density
)
```

Defaults match PLAN.md §666 "budget mode" — small but non-zero
penalties so the learned policy doesn't drift to an arbitrarily
expensive policy. Every coefficient lives in `RewardConfig` and is
overridable.

## Quickstart

### Bandit smoke

```python
from flybrain.training.rl import LinUCBBandit
import numpy as np

b = LinUCBBandit(num_arms=4, context_dim=64, alpha=0.5)
ctx = np.random.randn(64)
arm = b.select(ctx)              # pick under UCB
b.update(arm, ctx, reward=1.0)   # observe reward
```

`LinUCBBandit.select` and `ThompsonBandit.select` accept an optional
boolean `action_mask` so illegal actions are excluded from the argmax
without affecting the regression state.

### Offline REINFORCE on Phase-7 traces

```bash
python scripts/run_rl.py reinforce \
    --controller gnn \
    --traces data/traces/v1 \
    --warm-from runs/imitation/gnn.pt \
    --epochs 5 \
    --output runs/rl/gnn_reinforce.pt
```

### Offline PPO on the same batch

```bash
python scripts/run_rl.py ppo \
    --controller gnn \
    --traces data/traces/v1 \
    --warm-from runs/imitation/gnn.pt \
    --iterations 4 --epochs-per-batch 4 \
    --output runs/rl/gnn_ppo.pt
```

The CLI writes both a `.pt` checkpoint (controller `state_dict` +
sidecar) and a `.json` next to it with the per-iteration return/loss
curves so training-time monitoring is offline-friendly.

## Plugging the trained controller into Phase-9 baselines

Phase-9's runner already pre-wires three checkpoint slots:

```bash
export FLYBRAIN_BASELINE_RL=runs/rl/gnn_ppo.pt
python scripts/run_baselines.py --suite full_min --backend yandex \
    --tasks 40 --budget-rub 200 --output data/baselines/v2
```

Baseline #9 (`flybrain_rl`) will then load the PPO checkpoint and
appear in the comparison table next to manual / fully-connected /
imitation variants.

## Live smoke (Phase-7 YandexGPT traces, no extra cost)

```bash
$ python scripts/run_rl.py reinforce --controller gnn \
    --traces runs/expert_live_smoke --epochs 2 \
    --output /tmp/rl_smoke/gnn_reinforce.pt
[reinforce] 4 traces, controller=gnn
[saved] /tmp/rl_smoke/gnn_reinforce.pt + /tmp/rl_smoke/gnn_reinforce.json
{
  "epoch_returns": [1.0768, 1.0768],
  "epoch_losses": [6.21, 6.18],
  ...
}

$ python scripts/run_rl.py ppo --controller gnn \
    --traces runs/expert_live_smoke --iterations 2 --epochs-per-batch 2 \
    --output /tmp/rl_smoke/gnn_ppo.pt
[ppo] 4 traces, controller=gnn
{
  "iteration_returns": [1.0768, 1.0768],
  "iteration_losses": [-0.889, -0.891],
  "clip_fraction": 0.0,
  ...
}
```

The smoke run only has 4 traces so the curves are flat —
`run_rl.py` shines once the Phase-7 collection grows to 100+ traces
(estimated ~40 RUB on YandexGPT Lite at the rates from the live
trial, well within the 1000 RUB budget).

## Out of scope

* Online RL with live LLM rollouts inside `MAS.run` (would burn
  budget per training step). The current offline-RL design is
  budget-friendly and works with Phase-7 traces.
* Actor-Critic with a separate critic network — the controller's
  built-in value head is the baseline.
* GRPO / DPO style preference optimisation — README §12.3 lists
  these as alternatives but PLAN says "достаточно contextual
  bandit или simple PPO".
