"""Phase-8 reinforcement-learning utilities (PLAN.md §597-601).

The package layers three families of algorithms on top of the
Phase-5 controllers:

* :mod:`rewards`   — the reward formula from README §12.3.
* :mod:`bandit`    — contextual LinUCB / Thompson over discrete
                      kind+agent action heads.
* :mod:`reinforce` — vanilla policy-gradient on top of the existing
                      controller heads (works for GNN / RNN / Router).
* :mod:`ppo`       — clipped-objective PPO with a value-head
                      baseline. Same input/output contract as REINFORCE.
* :mod:`offline_rl`— per-trace replay buffer + offline REINFORCE
                      so we can re-train the controller from
                      already-collected traces without spending more
                      LLM tokens.
"""

from __future__ import annotations

from flybrain.training.rl.bandit import LinUCBBandit, ThompsonBandit
from flybrain.training.rl.offline_rl import (
    OfflineRLConfig,
    OfflineRLResult,
    offline_rl_train,
)
from flybrain.training.rl.ppo import PPOConfig, PPOResult, ppo_train
from flybrain.training.rl.reinforce import (
    ReinforceConfig,
    ReinforceResult,
    reinforce_train,
)
from flybrain.training.rl.rewards import RewardConfig, compute_reward

__all__ = [
    "LinUCBBandit",
    "OfflineRLConfig",
    "OfflineRLResult",
    "PPOConfig",
    "PPOResult",
    "ReinforceConfig",
    "ReinforceResult",
    "RewardConfig",
    "ThompsonBandit",
    "compute_reward",
    "offline_rl_train",
    "ppo_train",
    "reinforce_train",
]
