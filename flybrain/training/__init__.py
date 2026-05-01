"""Training entrypoints — Phase 6 (sim pretrain), Phase 7 (IL),
Phase 8 (RL/bandit).

Phase 6 ships:

* `simulation_pretrain` — supervised pretrain on synthetic
  ``(state, optimal_action)`` pairs.
"""

from __future__ import annotations

from flybrain.training.simulation_pretrain import (
    PretrainConfig,
    PretrainResult,
    expert_dataset,
    simulation_pretrain,
)

__all__ = [
    "PretrainConfig",
    "PretrainResult",
    "expert_dataset",
    "simulation_pretrain",
]
