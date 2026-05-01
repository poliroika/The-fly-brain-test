"""Training entrypoints — Phase 6 (sim pretrain), Phase 7 (IL),
Phase 8 (RL/bandit).

Phase 6 ships:

* `simulation_pretrain` — supervised pretrain on synthetic
  ``(state, optimal_action)`` pairs.

Phase 7 ships:

* `imitation_train` — supervised cloning from real expert traces
  collected with `scripts/collect_expert_traces.py`.
* `expert_dataset` helpers — load / replay trace JSONs and emit
  ``ImitationExample``\\s.
"""

from __future__ import annotations

from flybrain.training.expert_dataset import (
    ImitationExample,
    TraceFile,
    collect_examples,
    iter_traces,
    load_trace,
    trace_to_examples,
)
from flybrain.training.imitation import (
    ImitationConfig,
    ImitationResult,
    imitation_train,
)
from flybrain.training.simulation_pretrain import (
    PretrainConfig,
    PretrainResult,
    expert_dataset,
    simulation_pretrain,
)

__all__ = [
    "ImitationConfig",
    "ImitationExample",
    "ImitationResult",
    "PretrainConfig",
    "PretrainResult",
    "TraceFile",
    "collect_examples",
    "expert_dataset",
    "imitation_train",
    "iter_traces",
    "load_trace",
    "simulation_pretrain",
    "trace_to_examples",
]
