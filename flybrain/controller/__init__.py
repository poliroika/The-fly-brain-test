"""Controllers consumed by `MAS.run`.

Phase 2 ships:

* `Controller` Protocol (the public contract).
* `ManualController` — scripted, hand-tuned plan per task type.
* `RandomController` — uniform-random baseline.

Phase 5 (PLAN.md §577-583) adds the three torch-based controllers:

* `FlyBrainGNNController` — variant A (primary): GCN over the live
  AgentGraph + policy/value/aux heads.
* `FlyBrainRNNController` — variant B: GRUCell over time, ``A_fly``
  used as a sparse weight prior on the per-agent linear.
* `LearnedRouterController` — variant C: cross-attention from the
  global state vector to per-agent reps, with a fly regularizer that
  pulls the attention weights towards the fly adjacency.

The torch controllers are imported lazily so users that don't install
the `[ml]` extra still get the Phase-2 controllers without an
``ImportError`` at module load.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from flybrain.controller.action_space import (
    KIND_ACTIVATE_AGENT,
    KIND_ADD_EDGE,
    KIND_CALL_MEMORY,
    KIND_CALL_RETRIEVER,
    KIND_CALL_TOOL_EXECUTOR,
    KIND_CALL_VERIFIER,
    KIND_NAMES,
    KIND_REMOVE_EDGE,
    KIND_SCALE_EDGE,
    KIND_TERMINATE,
    NUM_KINDS,
    ActionMask,
    ActionSpace,
)
from flybrain.controller.base import Controller
from flybrain.controller.manual import ManualController
from flybrain.controller.random_ctrl import RandomController

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flybrain.controller.gnn_controller import FlyBrainGNNController
    from flybrain.controller.learned_router import LearnedRouterController
    from flybrain.controller.rnn_controller import FlyBrainRNNController

_LAZY = {
    "FlyBrainGNNController": ("flybrain.controller.gnn_controller", "FlyBrainGNNController"),
    "FlyBrainRNNController": ("flybrain.controller.rnn_controller", "FlyBrainRNNController"),
    "LearnedRouterController": (
        "flybrain.controller.learned_router",
        "LearnedRouterController",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        mod = importlib.import_module(module_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'flybrain.controller' has no attribute {name!r}")


__all__ = [
    "KIND_ACTIVATE_AGENT",
    "KIND_ADD_EDGE",
    "KIND_CALL_MEMORY",
    "KIND_CALL_RETRIEVER",
    "KIND_CALL_TOOL_EXECUTOR",
    "KIND_CALL_VERIFIER",
    "KIND_NAMES",
    "KIND_REMOVE_EDGE",
    "KIND_SCALE_EDGE",
    "KIND_TERMINATE",
    "NUM_KINDS",
    "ActionMask",
    "ActionSpace",
    "Controller",
    "FlyBrainGNNController",
    "FlyBrainRNNController",
    "LearnedRouterController",
    "ManualController",
    "RandomController",
]
