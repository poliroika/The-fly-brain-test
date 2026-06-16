"""Phase-9 baselines (PLAN.md §603-605, README §15).

Nine baselines as a single registry so `scripts/run_baselines.py`
can iterate them with one command::

    1. Manual MAS graph
    2. Fully connected MAS
    3. Random sparse graph
    4. Degree-preserving random graph
    5. Learned router without fly prior  (Phase-5 LearnedRouter, untrained)
    6. FlyBrain prior without training   (Phase-5 GNN init_from_fly, untrained)
    7. FlyBrain + simulation pretraining (Phase-6 checkpoint)
    8. FlyBrain + imitation learning     (Phase-7 checkpoint)
    9. FlyBrain + RL / bandit finetuning (Phase-8 checkpoint)

Baselines 1-4 are *static graphs*: they fix `initial_graph` at task
start and pair it with the `ManualController` plan. Baselines 5-6
are *untrained controllers* (their `init_from_fly_graph` is the only
prior they get). Baselines 7-9 are *trained controllers*: same
architecture, different checkpoint. The registry abstracts both
kinds behind a single `BaselineSpec.factory(...)` callable that
returns a `(controller, initial_graph)` pair the runner can plug in
verbatim.
"""

from __future__ import annotations

from flybrain.baselines.graphs import (
    degree_preserving_random_graph,
    empty_graph,
    fully_connected_graph,
    random_sparse_graph,
)
from flybrain.baselines.registry import (
    BUILTIN_SUITES,
    BaselineSpec,
    builtin_baselines,
    list_baselines,
)
from flybrain.baselines.round_robin import RoundRobinController

__all__ = [
    "BUILTIN_SUITES",
    "BaselineSpec",
    "RoundRobinController",
    "builtin_baselines",
    "degree_preserving_random_graph",
    "empty_graph",
    "fully_connected_graph",
    "list_baselines",
    "random_sparse_graph",
]
