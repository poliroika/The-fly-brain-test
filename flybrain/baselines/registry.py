"""Phase-9 baseline registry (PLAN.md §603-605, README §15).

Each entry is a small description of a baseline plus a *factory*
that produces ``(controller, initial_graph)`` for a given agent
roster. The registry lets ``scripts/run_baselines.py`` iterate the
suite without hand-coding 9 separate setups.

Static graph baselines (#1-#4) pair an ``ManualController`` with a
fixed initial graph; learned baselines (#5-#9) instantiate one of
the Phase-5 controllers and optionally load a checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flybrain.baselines.graphs import (
    degree_preserving_random_graph,
    empty_graph,
    fully_connected_graph,
    random_sparse_graph,
)
from flybrain.controller import ManualController, RandomController
from flybrain.controller.base import Controller

BaselineFactory = Callable[
    [list[str]],
    tuple[Controller, dict[str, Any] | None],
]


@dataclass(slots=True)
class BaselineSpec:
    """Lightweight description of one baseline.

    ``factory`` accepts ``agent_names`` and returns the
    ``(controller, initial_graph)`` pair the runtime ingests.
    ``initial_graph`` is ``None`` for baselines that rely on the
    runtime's default empty graph.
    """

    name: str
    """Stable identifier; used as the column header in the comparison table."""
    description: str
    factory: BaselineFactory
    tags: list[str] = field(default_factory=list)
    """Free-form tags (e.g. ``"static-graph"``, ``"untrained"``,
    ``"trained"``) so suites can filter."""


# -- factories -----------------------------------------------------------------


def _manual_with_graph(builder: Callable[[list[str]], dict[str, Any]]) -> BaselineFactory:
    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        return ManualController(), builder(agent_names)

    return factory


def _manual_baseline(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
    """#1 Manual MAS graph — runtime-default empty initial graph,
    plus the hand-tuned ManualController plan."""
    return ManualController(), empty_graph(agent_names)


def _fully_connected_baseline(
    agent_names: list[str],
) -> tuple[Controller, dict[str, Any] | None]:
    """#2 Fully connected MAS — broadcast graph + ManualController."""
    return ManualController(), fully_connected_graph(agent_names)


def _random_sparse_baseline(seed: int = 0) -> BaselineFactory:
    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        return RandomController(seed=seed), random_sparse_graph(agent_names, seed=seed)

    return factory


def _degree_preserving_baseline(seed: int = 0) -> BaselineFactory:
    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        return ManualController(), degree_preserving_random_graph(agent_names, seed=seed)

    return factory


def _learned_router_no_prior(
    builder_factory: Callable[[], Any] | None = None,
) -> BaselineFactory:
    """#5 LearnedRouter without fly-prior init."""

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.controller import LearnedRouterController
        from flybrain.embeddings import (
            AgentEmbedder,
            AgentGraphEmbedder,
            ControllerStateBuilder,
            FlyGraphEmbedder,
            MockEmbeddingClient,
            TaskEmbedder,
            TraceEmbedder,
        )

        client = MockEmbeddingClient(output_dim=32)
        agents_emb = AgentEmbedder(client)
        from flybrain.agents.specs import MINIMAL_15

        agents_emb.precompute_sync(MINIMAL_15)
        builder = ControllerStateBuilder(
            task=TaskEmbedder(client),
            agents=agents_emb,
            trace=TraceEmbedder(client),
            fly=FlyGraphEmbedder(dim=8),
            agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
        )
        ctrl = LearnedRouterController(
            builder=builder,
            task_dim=32,
            agent_dim=32,
            graph_dim=32,
            trace_dim=32 + 13,
            fly_dim=8,
            produced_dim=6,
            hidden_dim=32,
        )
        # Important: skip init_from_fly_graph to keep this baseline
        # honest — that's the whole point of #5.
        return ctrl, empty_graph(agent_names)

    return factory


def _flybrain_prior_untrained() -> BaselineFactory:
    """#6 FlyBrain prior without training."""

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.controller import FlyBrainGNNController
        from flybrain.embeddings import (
            AgentEmbedder,
            AgentGraphEmbedder,
            ControllerStateBuilder,
            FlyGraphEmbedder,
            MockEmbeddingClient,
            TaskEmbedder,
            TraceEmbedder,
        )

        client = MockEmbeddingClient(output_dim=32)
        agents_emb = AgentEmbedder(client)
        from flybrain.agents.specs import MINIMAL_15

        agents_emb.precompute_sync(MINIMAL_15)
        builder = ControllerStateBuilder(
            task=TaskEmbedder(client),
            agents=agents_emb,
            trace=TraceEmbedder(client),
            fly=FlyGraphEmbedder(dim=8),
            agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
        )
        ctrl = FlyBrainGNNController(
            builder=builder,
            task_dim=32,
            agent_dim=32,
            graph_dim=32,
            trace_dim=32 + 13,
            fly_dim=8,
            produced_dim=6,
            hidden_dim=32,
        )
        return ctrl, empty_graph(agent_names)

    return factory


def _flybrain_with_checkpoint(
    controller_name: str,
    label: str,
) -> BaselineFactory:
    """Generic factory for #7-#9: same architecture as #6 but with a
    pre-loaded checkpoint produced by Phase-6 / Phase-7 / Phase-8.

    The checkpoint path comes from the ``FLYBRAIN_BASELINE_<LABEL>``
    env var (or stays unloaded if the var is missing). This way
    ``run_baselines.py`` can be invoked without the checkpoints
    physically present and the baselines just degrade to the
    untrained variant.
    """
    import os

    env_var = f"FLYBRAIN_BASELINE_{label.upper()}"

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.embeddings import (
            AgentEmbedder,
            AgentGraphEmbedder,
            ControllerStateBuilder,
            FlyGraphEmbedder,
            MockEmbeddingClient,
            TaskEmbedder,
            TraceEmbedder,
        )

        client = MockEmbeddingClient(output_dim=32)
        agents_emb = AgentEmbedder(client)
        from flybrain.agents.specs import MINIMAL_15

        agents_emb.precompute_sync(MINIMAL_15)
        builder = ControllerStateBuilder(
            task=TaskEmbedder(client),
            agents=agents_emb,
            trace=TraceEmbedder(client),
            fly=FlyGraphEmbedder(dim=8),
            agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
        )
        if controller_name == "gnn":
            from flybrain.controller import FlyBrainGNNController

            ctrl_cls: Any = FlyBrainGNNController
        elif controller_name == "rnn":
            from flybrain.controller import FlyBrainRNNController

            ctrl_cls = FlyBrainRNNController
        elif controller_name == "router":
            from flybrain.controller import LearnedRouterController

            ctrl_cls = LearnedRouterController
        else:
            raise ValueError(f"unknown controller {controller_name!r}")

        ctrl = ctrl_cls(
            builder=builder,
            task_dim=32,
            agent_dim=32,
            graph_dim=32,
            trace_dim=32 + 13,
            fly_dim=8,
            produced_dim=6,
            hidden_dim=32,
        )

        ckpt_path = os.environ.get(env_var)
        if ckpt_path and Path(ckpt_path).exists():
            try:
                import torch

                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                sd = state.get("state_dict", state)
                ctrl.load_state_dict(sd, strict=False)
            except Exception as e:  # pragma: no cover - best effort
                import warnings

                warnings.warn(
                    f"failed to load {label} checkpoint at {ckpt_path}: {e}",
                    stacklevel=2,
                )
        return ctrl, empty_graph(agent_names)

    return factory


# -- registry ------------------------------------------------------------------


def builtin_baselines() -> list[BaselineSpec]:
    """The canonical 9-baseline list from README §15."""
    return [
        BaselineSpec(
            name="manual_graph",
            description="#1 Manual MAS graph + ManualController plan.",
            factory=_manual_baseline,
            tags=["static-graph", "no-llm-controller"],
        ),
        BaselineSpec(
            name="fully_connected",
            description="#2 Fully connected broadcast graph + ManualController.",
            factory=_fully_connected_baseline,
            tags=["static-graph"],
        ),
        BaselineSpec(
            name="random_sparse",
            description="#3 Erdos-Renyi sparse graph + RandomController.",
            factory=_random_sparse_baseline(),
            tags=["static-graph", "random"],
        ),
        BaselineSpec(
            name="degree_preserving",
            description=(
                "#4 Random graph with each node's out-degree fixed; ManualController on top."
            ),
            factory=_degree_preserving_baseline(),
            tags=["static-graph", "random"],
        ),
        BaselineSpec(
            name="learned_router_no_prior",
            description=("#5 Phase-5 LearnedRouter without `init_from_fly_graph`."),
            factory=_learned_router_no_prior(),
            tags=["learned", "untrained", "no-fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_prior_untrained",
            description="#6 Phase-5 GNN with fly-prior init but no training.",
            factory=_flybrain_prior_untrained(),
            tags=["learned", "untrained", "fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_sim_pretrain",
            description="#7 FlyBrain GNN + Phase-6 simulation pretraining.",
            factory=_flybrain_with_checkpoint("gnn", "SIM_PRETRAIN"),
            tags=["learned", "trained", "fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_imitation",
            description="#8 FlyBrain GNN + Phase-7 imitation learning.",
            factory=_flybrain_with_checkpoint("gnn", "IMITATION"),
            tags=["learned", "trained", "fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_rl",
            description="#9 FlyBrain GNN + Phase-8 RL/bandit finetuning.",
            factory=_flybrain_with_checkpoint("gnn", "RL"),
            tags=["learned", "trained", "fly-prior"],
        ),
    ]


BUILTIN_SUITES: dict[str, list[str]] = {
    # PLAN.md §605: `--suite full_min` should run all 9.
    "full_min": [
        "manual_graph",
        "fully_connected",
        "random_sparse",
        "degree_preserving",
        "learned_router_no_prior",
        "flybrain_prior_untrained",
        "flybrain_sim_pretrain",
        "flybrain_imitation",
        "flybrain_rl",
    ],
    "static": [
        "manual_graph",
        "fully_connected",
        "random_sparse",
        "degree_preserving",
    ],
    "learned": [
        "learned_router_no_prior",
        "flybrain_prior_untrained",
        "flybrain_sim_pretrain",
        "flybrain_imitation",
        "flybrain_rl",
    ],
    "smoke": ["manual_graph", "random_sparse"],
}


def list_baselines(
    suite: str = "full_min",
    *,
    extra: list[BaselineSpec] | None = None,
) -> list[BaselineSpec]:
    """Materialise a suite into a list of `BaselineSpec`s.

    Pass ``extra`` to inject custom baselines (e.g. an ablation under
    test) without modifying the registry."""
    if suite not in BUILTIN_SUITES:
        raise KeyError(f"unknown suite {suite!r}; choose one of {sorted(BUILTIN_SUITES)}")
    by_name = {b.name: b for b in builtin_baselines()}
    out = [by_name[n] for n in BUILTIN_SUITES[suite] if n in by_name]
    if extra:
        out.extend(extra)
    return out


__all__ = [
    "BUILTIN_SUITES",
    "BaselineFactory",
    "BaselineSpec",
    "builtin_baselines",
    "list_baselines",
]
