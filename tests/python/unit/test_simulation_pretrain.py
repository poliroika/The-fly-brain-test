"""Phase-6 smoke tests — task generator + synthetic MAS + supervised
pretraining loop.

The goal here is not to hit the PLAN.md §590 ≥0.85 sim-success metric
(that's the job of the standalone training script which runs for
several minutes); it's to verify the *plumbing* — generator emits
deterministic balanced tasks, synthetic MAS doesn't crash on every
controller, the supervised loop reduces loss across epochs.
"""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from flybrain.agents.specs import MINIMAL_15  # noqa: E402
from flybrain.controller import (  # noqa: E402
    FlyBrainGNNController,
    FlyBrainRNNController,
    LearnedRouterController,
    ManualController,
    RandomController,
)
from flybrain.embeddings import (  # noqa: E402
    AgentEmbedder,
    AgentGraphEmbedder,
    ControllerStateBuilder,
    FlyGraphEmbedder,
    MockEmbeddingClient,
    TaskEmbedder,
    TraceEmbedder,
)
from flybrain.sim import (  # noqa: E402
    OPTIMAL_ROUTES,
    TASK_TYPES,
    SyntheticMAS,
    SyntheticTask,
    TaskGenerator,
    component_for_agent,
    optimal_action_at,
)
from flybrain.training import (  # noqa: E402
    PretrainConfig,
    expert_dataset,
    simulation_pretrain,
)

EMB_DIM = 32
GRAPH_DIM = 32
FLY_DIM = 8
PRODUCED_DIM = 6


def _builder(agents: list[str] | None = None) -> ControllerStateBuilder:
    client = MockEmbeddingClient(output_dim=EMB_DIM)
    agent_emb = AgentEmbedder(client)
    asyncio.run(agent_emb.precompute(MINIMAL_15))
    return ControllerStateBuilder(
        task=TaskEmbedder(client),
        agents=agent_emb,
        trace=TraceEmbedder(client),
        fly=FlyGraphEmbedder(dim=FLY_DIM),
        agent_graph=AgentGraphEmbedder(in_dim=EMB_DIM, hidden_dim=16, out_dim=GRAPH_DIM),
    )


def _kwargs() -> dict:
    return {
        "task_dim": EMB_DIM,
        "agent_dim": EMB_DIM,
        "graph_dim": GRAPH_DIM,
        "trace_dim": EMB_DIM + 13,
        "fly_dim": FLY_DIM,
        "produced_dim": PRODUCED_DIM,
        "hidden_dim": 32,
    }


# -- task generator --------------------------------------------------------------


def test_task_generator_balanced_dataset_size() -> None:
    gen = TaskGenerator(seed=0)
    ds = gen.balanced_dataset(n_per_type=3)
    assert len(ds) == 3 * len(TASK_TYPES)
    by_type: dict[str, int] = {}
    for t in ds:
        assert t.task_type in TASK_TYPES
        by_type[t.task_type] = by_type.get(t.task_type, 0) + 1
    for tt in TASK_TYPES:
        assert by_type[tt] == 3


def test_task_generator_is_deterministic() -> None:
    g1 = TaskGenerator(seed=42).balanced_dataset(n_per_type=2)
    g2 = TaskGenerator(seed=42).balanced_dataset(n_per_type=2)
    assert [t.prompt for t in g1] == [t.prompt for t in g2]
    assert [t.task_id for t in g1] == [t.task_id for t in g2]


def test_optimal_action_at_walks_route_then_terminates() -> None:
    route = OPTIMAL_ROUTES["coding"]
    for step in range(len(route)):
        action = optimal_action_at("coding", step)
        assert action == {"kind": "activate_agent", "agent": route[step]}
    assert optimal_action_at("coding", len(route)) == {"kind": "terminate"}


def test_component_for_agent_falls_back_for_unknown() -> None:
    assert component_for_agent("UnknownAgent") == "plan"
    assert component_for_agent("Verifier") == "verifier_called"


# -- synthetic MAS --------------------------------------------------------------


def test_synthetic_mas_runs_random_controller() -> None:
    gen = TaskGenerator(seed=0)
    task = gen.sample(task_type="coding")
    sim = SyntheticMAS(
        agent_names=[a.name for a in MINIMAL_15],
        seed=42,
        max_steps=8,
    )
    out = sim.run(RandomController(), task)
    assert out.steps <= sim.max_steps
    assert 0.0 <= out.final_score <= 1.0
    assert len(out.actions) == out.steps
    assert len(out.states) == out.steps
    assert all("kind" in a for a in out.actions)


def test_synthetic_mas_runs_manual_controller_and_solves_more() -> None:
    """Manual controller (Phase 2) follows a hand-tuned plan per task
    type, so it should outperform random on the average final score
    across a small balanced dataset."""
    gen = TaskGenerator(seed=0)
    tasks = gen.balanced_dataset(n_per_type=2)
    sim = SyntheticMAS(agent_names=[a.name for a in MINIMAL_15], seed=42, max_steps=12)
    rand_scores = [sim.run(RandomController(), t).final_score for t in tasks]
    manual_scores = [sim.run(ManualController(), t).final_score for t in tasks]
    assert sum(manual_scores) >= sum(rand_scores) - 0.1


# -- expert_dataset --------------------------------------------------------------


def test_expert_dataset_emits_route_plus_terminate() -> None:
    tasks = [SyntheticTask(task_id="t0", task_type="coding", prompt="hi")]
    agent_names = [a.name for a in MINIMAL_15]
    examples = expert_dataset(tasks, agent_names=agent_names)
    route = OPTIMAL_ROUTES["coding"]
    # Every step + a final terminate action.
    assert len(examples) == len(route) + 1
    # Last example is a terminate action (label_kind == 8).
    assert examples[-1].label_kind == 8
    # Earlier examples are activate_agent.
    for ex in examples[:-1]:
        assert ex.label_kind == 0
        # The labelled agent index points back to the right slot.
        assert agent_names[ex.label_agent] in route


# -- supervised pretraining loop ------------------------------------------------


@pytest.mark.parametrize(
    "ctrl_cls",
    [FlyBrainGNNController, FlyBrainRNNController, LearnedRouterController],
)
def test_simulation_pretrain_reduces_loss(ctrl_cls) -> None:
    agent_names = [a.name for a in MINIMAL_15]
    builder = _builder()
    ctrl = ctrl_cls(builder=builder, **_kwargs())

    cfg = PretrainConfig(n_per_type=4, epochs=3, batch_size=8, learning_rate=3e-3)
    res = simulation_pretrain(ctrl, agent_names=agent_names, config=cfg)

    assert res.num_examples > 0
    assert len(res.losses) > 0
    assert len(res.epoch_accuracy) == cfg.epochs
    # Loss across the *averaged* first 25% of steps vs. last 25% should
    # drop — averaging smooths over single-batch outliers.
    n = len(res.losses)
    head = sum(res.losses[: max(1, n // 4)]) / max(1, n // 4)
    tail = sum(res.losses[-max(1, n // 4) :]) / max(1, n // 4)
    assert tail < head


def test_simulation_pretrain_returns_empty_result_for_empty_tasks() -> None:
    builder = _builder()
    ctrl = FlyBrainGNNController(builder=builder, **_kwargs())
    res = simulation_pretrain(
        ctrl,
        agent_names=[a.name for a in MINIMAL_15],
        config=PretrainConfig(n_per_type=0, epochs=1),
        tasks=[],
    )
    assert res.num_examples == 0
    assert res.losses == []


def test_simulation_pretrain_supports_subset_agent_names() -> None:
    """Passing only a subset of agent names should still produce
    meaningful examples (the labels just point into the smaller list).
    """
    subset = ["Planner", "Coder", "TestRunner", "Debugger", "Verifier"]
    gen = TaskGenerator(seed=1)
    tasks = gen.balanced_dataset(n_per_type=1)
    examples = expert_dataset(tasks, agent_names=subset)
    # All coding-route agents are in the subset, so the coding task
    # contributes len(route)+1 examples.
    coding_examples = [e for e in examples if e.runtime_state.task_type == "coding"]
    assert len(coding_examples) == len(OPTIMAL_ROUTES["coding"]) + 1
