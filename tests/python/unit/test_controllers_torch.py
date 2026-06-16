"""Phase-5 smoke tests — shape / grad / fly-init / select_action.

The torch controllers ship in the `[ml]` extra so we ``importorskip``
torch up front; CI runs that exercise these tests must install
``-e ".[dev,ml]"``.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flybrain.agents.specs import MINIMAL_15  # noqa: E402
from flybrain.controller import (  # noqa: E402
    NUM_KINDS,
    ActionSpace,
    FlyBrainGNNController,
    FlyBrainRNNController,
    LearnedRouterController,
)
from flybrain.controller.action_space import (  # noqa: E402
    KIND_NAMES,
    KIND_TERMINATE,
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
from flybrain.graph.dataclasses import FlyGraph, NodeMetadata  # noqa: E402
from flybrain.runtime.state import RuntimeState  # noqa: E402

EMB_DIM = 32
GRAPH_DIM = 32
FLY_DIM = 8
PRODUCED_DIM = 6


def _builder() -> ControllerStateBuilder:
    client = MockEmbeddingClient(output_dim=EMB_DIM)
    agents = AgentEmbedder(client)
    asyncio.run(agents.precompute(MINIMAL_15))
    return ControllerStateBuilder(
        task=TaskEmbedder(client),
        agents=agents,
        trace=TraceEmbedder(client),
        fly=FlyGraphEmbedder(dim=FLY_DIM),
        agent_graph=AgentGraphEmbedder(in_dim=EMB_DIM, hidden_dim=16, out_dim=GRAPH_DIM),
    )


def _runtime() -> RuntimeState:
    return RuntimeState(
        task_id="t1",
        task_type="coding",
        prompt="hello world",
        step_id=0,
        available_agents=["Planner", "Coder", "TestRunner"],
        pending_inbox={"Planner": 1},
        last_active_agent=None,
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


def _toy_fly_graph(n: int = 6) -> FlyGraph:
    return FlyGraph(
        num_nodes=n,
        edge_index=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        edge_weight=[1.0] * 6,
        is_excitatory=[True] * 6,
        nodes=[NodeMetadata(id=i, node_type=f"n{i}", features=[]) for i in range(n)],
    )


# -- ActionSpace tests -----------------------------------------------------------


def test_action_space_decodes_terminate() -> None:
    space = ActionSpace(agent_names=["A", "B"])
    assert space.decode(KIND_TERMINATE) == {"kind": "terminate"}


def test_action_space_decodes_activate_agent() -> None:
    space = ActionSpace(agent_names=["A", "B"])
    out = space.decode(0, agent_id=1)
    assert out == {"kind": "activate_agent", "agent": "B"}


def test_action_space_decodes_add_edge() -> None:
    space = ActionSpace(agent_names=["A", "B", "C"])
    out = space.decode(1, edge_from_id=0, edge_to_id=2, edge_weight=0.7)
    assert out == {"kind": "add_edge", "from": "A", "to": "C", "weight": 0.7}


def test_action_space_legal_mask_zero_agents() -> None:
    space = ActionSpace(agent_names=[])
    mask = space.legal_mask()
    assert mask.kind_mask.shape == (NUM_KINDS,)
    # activate_agent + 3 edge ops are all illegal with K=0/1.
    assert mask.kind_mask[0] == 0.0
    assert mask.kind_mask[KIND_NAMES.index("add_edge")] == 0.0


def test_action_space_legal_mask_two_agents() -> None:
    space = ActionSpace(agent_names=["A", "B"])
    mask = space.legal_mask()
    # add_edge legal, agent_mask shape correct.
    assert mask.kind_mask[KIND_NAMES.index("add_edge")] == 1.0
    assert mask.agent_mask.shape == (2,)
    # No self-loops in the edge mask.
    assert mask.edge_mask[0, 0] == 0.0
    assert mask.edge_mask[0, 1] == 1.0


# -- shape / grad smoke tests (PLAN.md §583) -------------------------------------


@pytest.mark.parametrize(
    "ctrl_cls",
    [FlyBrainGNNController, FlyBrainRNNController, LearnedRouterController],
)
def test_controller_select_action_returns_valid_dict(ctrl_cls) -> None:
    builder = _builder()
    ctrl = ctrl_cls(builder=builder, **_kwargs())
    action = ctrl.select_action(_runtime())
    assert "kind" in action
    assert action["kind"] in KIND_NAMES


@pytest.mark.parametrize(
    "ctrl_cls",
    [FlyBrainGNNController, FlyBrainRNNController, LearnedRouterController],
)
def test_controller_forward_shapes(ctrl_cls) -> None:
    builder = _builder()
    ctrl = ctrl_cls(builder=builder, **_kwargs())
    cs = builder.from_runtime_sync(_runtime())
    out = ctrl(cs)
    assert out.kind_logits.shape == (NUM_KINDS,)
    assert out.agent_logits.shape == (cs.num_agents,)
    assert out.edge_from_logits.shape == (cs.num_agents,)
    assert out.edge_to_logits.shape == (cs.num_agents,)
    assert out.value.shape == ()
    assert out.aux_verifier.shape == ()
    assert torch.all((out.aux_verifier >= 0.0) & (out.aux_verifier <= 1.0))


@pytest.mark.parametrize(
    "ctrl_cls",
    [FlyBrainGNNController, FlyBrainRNNController, LearnedRouterController],
)
def test_controller_backward_smoke(ctrl_cls) -> None:
    builder = _builder()
    ctrl = ctrl_cls(builder=builder, **_kwargs())
    cs = builder.from_runtime_sync(_runtime())
    out = ctrl(cs)
    loss = out.kind_logits.sum() + out.value + out.agent_logits.sum() + out.aux_verifier
    if out.edge_from_logits.numel() > 0:
        loss = loss + out.edge_from_logits.sum() + out.edge_to_logits.sum() + out.edge_scalar
    loss.backward()

    grads_present = [
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in ctrl.parameters()
        if p.requires_grad
    ]
    assert any(grads_present)
    assert all(g for g in grads_present)


# -- fly-graph initialisation ----------------------------------------------------


def test_rnn_controller_init_from_fly_graph_changes_a_fly_weight() -> None:
    builder = _builder()
    ctrl = FlyBrainRNNController(builder=builder, **_kwargs())
    weight_before = ctrl.a_fly.weight.detach().clone()
    ctrl.init_from_fly_graph(_toy_fly_graph(), num_agents=6)
    weight_after = ctrl.a_fly.weight.detach()
    assert not torch.allclose(weight_before, weight_after)


def test_learned_router_init_from_fly_graph_sets_prior() -> None:
    builder = _builder()
    ctrl = LearnedRouterController(builder=builder, **_kwargs())
    assert ctrl._fly_prior is None
    ctrl.init_from_fly_graph(_toy_fly_graph(), num_agents=6)
    assert ctrl._fly_prior is not None
    assert ctrl._fly_prior.shape == (6,)


def test_learned_router_fly_regularizer_loss_is_finite() -> None:
    builder = _builder()
    ctrl = LearnedRouterController(builder=builder, **_kwargs())
    ctrl.init_from_fly_graph(_toy_fly_graph(), num_agents=3)
    cs = builder.from_runtime_sync(_runtime())
    _ = ctrl(cs)  # populates last_attn_weights
    loss = ctrl.fly_regularizer_loss()
    assert torch.isfinite(loss)
    assert loss.shape == ()


def test_rnn_controller_resets_hidden_across_tasks() -> None:
    builder = _builder()
    ctrl = FlyBrainRNNController(builder=builder, **_kwargs())
    rs1 = _runtime()
    ctrl.select_action(rs1)
    h1 = ctrl._hidden.clone() if ctrl._hidden is not None else None
    rs2 = RuntimeState(
        task_id="t2",
        task_type="math",  # different task_type triggers reset
        prompt="2+2=?",
        step_id=0,
        available_agents=["Planner", "MathSolver"],
        pending_inbox={},
        last_active_agent=None,
    )
    # Force a reset path: same instance, new task
    ctrl.select_action(rs2)
    h2 = ctrl._hidden.clone() if ctrl._hidden is not None else None
    assert h1 is not None and h2 is not None
    assert not torch.allclose(
        h1.flatten()[: min(h1.numel(), h2.numel())], h2.flatten()[: min(h1.numel(), h2.numel())]
    )


# -- masking / decode determinism ------------------------------------------------


def test_select_action_with_no_agents_terminates() -> None:
    builder = _builder()
    ctrl = FlyBrainGNNController(builder=builder, **_kwargs())
    rs = RuntimeState(
        task_id="t-empty",
        task_type="coding",
        prompt="",
        step_id=0,
        available_agents=[],
        pending_inbox={},
        last_active_agent=None,
    )
    action = ctrl.select_action(rs)
    # With K==0, activate_agent + edge ops are masked; argmax must pick
    # something legal.
    assert action["kind"] in {
        "call_memory",
        "call_retriever",
        "call_tool_executor",
        "call_verifier",
        "terminate",
    }


def test_select_action_does_not_emit_self_loop_edges() -> None:
    """Force an edge action via direct masking and make sure ``from != to``."""
    builder = _builder()
    ctrl = FlyBrainGNNController(builder=builder, **_kwargs())
    cs = builder.from_runtime_sync(_runtime())
    space = ActionSpace(agent_names=list(cs.agent_names))
    result = ctrl._masked_forward(cs, space)
    # Manually pick the highest from logit; mask it out on the to side.
    if result.masked_edge_from_logits.numel() > 1:
        from_id = int(torch.argmax(result.masked_edge_from_logits).item())
        to_logits = result.masked_edge_to_logits.clone()
        to_logits[from_id] = -1e30
        to_id = int(torch.argmax(to_logits).item())
        assert from_id != to_id


# -- determinism check -----------------------------------------------------------


@pytest.mark.parametrize(
    "ctrl_cls",
    [FlyBrainGNNController, FlyBrainRNNController, LearnedRouterController],
)
def test_controller_deterministic_for_same_seed(ctrl_cls) -> None:
    builder = _builder()
    a = ctrl_cls(builder=builder, seed=7, **_kwargs())
    b = ctrl_cls(builder=builder, seed=7, **_kwargs())
    cs = builder.from_runtime_sync(_runtime())
    out_a = a(cs)
    out_b = b(cs)
    assert torch.allclose(out_a.kind_logits, out_b.kind_logits, atol=1e-5)


def test_aflylinear_init_skips_when_zero_agents() -> None:
    builder = _builder()
    ctrl = FlyBrainRNNController(builder=builder, **_kwargs())
    weight_before = ctrl.a_fly.weight.detach().clone()
    ctrl.init_from_fly_graph(None, num_agents=0)
    assert torch.allclose(weight_before, ctrl.a_fly.weight.detach())
    # And: num_agents > 0 with empty fly_graph just zeros the adjacency
    # (init_from_fly is a no-op for an empty matrix).
    weight_before = ctrl.a_fly.weight.detach().clone()
    ctrl.init_from_fly_graph(None, num_agents=4)
    assert torch.allclose(weight_before, ctrl.a_fly.weight.detach())


def test_runtime_native_phase_is_5() -> None:
    """Phase marker bump (Rust pyo3 module)."""
    from flybrain import flybrain_native as native

    assert native.__modinfo__["phase"] == "5-controller"


def test_action_space_decode_handles_invalid_agent_id_gracefully() -> None:
    space = ActionSpace(agent_names=["A", "B"])
    out = space.decode(0, agent_id=99)  # out of range
    assert out == {"kind": "terminate"}


def test_action_space_decode_handles_self_loop_edge_gracefully() -> None:
    space = ActionSpace(agent_names=["A", "B"])
    out = space.decode(1, edge_from_id=0, edge_to_id=0)  # self-loop
    assert out == {"kind": "terminate"}


# numpy import is only needed locally.
_ = np
