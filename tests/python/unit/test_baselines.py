"""Phase-9 baseline registry + graph-builder smoke tests."""

from __future__ import annotations

import pytest

from flybrain.agents.specs import MINIMAL_15
from flybrain.baselines import (
    BUILTIN_SUITES,
    RoundRobinController,
    builtin_baselines,
    degree_preserving_random_graph,
    empty_graph,
    fully_connected_graph,
    list_baselines,
    random_sparse_graph,
)
from flybrain.runtime.state import RuntimeState

AGENT_NAMES = [a.name for a in MINIMAL_15]


# -- graph builders ------------------------------------------------------------


def test_empty_graph_has_no_edges() -> None:
    g = empty_graph(AGENT_NAMES)
    assert g["nodes"] == AGENT_NAMES
    assert g["edges"] == {}


def test_fully_connected_graph_excludes_self_loops() -> None:
    g = fully_connected_graph(AGENT_NAMES)
    n = len(AGENT_NAMES)
    # Each source has exactly n-1 outgoing edges.
    for src, edges in g["edges"].items():
        assert src not in edges
        assert len(edges) == n - 1


def test_random_sparse_graph_is_deterministic_for_seed() -> None:
    a = random_sparse_graph(AGENT_NAMES, edge_prob=0.3, seed=42)
    b = random_sparse_graph(AGENT_NAMES, edge_prob=0.3, seed=42)
    c = random_sparse_graph(AGENT_NAMES, edge_prob=0.3, seed=43)
    assert a == b
    assert a != c


def test_degree_preserving_respects_target_degree() -> None:
    g = degree_preserving_random_graph(AGENT_NAMES, target_out_degree=3, seed=0)
    for src, edges in g["edges"].items():
        assert src not in edges
        assert len(edges) <= 3


def test_degree_preserving_uses_fly_adjacency() -> None:
    fly_adj = {
        AGENT_NAMES[0]: AGENT_NAMES[1:4],
        AGENT_NAMES[1]: [AGENT_NAMES[2]],
    }
    g = degree_preserving_random_graph(
        AGENT_NAMES, fly_adjacency=fly_adj, target_out_degree=99, seed=0
    )
    assert len(g["edges"][AGENT_NAMES[0]]) == 3
    assert len(g["edges"][AGENT_NAMES[1]]) == 1


# -- registry ------------------------------------------------------------------


def test_builtin_baselines_yields_nine_specs() -> None:
    specs = builtin_baselines()
    assert len(specs) == 9
    # Canonical README §15 order is preserved.
    names = [s.name for s in specs]
    assert names == BUILTIN_SUITES["full_min"]


@pytest.mark.parametrize("suite_name", sorted(BUILTIN_SUITES))
def test_list_baselines_matches_suite(suite_name: str) -> None:
    specs = list_baselines(suite_name)
    assert [s.name for s in specs] == BUILTIN_SUITES[suite_name]


def test_list_baselines_extra_appended() -> None:
    from flybrain.baselines import BaselineSpec

    extra = BaselineSpec(name="ablation_x", description="-", factory=lambda _: (None, None))  # type: ignore[arg-type]
    out = list_baselines("smoke", extra=[extra])
    assert out[-1].name == "ablation_x"


def test_list_baselines_rejects_unknown_suite() -> None:
    with pytest.raises(KeyError):
        list_baselines("not_a_real_suite")


@pytest.mark.parametrize("name", BUILTIN_SUITES["static"])
def test_static_baselines_construct(name: str) -> None:
    """Static-graph baselines must instantiate without torch / Yandex."""
    spec = next(s for s in builtin_baselines() if s.name == name)
    ctrl, graph = spec.factory(AGENT_NAMES)
    assert ctrl is not None
    assert graph is not None
    assert "nodes" in graph
    assert "edges" in graph


# -- round-robin controller ----------------------------------------------------


def _state(step_id: int = 0, last_active: str | None = None) -> RuntimeState:
    return RuntimeState(
        task_id="t1",
        task_type="coding",
        prompt="x",
        step_id=step_id,
        available_agents=AGENT_NAMES,
        pending_inbox={},
        last_active_agent=last_active,
    )


def test_round_robin_cycles_then_verifies_then_terminates() -> None:
    ctrl = RoundRobinController()
    actions = []
    for i in range(len(AGENT_NAMES) + 2):
        actions.append(ctrl.select_action(_state(step_id=i)))
    # First N actions activate each agent in order.
    for i, name in enumerate(AGENT_NAMES):
        assert actions[i] == {"kind": "activate_agent", "agent": name}
    # Then a verifier call, then terminate.
    assert actions[-2] == {"kind": "call_verifier"}
    assert actions[-1] == {"kind": "terminate"}


def test_round_robin_resets_on_new_task() -> None:
    ctrl = RoundRobinController()
    ctrl.select_action(_state(step_id=0))
    ctrl.select_action(_state(step_id=1))
    # New task: the cursor must reset.
    new = RuntimeState(
        task_id="t2",
        task_type="coding",
        prompt="y",
        step_id=0,
        available_agents=AGENT_NAMES,
        pending_inbox={},
        last_active_agent=None,
    )
    assert ctrl.select_action(new) == {
        "kind": "activate_agent",
        "agent": AGENT_NAMES[0],
    }
