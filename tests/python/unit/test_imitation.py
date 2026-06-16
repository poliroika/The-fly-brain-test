"""Phase-7 smoke tests — expert-trace dataset loader + imitation
training loop.

The tests use a hand-built trace fixture so we don't need YandexGPT
in CI. The end-to-end "collect via real LLM" path is exercised by the
``test_collect_expert_traces_dry_run`` test which uses the
deterministic ``MockLLMClient``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flybrain.agents.specs import MINIMAL_15  # noqa: E402
from flybrain.controller import (  # noqa: E402
    FlyBrainGNNController,
    FlyBrainRNNController,
    LearnedRouterController,
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
from flybrain.training import (  # noqa: E402
    ImitationConfig,
    collect_examples,
    imitation_train,
    iter_traces,
    load_trace,
    trace_to_examples,
)

EMB_DIM = 32
GRAPH_DIM = 32
FLY_DIM = 8
PRODUCED_DIM = 6


def _builder() -> ControllerStateBuilder:
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


# -- trace fixture --------------------------------------------------------------


def _make_synthetic_trace(task_id: str, task_type: str) -> dict:
    """Build a minimal trace JSON exercising every action kind we
    care about for imitation: activate_agent, add_edge, call_verifier,
    terminate."""
    return {
        "task_id": task_id,
        "task_type": task_type,
        "final_answer": "42",
        "verification": {"passed": True, "score": 0.92, "errors": []},
        "totals": {"tokens_in": 120, "tokens_out": 80, "cost_rub": 0.5},
        "steps": [
            {
                "step_id": 0,
                "active_agent": None,
                "graph_action": {"kind": "activate_agent", "agent": "Planner"},
                "verifier_score": None,
                "current_graph_hash": "h0",
            },
            {
                "step_id": 1,
                "active_agent": "Planner",
                "output_summary": "ok",
                "graph_action": {"kind": "activate_agent", "agent": "Coder"},
                "verifier_score": None,
                "current_graph_hash": "h1",
            },
            {
                "step_id": 2,
                "active_agent": "Coder",
                "graph_action": {
                    "kind": "add_edge",
                    "from": "Coder",
                    "to": "TestRunner",
                    "weight": 0.7,
                },
                "verifier_score": None,
                "current_graph_hash": "h2",
            },
            {
                "step_id": 3,
                "active_agent": None,
                "graph_action": {"kind": "call_verifier"},
                "verifier_score": 0.88,
                "current_graph_hash": "h3",
            },
            {
                "step_id": 4,
                "active_agent": None,
                "graph_action": {"kind": "terminate"},
                "verifier_score": 0.92,
                "current_graph_hash": "h4",
            },
        ],
    }


# -- dataset loader -------------------------------------------------------------


def test_load_and_iter_traces(tmp_path: Path) -> None:
    t1 = _make_synthetic_trace("t1", "coding")
    t2 = _make_synthetic_trace("t2", "math")
    (tmp_path / "t1.trace.json").write_text(json.dumps(t1))
    (tmp_path / "t2.trace.json").write_text(json.dumps(t2))

    one = load_trace(tmp_path / "t1.trace.json")
    assert one.task_id == "t1"
    assert one.task_type == "coding"
    assert one.verification_passed is True
    assert len(one.steps) == 5

    found = list(iter_traces(tmp_path))
    assert {t.task_id for t in found} == {"t1", "t2"}


def test_iter_traces_handles_corrupt_files(tmp_path: Path) -> None:
    (tmp_path / "good.trace.json").write_text(json.dumps(_make_synthetic_trace("g", "math")))
    (tmp_path / "broken.trace.json").write_text("{not json")
    found = list(iter_traces(tmp_path))
    assert {t.task_id for t in found} == {"g"}


def test_trace_to_examples_extracts_each_action() -> None:
    agent_names = [a.name for a in MINIMAL_15]
    trace = load_trace_dict(_make_synthetic_trace("tx", "coding"))
    examples = trace_to_examples(trace, agent_names=agent_names)
    # 2 activate_agent + 1 add_edge + 1 call_verifier + 1 terminate.
    assert len(examples) == 5
    # First two: activate_agent labels point at Planner / Coder.
    assert examples[0].label_kind == 0  # KIND_ACTIVATE_AGENT
    assert agent_names[examples[0].label_agent] == "Planner"
    assert examples[1].label_kind == 0
    assert agent_names[examples[1].label_agent] == "Coder"
    # Third: add_edge with from=Coder, to=TestRunner.
    assert examples[2].label_kind == 1  # KIND_ADD_EDGE
    assert agent_names[examples[2].label_edge_from] == "Coder"
    assert agent_names[examples[2].label_edge_to] == "TestRunner"
    # Verifier + terminate steps.
    assert examples[3].label_kind == 7  # KIND_CALL_VERIFIER
    assert examples[3].aux_target == pytest.approx(0.88)
    assert examples[4].label_kind == 8  # KIND_TERMINATE


def test_collect_examples_filters_failed_traces() -> None:
    agent_names = [a.name for a in MINIMAL_15]
    good = load_trace_dict(_make_synthetic_trace("g", "coding"))
    bad = load_trace_dict(_make_synthetic_trace("b", "coding"))
    bad.verification_passed = False

    keep_only = collect_examples([good, bad], agent_names=agent_names, only_passed=True)
    keep_all = collect_examples([good, bad], agent_names=agent_names, only_passed=False)
    assert 0 < len(keep_only) < len(keep_all)


def test_trace_to_examples_drops_unknown_agents() -> None:
    """If the expert references an agent we don't have a slot for,
    the example must be dropped (otherwise label index is invalid)."""
    trace_dict = _make_synthetic_trace("tx", "coding")
    # Expert activates an unknown agent.
    trace_dict["steps"][0]["graph_action"] = {
        "kind": "activate_agent",
        "agent": "NotARealAgent",
    }
    trace = load_trace_dict(trace_dict)
    examples = trace_to_examples(trace, agent_names=[a.name for a in MINIMAL_15])
    # Step 0 dropped: 4 examples instead of 5.
    assert len(examples) == 4


# -- imitation training loop ----------------------------------------------------


@pytest.mark.parametrize(
    "ctrl_cls",
    [FlyBrainGNNController, FlyBrainRNNController, LearnedRouterController],
)
def test_imitation_train_reduces_loss(ctrl_cls, tmp_path: Path) -> None:
    """Each controller variant should be trainable on a small fixture
    of synthetic traces; the loss should drop across epochs."""
    # Write 4 traces to disk so the loader path is exercised.
    for i in range(4):
        (tmp_path / f"t{i}.trace.json").write_text(
            json.dumps(_make_synthetic_trace(f"t{i}", "coding"))
        )

    builder = _builder()
    ctrl = ctrl_cls(builder=builder, **_kwargs())

    cfg = ImitationConfig(epochs=3, batch_size=4, learning_rate=3e-3, only_passed=True)
    res = imitation_train(
        ctrl,
        traces_dir=tmp_path,
        agent_names=[a.name for a in MINIMAL_15],
        config=cfg,
    )

    assert res.num_examples > 0
    assert res.num_train > 0
    assert res.num_eval > 0
    assert len(res.losses) > 0
    head = sum(res.losses[: max(1, len(res.losses) // 4)]) / max(1, len(res.losses) // 4)
    tail = sum(res.losses[-max(1, len(res.losses) // 4) :]) / max(1, len(res.losses) // 4)
    assert tail < head


def test_imitation_train_handles_empty_dataset(tmp_path: Path) -> None:
    builder = _builder()
    ctrl = FlyBrainGNNController(builder=builder, **_kwargs())
    res = imitation_train(
        ctrl,
        traces_dir=tmp_path,  # empty
        agent_names=[a.name for a in MINIMAL_15],
        config=ImitationConfig(epochs=1),
    )
    assert res.num_examples == 0
    assert res.losses == []


# -- helpers --------------------------------------------------------------------


def load_trace_dict(d: dict):  # type: ignore[no-untyped-def]
    """Build a TraceFile in-memory from a dict (used so we don't have
    to write to disk for every test case)."""
    from flybrain.training.expert_dataset import TraceFile

    verification = d.get("verification") or {}
    return TraceFile(
        path=Path("/in-memory"),
        task_id=str(d.get("task_id", "x")),
        task_type=str(d.get("task_type", "coding")),
        final_answer=d.get("final_answer"),
        verification_passed=bool(verification.get("passed", False)),
        verification_score=float(verification.get("score", 0.0)),
        totals=dict(d.get("totals") or {}),
        steps=list(d.get("steps") or []),
    )
