"""Smoke tests for the Phase-2 PyO3 runtime classes.

Mirrors the Phase-1 `test_native_bindings.py` style: stand the JSON
boundary up, exercise each method, assert state and return shapes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

native = pytest.importorskip("flybrain.flybrain_native")


def make_graph() -> dict:
    return {
        "nodes": ["Planner", "Coder", "Verifier", "Finalizer"],
        "edges": {},
    }


def test_modinfo_advertises_phase_5() -> None:
    assert native.__modinfo__["phase"] == "5-controller"


def test_scheduler_activate_known_agent() -> None:
    s = native.Scheduler(make_graph())
    out = s.apply({"kind": "activate_agent", "agent": "Planner"})
    assert out == {"kind": "run_agent", "agent": "Planner"}
    assert s.last_active_agent == "Planner"
    assert s.is_terminated is False
    assert s.step_id == 0


def test_scheduler_advance_step_increments() -> None:
    s = native.Scheduler(make_graph())
    s.advance_step()
    s.advance_step()
    assert s.step_id == 2


def test_scheduler_terminate_blocks_apply() -> None:
    s = native.Scheduler(make_graph())
    s.apply({"kind": "terminate"})
    assert s.is_terminated is True
    with pytest.raises(ValueError):
        s.apply({"kind": "call_verifier"})


def test_scheduler_add_edge_changes_hash() -> None:
    s = native.Scheduler(make_graph())
    h0 = s.current_graph_hash
    out = s.apply({"kind": "add_edge", "from": "Planner", "to": "Coder", "weight": 1.0})
    assert out == {"kind": "graph_mutation"}
    assert s.current_graph_hash != h0


def test_scheduler_unknown_agent_is_value_error() -> None:
    s = native.Scheduler(make_graph())
    with pytest.raises(ValueError):
        s.apply({"kind": "activate_agent", "agent": "Ghost"})


def test_scheduler_agent_graph_round_trip() -> None:
    s = native.Scheduler(make_graph())
    out = s.agent_graph()
    assert "Planner" in out["nodes"]
    s.apply({"kind": "add_edge", "from": "Planner", "to": "Coder", "weight": 0.5})
    out2 = s.agent_graph()
    assert out2["edges"]["Planner"]["Coder"] == pytest.approx(0.5)


def test_message_bus_round_trip() -> None:
    bus = native.MessageBus()
    mid = bus.send("Planner", "Coder", {"text": "hi"}, 0)
    assert mid == 0
    msg = bus.pop("Coder")
    assert msg["sender"] == "Planner"
    assert msg["recipient"] == "Coder"
    assert msg["content"] == {"text": "hi"}
    assert bus.pending("Coder") == 0
    assert bus.total() == 1


def test_message_bus_pop_empty_returns_none() -> None:
    bus = native.MessageBus()
    assert bus.pop("Coder") is None


def test_trace_writer_records_and_finalises(tmp_path: Path) -> None:
    sink = tmp_path / "traces" / "t1.steps.jsonl"
    w = native.TraceWriter("t1", "coding", str(sink))
    step = {
        "step_id": 0,
        "t_unix_ms": 0,
        "active_agent": "Coder",
        "input_msg_id": None,
        "output_summary": "wrote code",
        "tool_calls": [],
        "errors": [],
        "tokens_in": 10,
        "tokens_out": 5,
        "latency_ms": 1,
        "verifier_score": None,
        "current_graph_hash": "deadbeef",
        "graph_action": {"kind": "activate_agent", "agent": "Coder"},
        "cost_rub": 0.01,
    }
    w.record_step(step)
    assert w.step_count() == 1

    snap = w.snapshot()
    assert snap["task_id"] == "t1"
    assert snap["task_type"] == "coding"

    trace = w.finalize(final_answer="done", verification=None, metadata={"controller": "manual"})
    assert trace["final_answer"] == "done"
    assert trace["totals"]["llm_calls"] == 1
    assert sink.exists()
    body = sink.read_text().strip().splitlines()
    assert len(body) == 1


def test_trace_writer_finalize_idempotent_check() -> None:
    w = native.TraceWriter("t1", "coding", None)
    w.finalize()
    with pytest.raises(ValueError):
        w.finalize()
