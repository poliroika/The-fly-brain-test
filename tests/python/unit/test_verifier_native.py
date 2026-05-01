"""Smoke tests for the Phase-3 PyO3 verifier bindings."""

from __future__ import annotations

from flybrain import flybrain_native as native


def test_schema_check_passes_well_formed_dict() -> None:
    schema = {
        "type": "object",
        "required": ["final_answer"],
        "properties": {"final_answer": {"type": "string", "minLength": 1}},
    }
    r = native.schema_check({"final_answer": "42"}, schema)
    assert r["passed"] is True


def test_schema_check_reports_missing_required_field() -> None:
    schema = {
        "type": "object",
        "required": ["final_answer"],
        "properties": {"final_answer": {"type": "string"}},
    }
    r = native.schema_check({}, schema)
    assert r["passed"] is False
    assert r["failed_component"] == "schema"
    assert any("final_answer" in e for e in r["errors"])


def test_tool_use_check_allow_list_violation() -> None:
    calls = [{"name": "rm_rf", "args": {}}]
    r = native.tool_use_check(calls, allowed=["python_exec"])
    assert r["passed"] is False
    assert r["failed_component"] == "tool_use"


def test_tool_use_check_required_args() -> None:
    calls = [{"name": "python_exec", "args": {}}]
    r = native.tool_use_check(
        calls,
        allowed=["python_exec"],
        requirements={"python_exec": ["code"]},
    )
    assert r["passed"] is False
    assert any("code" in e for e in r["errors"])


def test_tool_use_check_passes_when_no_constraints() -> None:
    r = native.tool_use_check([{"name": "anything", "args": {"x": 1}}])
    assert r["passed"] is True


def test_unit_test_check_pass_and_fail() -> None:
    assert native.unit_test_check({"passed": 3, "failed": 0, "all_passed": True})["passed"] is True
    bad = native.unit_test_check({"passed": 1, "failed": 2, "all_passed": False})
    assert bad["passed"] is False
    assert bad["suggested_next_agent"] == "Debugger"


def test_trace_check_step_id_discontinuity() -> None:
    bad_trace = {
        "task_id": "t1",
        "task_type": "coding",
        "steps": [
            {
                "step_id": 0,
                "active_agent": "A",
                "tokens_in": 0,
                "tokens_out": 0,
                "tool_calls": [],
                "errors": [],
            },
            {
                "step_id": 5,
                "active_agent": "B",
                "tokens_in": 0,
                "tokens_out": 0,
                "tool_calls": [],
                "errors": [],
            },
        ],
        "totals": {
            "tokens_in": 0,
            "tokens_out": 0,
            "llm_calls": 2,
            "tool_calls_total": 0,
            "tool_call_failures": 0,
            "latency_ms": 0,
            "cost_rub": 0.0,
        },
        "final_answer": "x",
    }
    r = native.trace_check(bad_trace)
    assert r["passed"] is False
    assert any("discontinuity" in e for e in r["errors"])


def test_trace_check_passes_well_formed_trace() -> None:
    good = {
        "task_id": "t1",
        "task_type": "math",
        "steps": [
            {
                "step_id": 0,
                "active_agent": "MathSolver",
                "tokens_in": 1,
                "tokens_out": 1,
                "tool_calls": [],
                "errors": [],
            },
        ],
        "totals": {
            "tokens_in": 1,
            "tokens_out": 1,
            "llm_calls": 1,
            "tool_calls_total": 0,
            "tool_call_failures": 0,
            "latency_ms": 0,
            "cost_rub": 0.0,
        },
        "final_answer": "42",
    }
    r = native.trace_check(good)
    assert r["passed"] is True
    assert r["score"] == 1.0


def test_verification_result_round_trip_through_python() -> None:
    payload = {
        "passed": False,
        "score": 0.42,
        "errors": ["err"],
        "warnings": [],
        "failed_component": "schema",
        "suggested_next_agent": "Coder",
        "reward_delta": -0.5,
    }
    out = native.verification_result_round_trip(payload)
    assert out["failed_component"] == "schema"
    # f32 round-trip introduces a tiny rounding error.
    assert abs(out["score"] - 0.42) < 1e-6
