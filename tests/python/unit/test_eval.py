"""Phase-10 eval-pipeline tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from flybrain.eval import (
    AggregateMetrics,
    ReportInputs,
    aggregate,
    csv_table,
    markdown_table,
    metrics_from_trace,
    metrics_from_trace_path,
    render_report,
    write_report,
)
from flybrain.eval.reports import select_cherry_picks


def _trace(*, passed: bool, score: float = 0.85) -> dict:
    return {
        "task_id": "t1",
        "task_type": "coding",
        "steps": [
            {
                "graph_action": {"kind": "activate_agent", "agent": "Coder"},
                "graph_density": 0.2,
            },
            {
                "graph_action": {"kind": "activate_agent", "agent": "TestRunner"},
                "graph_density": 0.3,
            },
        ],
        "totals": {
            "tokens_in": 200,
            "tokens_out": 80,
            "llm_calls": 3,
            "tool_calls": 1,
            "failed_tool_calls": 0,
            "latency_ms": 1500,
            "cost_rub": 0.12,
        },
        "verification": {
            "passed": passed,
            "score": score,
            "failed_component": None if passed else "tests_run",
        },
    }


# -- metrics_from_trace --------------------------------------------------------


def test_metrics_from_trace_extracts_all_fields() -> None:
    m = metrics_from_trace(_trace(passed=True), benchmark="humaneval")
    assert m.task_id == "t1"
    assert m.benchmark == "humaneval"
    assert m.task_type == "coding"
    assert m.success is True
    assert m.tokens_in == 200
    assert m.tokens_out == 80
    assert m.total_tokens == 280
    assert m.llm_calls == 3
    assert m.failed_tool_calls == 0
    assert m.latency_ms == 1500
    assert m.cost_rub == 0.12
    assert m.num_steps == 2
    assert 0.2 <= m.graph_density <= 0.3


def test_metrics_from_trace_handles_missing_keys() -> None:
    m = metrics_from_trace({"task_id": "x"})
    assert m.task_id == "x"
    assert m.success is False
    assert m.total_tokens == 0
    assert m.num_steps == 0
    assert m.graph_density == 0.0


def test_metrics_from_trace_path_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "t.trace.json"
    p.write_text(json.dumps(_trace(passed=True)))
    m = metrics_from_trace_path(p, benchmark="gsm8k")
    assert m.benchmark == "gsm8k"
    assert m.success is True


# -- aggregate -----------------------------------------------------------------


def test_aggregate_averages_correctly() -> None:
    rows = [
        metrics_from_trace(_trace(passed=True, score=0.9), benchmark="humaneval"),
        metrics_from_trace(_trace(passed=False, score=0.4), benchmark="humaneval"),
        metrics_from_trace(_trace(passed=True, score=0.8), benchmark="humaneval"),
    ]
    agg = aggregate(rows, name="manual_graph", benchmark="humaneval")
    assert agg.num_tasks == 3
    assert abs(agg.success_rate - 2 / 3) < 1e-9
    assert abs(agg.verifier_pass_rate - (0.9 + 0.4 + 0.8) / 3) < 1e-9
    assert agg.avg_total_tokens == 280
    # cost_per_solved = total_cost / solved = (0.12 * 3) / 2
    assert abs(agg.cost_per_solved_rub - (0.12 * 3) / 2) < 1e-9


def test_aggregate_handles_empty_input() -> None:
    agg = aggregate([], name="dummy", benchmark="x")
    assert agg.num_tasks == 0
    assert agg.success_rate == 0.0
    assert agg.cost_per_solved_rub == float("inf")


def test_aggregate_zero_solved_returns_inf_cost_per_solved() -> None:
    rows = [
        metrics_from_trace(_trace(passed=False), benchmark="humaneval"),
        metrics_from_trace(_trace(passed=False), benchmark="humaneval"),
    ]
    agg = aggregate(rows, name="dummy", benchmark="humaneval")
    assert agg.success_rate == 0.0
    assert agg.cost_per_solved_rub == float("inf")


# -- tables --------------------------------------------------------------------


def _agg(name: str, *, success: float, cost_per_solved: float) -> AggregateMetrics:
    return AggregateMetrics(
        name=name,
        benchmark="humaneval",
        num_tasks=10,
        success_rate=success,
        verifier_pass_rate=0.7,
        avg_total_tokens=210.0,
        avg_llm_calls=3.0,
        avg_failed_tool_calls=0.5,
        avg_latency_ms=2000.0,
        avg_cost_rub=0.1,
        avg_steps=4.0,
        avg_graph_density=0.25,
        cost_per_solved_rub=cost_per_solved,
    )


def test_markdown_table_renders_header_and_rows() -> None:
    rows = [_agg("manual_graph", success=0.5, cost_per_solved=0.2)]
    md = markdown_table(rows)
    assert "| Method | Benchmark | Tasks |" in md
    assert "| manual_graph | humaneval | 10 |" in md


def test_markdown_table_renders_inf_as_unicode() -> None:
    rows = [_agg("nope", success=0.0, cost_per_solved=float("inf"))]
    md = markdown_table(rows)
    assert "∞" in md


def test_csv_table_includes_header() -> None:
    rows = [_agg("manual_graph", success=0.5, cost_per_solved=0.2)]
    out = csv_table(rows)
    first_line = out.splitlines()[0]
    assert (
        first_line
        == "name,benchmark,num_tasks,success_rate,verifier_pass_rate,avg_total_tokens,avg_llm_calls,avg_latency_ms,avg_cost_rub,cost_per_solved_rub"
    )
    second_line = out.splitlines()[1]
    assert second_line.startswith("manual_graph,humaneval,10,")


# -- reports -------------------------------------------------------------------


def test_render_report_inlines_tables_and_traces() -> None:
    overall = [_agg("manual_graph", success=0.5, cost_per_solved=0.2)]
    per_benchmark = {"humaneval": overall}
    text = render_report(
        ReportInputs(
            suite_name="full_min",
            overall=overall,
            per_benchmark=per_benchmark,
            trace_paths=[Path("data/traces/x.trace.json")],
        )
    )
    assert "Suite: `full_min`" in text
    assert "manual_graph" in text
    assert "data/traces/x.trace.json" in text
    # Phase-11 template surface area.
    assert "## 7. Deliverables checklist" in text
    assert "## 8. Experiments coverage" in text
    assert "## 9. Discussion" in text


def test_render_report_handles_none_cost_per_solved() -> None:
    """Strict-JSON serialisation of `float('inf')` becomes `null` in
    `comparison_overall.json`, which loads back as `None`. The standalone
    `flybrain-py report` path used to crash with `TypeError` from
    `math.isfinite(None)`; the report should now render `∞` instead.
    """
    # Mimic what `AggregateMetrics(**row)` produces after a JSON round-trip
    # on a method with zero solved tasks.
    row = _agg("dud", success=0.0, cost_per_solved=float("inf"))
    row_dict = dataclasses.asdict(row)
    row_dict["cost_per_solved_rub"] = None
    overall = [AggregateMetrics(**row_dict)]
    text = render_report(
        ReportInputs(
            suite_name="full_min",
            overall=overall,
            per_benchmark={"humaneval": overall},
            trace_paths=[],
        )
    )
    assert "∞" in text
    assert "dud" in text


def test_write_report_creates_parents(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "report.md"
    write_report(
        ReportInputs(
            suite_name="x",
            overall=[],
            per_benchmark={},
            trace_paths=[],
        ),
        out,
    )
    assert out.exists()
    assert "Suite: `x`" in out.read_text()


# -- cherry-pick selection -----------------------------------------------------


def _write_trace(
    p: Path, *, passed: bool, score: float, fc: str | None, densities: list[float] | None = None
) -> None:
    densities = densities or [0.2, 0.3]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "task_id": p.stem,
                "task_type": "coding",
                "steps": [
                    {"graph_action": {"kind": "activate_agent", "agent": "A"}, "graph_density": d}
                    for d in densities
                ],
                "totals": {
                    "tokens_in": 1,
                    "tokens_out": 1,
                    "llm_calls": 1,
                    "tool_calls": 0,
                    "failed_tool_calls": 0,
                    "latency_ms": 1,
                    "cost_rub": 0.01,
                },
                "verification": {"passed": passed, "score": score, "failed_component": fc},
            }
        )
    )


def test_select_cherry_picks_returns_solved_failed_and_routing(tmp_path: Path) -> None:
    # Solved trace, high score.
    _write_trace(
        tmp_path / "manual_graph" / "humaneval" / "good.trace.json",
        passed=True,
        score=0.95,
        fc=None,
    )
    # Failed trace with explicit failed_component.
    _write_trace(
        tmp_path / "random_sparse" / "humaneval" / "bad.trace.json",
        passed=False,
        score=0.30,
        fc="tests_run",
    )
    # Routing-interesting trace (wide density spread).
    _write_trace(
        tmp_path / "fully_connected" / "humaneval" / "wild.trace.json",
        passed=True,
        score=0.80,
        fc=None,
        densities=[0.05, 0.95, 0.4, 0.6],
    )

    picks = select_cherry_picks(tmp_path, max_picks=3)
    paths = {p.path.name: p.reason for p in picks}
    assert "good.trace.json" in paths
    assert "bad.trace.json" in paths
    assert "wild.trace.json" in paths
    assert "highest-scoring" in paths["good.trace.json"]
    assert "tests_run" in paths["bad.trace.json"]
    assert "density" in paths["wild.trace.json"].lower()


def test_select_cherry_picks_handles_empty_dir(tmp_path: Path) -> None:
    assert select_cherry_picks(tmp_path) == []
