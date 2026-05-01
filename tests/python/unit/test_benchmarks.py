"""Phase-10 benchmark loader + runner smoke tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from flybrain.agents.specs import MINIMAL_15
from flybrain.baselines import (
    RoundRobinController,
    empty_graph,
)
from flybrain.benchmarks import (
    BENCHMARK_REGISTRY,
    DEFAULT_BENCHMARK_SUITE,
    BenchmarkRunner,
    BenchmarkRunnerConfig,
    BenchmarkTask,
    list_benchmarks,
    load_bbh_mini,
    load_benchmark,
    load_gsm8k,
    load_humaneval,
    load_synthetic_routing,
    parse_gsm8k_answer,
    run_benchmark_sync,
)
from flybrain.benchmarks.base import KNOWN_TASK_TYPES
from flybrain.llm.mock_client import MockLLMClient, MockRule
from flybrain.runtime import MAS, Agent, AgentSpec, MASConfig

AGENT_NAMES = [a.name for a in MINIMAL_15]


# -- registry ------------------------------------------------------------------


def test_registry_lists_four_benchmarks() -> None:
    assert sorted(BENCHMARK_REGISTRY) == [
        "bbh_mini",
        "gsm8k",
        "humaneval",
        "synthetic_routing",
    ]
    assert set(DEFAULT_BENCHMARK_SUITE) == set(BENCHMARK_REGISTRY)
    assert list_benchmarks() == sorted(BENCHMARK_REGISTRY)


def test_load_benchmark_unknown_name() -> None:
    with pytest.raises(KeyError):
        load_benchmark("not_a_benchmark")


# -- per-loader determinism + shape -------------------------------------------


def test_synthetic_routing_is_deterministic_and_typed() -> None:
    a = load_synthetic_routing(num_tasks=5, seed=11)
    b = load_synthetic_routing(num_tasks=5, seed=11)
    c = load_synthetic_routing(num_tasks=5, seed=12)
    # Same seed → identical prompts.
    assert [t.prompt for t in a] == [t.prompt for t in b]
    # Different seed → different sampled prompts.
    assert [t.prompt for t in a] != [t.prompt for t in c]
    for t in a:
        assert isinstance(t, BenchmarkTask)
        assert t.task_type in KNOWN_TASK_TYPES
        assert t.benchmark == "synthetic_routing"


def test_humaneval_loader_uses_fixture_when_canonical_path_missing() -> None:
    tasks = load_humaneval(num_tasks=3, seed=0)
    assert len(tasks) == 3
    for t in tasks:
        assert t.task_type == "coding"
        assert t.benchmark == "humaneval"
        assert t.unit_tests is not None and "def check" in t.unit_tests
        assert t.metadata.get("entry_point")


def test_gsm8k_loader_extracts_final_numeric_answer() -> None:
    tasks = load_gsm8k(num_tasks=5, seed=0)
    assert len(tasks) == 5
    for t in tasks:
        assert t.task_type == "math"
        assert t.benchmark == "gsm8k"
        # Ground-truth comes from the `#### N` suffix.
        assert t.ground_truth and t.ground_truth.replace("-", "").replace(".", "").isdigit()


def test_parse_gsm8k_answer() -> None:
    assert parse_gsm8k_answer("Some reasoning.\n#### 42") == "42"
    assert parse_gsm8k_answer("step\n#### 1,234") == "1234"
    assert parse_gsm8k_answer("just one line: 7") == "just one line: 7"


def test_bbh_mini_loader_renders_options() -> None:
    tasks = load_bbh_mini(num_tasks=3, seed=0)
    assert len(tasks) == 3
    for t in tasks:
        assert t.task_type == "research"
        assert t.benchmark == "bbh_mini"
        if t.options:
            for opt in t.options:
                assert opt in t.prompt


def test_loader_raises_for_explicit_missing_path(tmp_path: Path) -> None:
    bogus = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        load_humaneval(path=bogus)
    with pytest.raises(FileNotFoundError):
        load_gsm8k(path=bogus)
    with pytest.raises(FileNotFoundError):
        load_bbh_mini(path=bogus)


def test_synthetic_routing_rejects_unknown_task_type() -> None:
    with pytest.raises(ValueError):
        load_synthetic_routing(num_tasks=2, seed=0, task_types=["bogus"])


def test_to_runtime_task_drops_benchmark_specific_fields() -> None:
    tasks = load_synthetic_routing(num_tasks=1, seed=0)
    runtime = tasks[0].to_runtime_task()
    assert runtime.task_id == tasks[0].task_id
    assert runtime.prompt == tasks[0].prompt
    assert runtime.task_type == tasks[0].task_type


# -- runner --------------------------------------------------------------------


def _mock_llm() -> MockLLMClient:
    client = MockLLMClient()
    client.rules.extend(
        [
            MockRule(pattern=r"role: planner", response="Plan: 1)"),
            MockRule(pattern=r"role: coder", response="```python\ndef f(x): return x\n```"),
            MockRule(pattern=r"role: tester", response="tests_run: 1/1 ok"),
            MockRule(pattern=r"role: math", response="final_answer: 42"),
            MockRule(
                pattern=r"role: retriever",
                response="bullets",
            ),
            MockRule(
                pattern=r"role: researcher",
                response="final_answer: ok",
            ),
            MockRule(
                pattern=r"role: tool_executor",
                response="final_answer: ok",
            ),
            MockRule(pattern=r"role: finalizer", response="final_answer: done"),
        ]
    )
    return client


def _agent_factory(spec: AgentSpec, llm) -> Agent:
    from flybrain.runtime.tools import default_tool_registry

    return Agent(spec=spec, llm=llm, tools=default_tool_registry())


def test_runner_persists_traces_and_summarises(tmp_path: Path) -> None:
    tasks = load_synthetic_routing(num_tasks=2, seed=0)
    llm = _mock_llm()
    mas = MAS.from_specs(
        list(MINIMAL_15),
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=8, trace_dir=tmp_path / "traces"),
    )
    controller = RoundRobinController()
    runner = BenchmarkRunner(
        mas,
        controller,
        initial_graph=empty_graph(AGENT_NAMES),
        config=BenchmarkRunnerConfig(parallelism=1, max_retries=1),
    )

    report = run_benchmark_sync(
        runner,
        "synthetic_routing",
        tasks,
        output_dir=tmp_path / "out",
    )
    assert report.benchmark == "synthetic_routing"
    assert report.completed == 2
    assert report.errored == 0
    assert len(report.outcomes) == 2
    for outcome in report.outcomes:
        assert outcome.trace_path is not None
        assert Path(outcome.trace_path).exists()

    summary = json.loads((tmp_path / "out" / "synthetic_routing" / "summary.json").read_text())
    assert summary["benchmark"] == "synthetic_routing"
    assert summary["completed"] == 2


def test_runner_handles_runtime_exception(tmp_path: Path) -> None:
    """When MAS.run raises, the runner records an error outcome instead
    of crashing the whole benchmark."""

    class _ExplodingMAS:
        async def run(self, *args, **kwargs):
            raise RuntimeError("boom")

    runner = BenchmarkRunner(
        _ExplodingMAS(),  # type: ignore[arg-type]
        RoundRobinController(),
        config=BenchmarkRunnerConfig(parallelism=1, max_retries=1),
    )
    tasks = load_synthetic_routing(num_tasks=1, seed=0)

    async def _go():
        return await runner.run_benchmark(
            "synthetic_routing",
            tasks,
            output_dir=tmp_path,
        )

    report = asyncio.new_event_loop().run_until_complete(_go())
    assert report.completed == 0
    assert report.errored == 1
    assert report.outcomes[0].error is not None
    assert "RuntimeError" in report.outcomes[0].error


def test_runner_parallelism_runs_async(tmp_path: Path) -> None:
    """Concurrency=2 should still produce one outcome per task."""
    tasks = load_synthetic_routing(num_tasks=4, seed=3)
    llm = _mock_llm()
    mas = MAS.from_specs(
        list(MINIMAL_15),
        llm,
        agent_factory=_agent_factory,
        config=MASConfig(max_steps=8, trace_dir=tmp_path / "traces"),
    )
    controller = RoundRobinController()
    runner = BenchmarkRunner(
        mas,
        controller,
        initial_graph=empty_graph(AGENT_NAMES),
        config=BenchmarkRunnerConfig(parallelism=2, max_retries=1),
    )
    report = asyncio.new_event_loop().run_until_complete(
        runner.run_benchmark("synthetic_routing", tasks, output_dir=tmp_path / "out")
    )
    assert report.completed == 4
    assert {o.task_id for o in report.outcomes} == {t.task_id for t in tasks}
