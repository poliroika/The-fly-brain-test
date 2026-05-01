"""Tests for `flybrain.verification.pipeline`.

We keep the LLM judges off in most tests (they require an `LLMClient`)
and exercise the rule-based / Rust verifiers. A separate test wires in
a `MockLLMClient` to confirm the LLM-judge plumbing works end-to-end.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from flybrain.llm.mock_client import MockLLMClient, MockRule
from flybrain.verification import (
    VerificationConfig,
    VerificationContext,
    VerificationPipeline,
    aggregate,
)
from flybrain.verification.llm import FactualJudge, ReasoningJudge
from flybrain.verification.result import VerificationResult, fail, passing


def _ctx(**over) -> VerificationContext:
    base = dict(
        task_id="t1",
        task_type="coding",
        prompt="solve the task",
        candidate_answer="final_answer: 42",
        reference="42",
        tool_calls=[],
        unit_test_payload=None,
        final_envelope=None,
    )
    base.update(over)
    return VerificationContext(**base)


def test_aggregate_no_results_passes() -> None:
    r = aggregate([])
    assert r.passed
    assert r.score == 1.0


def test_aggregate_combines_pass_and_fail() -> None:
    items: list[tuple[str, VerificationResult]] = [
        ("schema", passing(1.0)),
        ("tool_use", fail("oops", component="tool_use")),
    ]
    r = aggregate(items)
    assert not r.passed
    assert r.failed_component == "tool_use"
    assert r.score <= 0.5
    assert any("oops" in e for e in r.errors)


@pytest.mark.asyncio
async def test_pipeline_passes_with_default_config() -> None:
    pipeline = VerificationPipeline()
    r = await pipeline.run_async(_ctx())
    assert r.passed, r.errors


@pytest.mark.asyncio
async def test_pipeline_fails_when_schema_missing_field() -> None:
    pipeline = VerificationPipeline()
    r = await pipeline.run_async(_ctx(candidate_answer=""))
    assert not r.passed
    assert r.failed_component == "schema"


@pytest.mark.asyncio
async def test_pipeline_runs_unit_test_check() -> None:
    pipeline = VerificationPipeline()
    bad_unit = {"passed": 1, "failed": 2, "all_passed": False}
    r = await pipeline.run_async(_ctx(unit_test_payload=bad_unit))
    assert not r.passed
    assert "unit_test" in (r.failed_component or "")


@pytest.mark.asyncio
async def test_pipeline_tool_use_allow_list() -> None:
    cfg = VerificationConfig()
    cfg.allowed_tools = ["python_exec"]
    pipeline = VerificationPipeline(config=cfg)
    r = await pipeline.run_async(_ctx(tool_calls=[{"name": "rm_rf", "args": {}}]))
    assert not r.passed


def test_for_task_type_math_enables_reasoning_disables_tool_use() -> None:
    cfg = VerificationConfig.for_task_type("math")
    assert cfg.use_reasoning_llm is True
    assert cfg.use_tool_use is False
    assert cfg.use_unit_test is False


def test_for_task_type_research_enables_factual() -> None:
    cfg = VerificationConfig.for_task_type("research")
    assert cfg.use_factual_llm is True
    assert cfg.use_unit_test is False


@pytest.mark.asyncio
async def test_pipeline_runs_factual_judge_with_mock_llm() -> None:
    cfg = VerificationConfig.for_task_type("research")
    cfg.use_schema = False  # focus on the LLM path
    cfg.use_tool_use = False

    llm = MockLLMClient(
        rules=[MockRule(pattern=".", response='{"passed": true, "score": 0.9, "errors": []}')]
    )
    pipeline = VerificationPipeline(config=cfg, factual=FactualJudge(llm))
    r = await pipeline.run_async(_ctx(task_type="research"))
    assert r.passed
    # The mock judge said score=0.9; aggregate over a single result is 0.9.
    assert 0.85 <= r.score <= 1.0


@pytest.mark.asyncio
async def test_pipeline_propagates_judge_failure() -> None:
    cfg = VerificationConfig.for_task_type("math")
    cfg.use_schema = False

    llm = MockLLMClient(
        rules=[
            MockRule(pattern=".", response='{"passed": false, "score": 0.1, "errors": ["wrong"]}')
        ]
    )
    pipeline = VerificationPipeline(config=cfg, reasoning=ReasoningJudge(llm))
    r = await pipeline.run_async(_ctx(task_type="math"))
    assert not r.passed
    assert any("wrong" in e for e in r.errors)


@pytest.mark.asyncio
async def test_pipeline_skips_judge_when_not_configured() -> None:
    cfg = VerificationConfig.for_task_type("math")  # asks for reasoning_llm
    cfg.use_schema = False
    pipeline = VerificationPipeline(config=cfg)  # no `reasoning=` provided
    r = await pipeline.run_async(_ctx(task_type="math"))
    # No judge ran; default tool_use+schema are off; aggregate over empty
    # result is a pass with score 1.0.
    assert r.passed
    assert r.score == 1.0


def test_run_sync_works_outside_event_loop() -> None:
    pipeline = VerificationPipeline()
    r = pipeline.run_sync(_ctx())
    assert r.passed


def _all_components(results: Iterable[tuple[str, VerificationResult]]) -> list[str]:
    return [name for name, _ in results]
