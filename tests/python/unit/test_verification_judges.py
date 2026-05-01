"""Tests for the LLM-backed judges in `flybrain.verification.llm`."""

from __future__ import annotations

import pytest

from flybrain.llm.mock_client import MockLLMClient, MockRule
from flybrain.verification.llm._judge import parse_judge_response
from flybrain.verification.llm.factual import FactualJudge
from flybrain.verification.llm.reasoning import ReasoningJudge


def test_parse_judge_response_handles_pure_json() -> None:
    o = parse_judge_response('{"passed": true, "score": 0.8, "errors": []}')
    assert o.passed
    assert o.score == 0.8
    assert o.errors == []


def test_parse_judge_response_handles_prefixed_prose() -> None:
    text = (
        "Sure, here's my verdict:\n"
        '{"passed": false, "score": 0.2, "errors": ["off-topic"]}\n'
        "Hope that helps!"
    )
    o = parse_judge_response(text)
    assert not o.passed
    assert "off-topic" in o.errors


def test_parse_judge_response_falls_back_to_failure_on_garbage() -> None:
    o = parse_judge_response("not json at all")
    assert not o.passed
    assert o.errors


def test_parse_judge_response_clamps_score() -> None:
    o = parse_judge_response('{"passed": true, "score": 9.9}')
    assert o.passed
    assert 0.0 <= o.score <= 1.0


def _llm_returning(content: str) -> MockLLMClient:
    return MockLLMClient(rules=[MockRule(pattern=".", response=content)])


@pytest.mark.asyncio
async def test_factual_judge_passes_through_pass_verdict() -> None:
    judge = FactualJudge(_llm_returning('{"passed": true, "score": 0.95, "errors": []}'))
    r = await judge.verify(candidate="Paris", reference="The capital of France is Paris.")
    assert r.passed
    assert r.score == 0.95


@pytest.mark.asyncio
async def test_factual_judge_short_circuits_empty_candidate() -> None:
    judge = FactualJudge(_llm_returning('{"passed": true, "score": 1.0}'))
    r = await judge.verify(candidate="", reference="x")
    assert not r.passed
    assert r.failed_component == "factual"


@pytest.mark.asyncio
async def test_factual_judge_skips_when_reference_empty() -> None:
    judge = FactualJudge(_llm_returning('{"passed": true, "score": 1.0}'))
    r = await judge.verify(candidate="answer", reference="")
    assert r.passed
    assert r.warnings  # we recorded a "skipped" warning


@pytest.mark.asyncio
async def test_reasoning_judge_failure_carries_errors() -> None:
    judge = ReasoningJudge(
        _llm_returning('{"passed": false, "score": 0.1, "errors": ["not internally consistent"]}')
    )
    r = await judge.verify(task="prove 2+2=4", candidate="trust me")
    assert not r.passed
    assert r.failed_component == "reasoning"
    assert any("internally" in e for e in r.errors)


@pytest.mark.asyncio
async def test_reasoning_judge_handles_garbage_response() -> None:
    judge = ReasoningJudge(_llm_returning("garbage that is not JSON"))
    r = await judge.verify(task="x", candidate="y")
    assert not r.passed
