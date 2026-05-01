"""Tests for the deterministic mock LLM client."""

from __future__ import annotations

import pytest

from flybrain.llm import LLMResponse, Message, MockLLMClient, ModelTier


@pytest.mark.asyncio
async def test_default_response_echoes_user_content() -> None:
    client = MockLLMClient()
    msg = Message(role="user", content="hello world")
    out = await client.complete([msg])
    assert isinstance(out, LLMResponse)
    assert "hello world" in out.content


@pytest.mark.asyncio
async def test_rule_match_overrides_default() -> None:
    client = MockLLMClient()
    client.add_rule(r"weather", "it is sunny")
    out = await client.complete([Message(role="user", content="how is the weather?")])
    assert out.content == "it is sunny"


@pytest.mark.asyncio
async def test_lite_costs_less_than_pro_for_same_tokens() -> None:
    client = MockLLMClient()
    msg = [Message(role="user", content="x" * 500)]
    lite = await client.complete(msg, tier=ModelTier.LITE)
    pro = await client.complete(msg, tier=ModelTier.PRO)
    assert pro.cost_rub > lite.cost_rub


@pytest.mark.asyncio
async def test_response_includes_token_estimates() -> None:
    client = MockLLMClient()
    out = await client.complete([Message(role="user", content="hi")])
    assert out.tokens_in >= 1
    assert out.tokens_out >= 1
    assert out.cost_rub > 0
