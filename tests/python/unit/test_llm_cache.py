"""Tests for the SQLite LLM cache."""

from __future__ import annotations

from pathlib import Path

from flybrain.llm import LLMResponse, Message
from flybrain.llm.cache import SQLiteCache, cache_key


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "cache.sqlite")
    msgs = [Message(role="user", content="hi")]
    key = cache_key("yandexgpt-lite/latest", 0.3, msgs)

    assert cache.get(key) is None

    response = LLMResponse(
        content="ok",
        tokens_in=10,
        tokens_out=5,
        latency_ms=42,
        cost_rub=0.06,
        model="yandexgpt-lite/latest",
        cached=False,
    )
    cache.put(key, response)
    cached = cache.get(key)
    assert cached is not None
    assert cached.content == "ok"
    assert cached.tokens_in == 10
    assert cached.tokens_out == 5
    assert cached.cached is True


def test_cache_key_is_stable_across_calls() -> None:
    msgs_a = [Message(role="user", content="hello")]
    msgs_b = [Message(role="user", content="hello")]
    assert cache_key("m", 0.3, msgs_a) == cache_key("m", 0.3, msgs_b)


def test_cache_key_changes_when_temperature_changes() -> None:
    msgs = [Message(role="user", content="hello")]
    assert cache_key("m", 0.3, msgs) != cache_key("m", 0.7, msgs)


def test_cache_size(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "cache.sqlite")
    response = LLMResponse(content="ok", tokens_in=1, tokens_out=1, model="m")
    cache.put("k1", response)
    cache.put("k2", response)
    assert cache.size() == 2
