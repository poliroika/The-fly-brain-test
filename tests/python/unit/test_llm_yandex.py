"""Tests for the YandexClient wrapper.

These tests do NOT hit Yandex AI Studio — they monkeypatch the SDK to keep
CI hermetic and the 2000 ₽ budget safe. A live smoke test guarded by an
explicit env var lives in `tests/python/integration/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from flybrain.llm import (
    BudgetTracker,
    Message,
    ModelTier,
    SQLiteCache,
    YandexClient,
)
from flybrain.llm.budget import BudgetExceededError
from flybrain.llm.yandex_client import YandexConfig


class _FakeUsage:
    def __init__(self, ti: int, to: int) -> None:
        self.input_text_tokens = ti
        self.completion_tokens = to


class _FakeAlt:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResult(list):
    def __init__(self, text: str, ti: int, to: int) -> None:
        super().__init__([_FakeAlt(text)])
        self.usage = _FakeUsage(ti, to)


class _FakeCompletionsChain:
    def __init__(self, text: str, tokens_in: int, tokens_out: int) -> None:
        self._text = text
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out

    def configure(self, **_: Any) -> _FakeCompletionsChain:
        return self

    async def run(self, _messages: list[dict[str, str]]) -> _FakeResult:
        return _FakeResult(self._text, self._tokens_in, self._tokens_out)


class _FakeModels:
    def __init__(self, text: str, tokens_in: int, tokens_out: int) -> None:
        self._text = text
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out

    def completions(self, _model_uri: str) -> _FakeCompletionsChain:
        return _FakeCompletionsChain(self._text, self._tokens_in, self._tokens_out)


class _FakeSDK:
    def __init__(self, text: str = "ok", tokens_in: int = 100, tokens_out: int = 50) -> None:
        self.models = _FakeModels(text, tokens_in, tokens_out)


def _make_config() -> YandexConfig:
    return YandexConfig(folder_id="b1g_test", api_key="dummy")


@pytest.mark.asyncio
async def test_complete_uses_lite_uri_by_default() -> None:
    client = YandexClient(config=_make_config())
    client._sdk = _FakeSDK("hello", 80, 20)
    out = await client.complete([Message(role="user", content="hi")])
    assert "yandexgpt-lite/latest" in out.model
    assert out.tokens_in == 80
    assert out.tokens_out == 20
    assert out.cost_rub > 0


@pytest.mark.asyncio
async def test_complete_uses_pro_uri_when_requested() -> None:
    client = YandexClient(config=_make_config())
    client._sdk = _FakeSDK("hello", 80, 20)
    out = await client.complete(
        [Message(role="user", content="hi")],
        tier=ModelTier.PRO,
    )
    assert "yandexgpt/latest" in out.model
    assert "lite" not in out.model


@pytest.mark.asyncio
async def test_cache_hit_skips_sdk(tmp_path: Path) -> None:
    cache = SQLiteCache(tmp_path / "cache.sqlite")
    client = YandexClient(config=_make_config(), cache=cache)
    client._sdk = _FakeSDK("first", 80, 20)

    msgs = [Message(role="user", content="hi")]
    first = await client.complete(msgs)
    assert first.cached is False

    # Mutate the SDK so any second real call would return a different value.
    client._sdk = _FakeSDK("DIFFERENT", 999, 999)
    second = await client.complete(msgs)
    assert second.cached is True
    assert second.content == "first"


@pytest.mark.asyncio
async def test_budget_blocks_when_over_cap() -> None:
    budget = BudgetTracker(hard_cap_rub=0.0001)
    client = YandexClient(config=_make_config(), budget=budget)
    client._sdk = _FakeSDK("hello", 1000, 1000)
    with pytest.raises(BudgetExceededError):
        await client.complete([Message(role="user", content="x" * 10_000)])


@pytest.mark.asyncio
async def test_budget_records_cost_after_call() -> None:
    budget = BudgetTracker(hard_cap_rub=2000.0)
    client = YandexClient(config=_make_config(), budget=budget)
    client._sdk = _FakeSDK("hello", 80, 20)
    await client.complete([Message(role="user", content="hi")])
    assert budget.llm_calls == 1
    assert budget.cost_rub > 0


def test_config_from_env_requires_folder_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.delenv("folder_id", raising=False)
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    with pytest.raises(RuntimeError):
        YandexConfig.from_env()


def test_config_from_env_accepts_lowercase_folder_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    monkeypatch.setenv("folder_id", "b1g_lowercase")
    monkeypatch.setenv("YANDEX_API_KEY", "k")
    cfg = YandexConfig.from_env()
    assert cfg.folder_id == "b1g_lowercase"
    assert cfg.api_key == "k"
