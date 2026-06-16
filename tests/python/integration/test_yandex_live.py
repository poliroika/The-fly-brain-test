"""Live Yandex AI Studio smoke test, opt-in only.

Runs a single 1-token completion against the real API. Skipped unless the
caller explicitly sets `FLYBRAIN_RUN_LIVE_LLM=1`. Cost is on the order of
0.5 ₽ per run.
"""

from __future__ import annotations

import os

import pytest

from flybrain.llm import (
    BudgetTracker,
    Message,
    ModelTier,
    SQLiteCache,
    YandexClient,
)
from flybrain.llm.yandex_client import YandexConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("FLYBRAIN_RUN_LIVE_LLM") != "1",
    reason="opt-in live LLM test (set FLYBRAIN_RUN_LIVE_LLM=1 to run)",
)


@pytest.mark.asyncio
async def test_one_token_smoke(tmp_path) -> None:
    config = YandexConfig.from_env()
    cache = SQLiteCache(tmp_path / "live_cache.sqlite")
    budget = BudgetTracker(hard_cap_rub=5.0)
    client = YandexClient(config=config, cache=cache, budget=budget)

    response = await client.complete(
        [Message(role="user", content="reply with the single character: ok")],
        tier=ModelTier.LITE,
        temperature=0.0,
        max_tokens=8,
    )
    assert response.content
    assert response.tokens_in > 0
    assert response.tokens_out > 0
    assert response.cost_rub > 0
    assert budget.llm_calls == 1
