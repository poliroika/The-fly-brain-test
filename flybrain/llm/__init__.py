"""LLM clients used by every agent.

Phase 0 ships:

* `LLMClient` ABC and `LLMResponse` / `Message` value types.
* `MockLLMClient` for deterministic CI runs.
* `YandexClient` thin wrapper around `yandex-ai-studio-sdk`.
* `SQLiteCache` keyed on `hash(messages, model, temperature)`.
* `BudgetTracker` keeping running cost / token / call counters.
"""

from __future__ import annotations

from flybrain.llm.base import LLMClient, LLMResponse, Message, ModelTier
from flybrain.llm.budget import BudgetExceededError, BudgetTracker
from flybrain.llm.cache import SQLiteCache
from flybrain.llm.mock_client import MockLLMClient
from flybrain.llm.pricing import RATE_LITE_RUB_PER_1K, RATE_PRO_RUB_PER_1K, estimate_cost_rub
from flybrain.llm.yandex_client import YandexClient

__all__ = [
    "RATE_LITE_RUB_PER_1K",
    "RATE_PRO_RUB_PER_1K",
    "BudgetExceededError",
    "BudgetTracker",
    "LLMClient",
    "LLMResponse",
    "Message",
    "MockLLMClient",
    "ModelTier",
    "SQLiteCache",
    "YandexClient",
    "estimate_cost_rub",
]
