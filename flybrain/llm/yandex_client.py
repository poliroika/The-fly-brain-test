"""YandexGPT chat completion client built on `yandex-cloud-ml-sdk`.

Phase 0 implements the chat path with cache + budget tracking. Embedding
support, function calling and streaming arrive in subsequent phases. We
keep the dependency on `yandex_cloud_ml_sdk` lazy so the module imports
cleanly in test environments where the SDK isn't installed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from flybrain.llm.base import LLMClient, LLMResponse, Message, ModelTier
from flybrain.llm.budget import BudgetTracker
from flybrain.llm.cache import SQLiteCache, cache_key
from flybrain.llm.pricing import estimate_cost_rub


@dataclass(slots=True)
class YandexConfig:
    folder_id: str
    api_key: str
    lite_model: str = "yandexgpt-lite/latest"
    pro_model: str = "yandexgpt/latest"
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> YandexConfig:
        # We accept multiple env-var spellings since users sometimes set the
        # folder id without the YANDEX_ prefix.
        folder_id = os.environ.get("YANDEX_FOLDER_ID") or os.environ.get("folder_id") or ""
        api_key = os.environ.get("YANDEX_API_KEY") or os.environ.get("yandex_api_key") or ""
        if not folder_id:
            raise RuntimeError(
                "YANDEX_FOLDER_ID (or folder_id) is not set; cannot init Yandex client"
            )
        if not api_key:
            raise RuntimeError("YANDEX_API_KEY is not set; cannot init Yandex client")
        return cls(folder_id=folder_id, api_key=api_key)


@dataclass(slots=True)
class YandexClient(LLMClient):
    config: YandexConfig
    cache: SQLiteCache | None = None
    budget: BudgetTracker | None = None
    _sdk: Any = field(default=None, init=False, repr=False)

    def _model_uri(self, tier: ModelTier) -> str:
        name = self.config.lite_model if tier == ModelTier.LITE else self.config.pro_model
        return f"gpt://{self.config.folder_id}/{name}"

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            try:
                from yandex_cloud_ml_sdk import AsyncYCloudML
            except ImportError as e:  # pragma: no cover - runtime dep
                raise RuntimeError(
                    "yandex-cloud-ml-sdk is not installed; "
                    "install with `uv pip install yandex-cloud-ml-sdk`"
                ) from e
            self._sdk = AsyncYCloudML(folder_id=self.config.folder_id, auth=self.config.api_key)
        return self._sdk

    async def complete(
        self,
        messages: list[Message],
        *,
        tier: ModelTier = ModelTier.LITE,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        model_uri = self._model_uri(tier)
        key = cache_key(model_uri, temperature, messages)

        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        if self.budget is not None:
            # Pessimistic projection: assume the response uses the full
            # max_tokens. We refund the difference once we know the real
            # output count.
            projected_in = sum(max(1, len(m.content) // 4) for m in messages)
            projected = estimate_cost_rub(tier, projected_in, max_tokens)
            self.budget.reserve(projected)

        sdk = self._get_sdk()
        sdk_messages = [{"role": m.role, "text": m.content} for m in messages]

        t0 = time.perf_counter()
        result = await (
            sdk.models.completions(model_uri)
            .configure(temperature=temperature, max_tokens=max_tokens)
            .run(sdk_messages)
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # The SDK returns an iterable of alternatives with `.text` on each;
        # we take the first.
        try:
            content = result[0].text
        except Exception:  # pragma: no cover - SDK schema drift
            content = str(result)

        usage = getattr(result, "usage", None)
        tokens_in = int(getattr(usage, "input_text_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        if tokens_in == 0 and tokens_out == 0:
            tokens_in = sum(max(1, len(m.content) // 4) for m in messages)
            tokens_out = max(1, len(content) // 4)

        cost = estimate_cost_rub(tier, tokens_in, tokens_out)

        if self.budget is not None:
            self.budget.record(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_rub=cost,
            )

        response = LLMResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_rub=cost,
            model=model_uri,
            cached=False,
            raw=None,
        )
        if self.cache is not None:
            self.cache.put(key, response)
        return response

    async def aclose(self) -> None:  # pragma: no cover - SDK manages its loop
        return None
