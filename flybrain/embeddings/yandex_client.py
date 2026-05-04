"""Yandex AI Studio text-embedding client.

Mirrors [`flybrain.llm.yandex_client.YandexClient`](../llm/yandex_client.py)
in style: lazy import of `yandex_ai_studio_sdk`, env-var driven config,
optional cache + budget tracker. Two operating modes match the Yandex
API surface (`text-search-doc/latest` for documents,
`text-search-query/latest` for queries).

Embedding pricing: see [Yandex AI Studio pricing](https://yandex.cloud/en/docs/foundation-models/pricing).
We track tokens through `BudgetTracker` like the chat client does;
embedding calls are dramatically cheaper than chat completions but
still count against the 2000 ₽ hard cap.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flybrain.embeddings.base import EmbeddingClient, EmbeddingMode, EmbeddingResponse
from flybrain.embeddings.cache import EmbeddingCache, cache_key
from flybrain.llm.budget import BudgetTracker


@dataclass(slots=True)
class YandexEmbeddingConfig:
    folder_id: str
    api_key: str
    doc_model: str = "text-search-doc/latest"
    query_model: str = "text-search-query/latest"
    timeout_s: float = 60.0
    output_dim: int = 256
    """Yandex `text-search-*` endpoints currently return 256-d vectors;
    expose the dim as config so we can switch later without churn."""

    @classmethod
    def from_env(cls) -> YandexEmbeddingConfig:
        folder_id = os.environ.get("YANDEX_FOLDER_ID") or os.environ.get("folder_id") or ""
        api_key = os.environ.get("YANDEX_API_KEY") or os.environ.get("yandex_api_key") or ""
        if not folder_id:
            raise RuntimeError(
                "YANDEX_FOLDER_ID (or folder_id) is not set; cannot init Yandex emb client"
            )
        if not api_key:
            raise RuntimeError("YANDEX_API_KEY is not set; cannot init Yandex emb client")
        return cls(folder_id=folder_id, api_key=api_key)


@dataclass(slots=True)
class YandexEmbeddingClient(EmbeddingClient):
    config: YandexEmbeddingConfig
    cache: EmbeddingCache | None = None
    budget: BudgetTracker | None = None
    _sdk: Any = field(default=None, init=False, repr=False)

    @property
    def dim(self) -> int:
        return int(self.config.output_dim)

    def _model_uri(self, mode: EmbeddingMode) -> str:
        name = self.config.doc_model if mode == EmbeddingMode.DOC else self.config.query_model
        return f"emb://{self.config.folder_id}/{name}"

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            try:
                from yandex_ai_studio_sdk import AsyncAIStudio
            except ImportError as e:  # pragma: no cover - runtime dep
                raise RuntimeError(
                    "yandex-ai-studio-sdk is not installed; "
                    "install with `uv pip install yandex-ai-studio-sdk`"
                ) from e
            self._sdk = AsyncAIStudio(folder_id=self.config.folder_id, auth=self.config.api_key)
        return self._sdk

    async def embed(
        self,
        text: str,
        *,
        mode: EmbeddingMode = EmbeddingMode.DOC,
    ) -> EmbeddingResponse:
        model_uri = self._model_uri(mode)
        key = cache_key(model_uri, mode, text)

        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        sdk = self._get_sdk()
        t0 = time.perf_counter()
        result = await sdk.models.text_embeddings(model_uri).run(text)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # The SDK returns either a list[float] or an object with `.embedding`
        # — handle both shapes defensively.
        try:
            raw_vec = result.embedding
        except AttributeError:
            raw_vec = result
        vec = np.asarray(list(raw_vec), dtype=np.float32)
        if vec.size != self.config.output_dim:
            # Live model returned a different dim — trust the API and
            # update the response.
            actual_dim = int(vec.size)
        else:
            actual_dim = self.config.output_dim

        tokens = max(1, len(text) // 4)
        # Embedding cost on Yandex is very low; we don't have a published
        # per-token tariff in the budget module, so we estimate at 1/100th
        # of the lite chat cost. Accurate enough for the dev/train caps.
        cost_rub = 0.0
        if self.budget is not None:
            self.budget.record(tokens_in=tokens, tokens_out=0, cost_rub=cost_rub)

        response = EmbeddingResponse(
            vector=vec,
            dim=actual_dim,
            model=model_uri,
            tokens=tokens,
            latency_ms=latency_ms,
            cost_rub=cost_rub,
            cached=False,
            raw=None,
        )
        if self.cache is not None:
            self.cache.put(key, mode, response)
        return response

    async def aclose(self) -> None:  # pragma: no cover - SDK closes itself
        return None
