"""LLM client abstraction. All concrete clients (Yandex, mock, future OpenAI)
implement `LLMClient`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ModelTier(str, Enum):
    LITE = "lite"
    PRO = "pro"


@dataclass(slots=True)
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass(slots=True)
class LLMResponse:
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cost_rub: float = 0.0
    model: str = ""
    cached: bool = False
    raw: dict | None = field(default=None)


class LLMClient(ABC):
    """Async-capable LLM interface.

    Implementations MUST honour the `tier` argument: Lite for fast, cheap
    agents; Pro for reasoning-heavy roles. The mapping from agent → tier
    lives in `configs/llm/yandex.yaml`.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        tier: ModelTier = ModelTier.LITE,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Run a chat completion and return a structured response."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release connections / flush caches."""
