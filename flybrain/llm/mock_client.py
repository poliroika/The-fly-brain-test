"""Deterministic mock LLM client used in CI and local smoke tests.

Looks up responses by the *last user message*. If no rule matches, it
echoes the last user content verbatim with a small prefix so tests don't
need to anticipate every prompt.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from flybrain.llm.base import LLMClient, LLMResponse, Message, ModelTier
from flybrain.llm.pricing import estimate_cost_rub


@dataclass(slots=True)
class MockRule:
    """A single regex → response rule. First match wins."""

    pattern: str
    response: str
    tokens_out: int | None = None


@dataclass(slots=True)
class MockLLMClient(LLMClient):
    rules: list[MockRule] = field(default_factory=list)
    default_response: str = "[mock] no rule matched"
    fixed_latency_ms: int = 1
    deterministic_tokens: bool = True

    def add_rule(self, pattern: str, response: str, tokens_out: int | None = None) -> None:
        self.rules.append(MockRule(pattern=pattern, response=response, tokens_out=tokens_out))

    async def complete(
        self,
        messages: list[Message],
        *,
        tier: ModelTier = ModelTier.LITE,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        content = self.default_response + ": " + last_user[:200]
        tokens_out_override: int | None = None
        for rule in self.rules:
            if re.search(rule.pattern, last_user, flags=re.IGNORECASE | re.DOTALL):
                content = rule.response
                tokens_out_override = rule.tokens_out
                break

        tokens_in = (
            sum(max(1, len(m.content) // 4) for m in messages) if self.deterministic_tokens else 0
        )
        tokens_out = (
            tokens_out_override if tokens_out_override is not None else max(1, len(content) // 4)
        )
        cost = estimate_cost_rub(tier, tokens_in, tokens_out)
        time.sleep(self.fixed_latency_ms / 1000.0)

        return LLMResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=self.fixed_latency_ms,
            cost_rub=cost,
            model=f"mock/{tier.value}",
            cached=False,
            raw=None,
        )

    async def aclose(self) -> None:  # pragma: no cover - nothing to release
        return None
