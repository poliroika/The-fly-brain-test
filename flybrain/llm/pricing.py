"""Yandex AI Studio pricing constants and cost estimation.

Numbers are public-pricing estimates as of 2025 and can be overridden via
Hydra (`configs/llm/yandex.yaml`). They are intentionally pessimistic so the
budget tracker errs on the side of stopping too early rather than too late.
"""

from __future__ import annotations

from flybrain.llm.base import ModelTier

RATE_LITE_RUB_PER_1K: float = 0.40
RATE_PRO_RUB_PER_1K: float = 1.20
RATE_EMBED_RUB_PER_1K: float = 0.10


def estimate_cost_rub(tier: ModelTier, tokens_in: int, tokens_out: int) -> float:
    """Return an upper-bound RUB cost for a single completion.

    Yandex prices input and output tokens identically per their public API
    pricing page; we use a single per-1k token rate weighted by tier.
    """
    rate = RATE_LITE_RUB_PER_1K if tier == ModelTier.LITE else RATE_PRO_RUB_PER_1K
    total = tokens_in + tokens_out
    return rate * total / 1000.0
