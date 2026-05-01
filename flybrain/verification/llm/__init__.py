"""LLM-backed verifiers (factual / reasoning judges).

Both judges take an `LLMClient` (Yandex or `MockLLMClient`) and a
free-form text payload, send a structured judge prompt, and parse the
response into a `VerificationResult`.

Phase 3 ships only the prompt scaffolding and the deterministic
mock-driven path. Phase 6/9 wire in real Yandex LLM calls in the
trainer / evaluator.
"""

from __future__ import annotations

from flybrain.verification.llm.factual import FactualJudge
from flybrain.verification.llm.reasoning import ReasoningJudge

__all__ = ["FactualJudge", "ReasoningJudge"]
