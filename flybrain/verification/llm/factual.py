"""Factual judge: verifies a candidate answer against a ground-truth or
the source documents the agent was supposed to read.

The judge prompt asks the model to answer one yes/no question
("does this answer accurately reflect the provided ground-truth /
sources?") and produce a JSON envelope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flybrain.llm.base import ModelTier
from flybrain.verification.llm._judge import run_judge
from flybrain.verification.result import VerificationResult

if TYPE_CHECKING:
    from flybrain.llm.base import LLMClient

SYSTEM_PROMPT = (
    "You are a strict fact-checker. Compare the candidate answer against "
    "the reference (ground truth and/or source snippets). Reply with a "
    'single JSON object: {"passed": <bool>, "score": <float in [0,1]>, '
    '"errors": [<short reasons>]}. Only mark passed=true if the candidate '
    "is materially consistent with the reference."
)


class FactualJudge:
    """Wraps an `LLMClient` as a factual verifier."""

    component: str = "factual"
    suggested_next_agent: str | None = "Researcher"

    def __init__(self, llm: LLMClient, *, tier: ModelTier = ModelTier.PRO) -> None:
        self.llm = llm
        self.tier = tier

    async def verify(self, *, candidate: str, reference: str) -> VerificationResult:
        if not candidate.strip():
            return VerificationResult(
                passed=False,
                score=0.0,
                errors=["factual: empty candidate answer"],
                failed_component=self.component,
                suggested_next_agent=self.suggested_next_agent,
                reward_delta=-1.0,
            )
        if not reference.strip():
            # Nothing to check against — soft pass.
            return VerificationResult(
                passed=True, score=0.5, warnings=["factual: reference is empty, judge skipped"]
            )
        user_prompt = (
            f"Reference:\n{reference.strip()}\n\n"
            f"Candidate answer:\n{candidate.strip()}\n\n"
            'Reply with a JSON object: {"passed": ..., "score": ..., "errors": [...]}'
        )
        return await run_judge(
            self.llm,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tier=self.tier,
            component=self.component,
            suggested_next_agent=self.suggested_next_agent,
        )
