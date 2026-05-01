"""Reasoning judge: rates whether the chain-of-thought / plan that
produced the candidate answer is internally consistent and actually
solves the task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flybrain.llm.base import ModelTier
from flybrain.verification.llm._judge import run_judge
from flybrain.verification.result import VerificationResult

if TYPE_CHECKING:
    from flybrain.llm.base import LLMClient

SYSTEM_PROMPT = (
    "You are a reasoning auditor. Read the task and the candidate "
    "solution (which may include intermediate plan / chain-of-thought "
    "and the final answer). Decide whether the reasoning is internally "
    "consistent AND actually addresses the task. Reply with a single "
    'JSON object: {"passed": <bool>, "score": <float in [0,1]>, "errors": [...]}.'
)


class ReasoningJudge:
    component: str = "reasoning"
    suggested_next_agent: str | None = "Critic"

    def __init__(self, llm: LLMClient, *, tier: ModelTier = ModelTier.PRO) -> None:
        self.llm = llm
        self.tier = tier

    async def verify(self, *, task: str, candidate: str) -> VerificationResult:
        if not candidate.strip():
            return VerificationResult(
                passed=False,
                score=0.0,
                errors=["reasoning: empty candidate"],
                failed_component=self.component,
                suggested_next_agent=self.suggested_next_agent,
                reward_delta=-1.0,
            )
        user_prompt = (
            f"Task:\n{task.strip()}\n\n"
            f"Candidate solution:\n{candidate.strip()}\n\n"
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
