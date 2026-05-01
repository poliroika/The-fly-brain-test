"""Shared scaffolding for LLM-backed verifiers.

The judge prompt asks for a strict JSON response of the shape

    {"passed": <bool>, "score": <float in [0,1]>, "errors": [<str>, ...]}

We parse that envelope (best-effort: substring-find the first `{...}`
block, `json.loads` it) and project it onto a `VerificationResult`.

If parsing fails we treat the judge as having errored and emit a
soft-fail `VerificationResult` with `score=0.5` so a single flaky
judge response doesn't terminate the loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flybrain.verification.result import VerificationResult

if TYPE_CHECKING:
    from flybrain.llm.base import LLMClient, ModelTier

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class JudgeOutcome:
    passed: bool
    score: float
    errors: list[str]
    raw: str


def parse_judge_response(text: str) -> JudgeOutcome:
    """Extract `{passed, score, errors}` from a judge's free-form reply.

    Tolerant of leading/trailing prose: we match the first JSON object
    we find. If anything goes wrong we report a `passed=False`
    outcome so the pipeline can decide what to do.
    """
    match = _JSON_RE.search(text)
    if not match:
        return JudgeOutcome(False, 0.0, [f"judge returned non-JSON: {text[:80]}"], text)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return JudgeOutcome(False, 0.0, [f"judge JSON parse error: {exc}"], text)

    passed = bool(payload.get("passed", False))
    score_raw = payload.get("score", 1.0 if passed else 0.0)
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score = 1.0 if passed else 0.0
    score = max(0.0, min(1.0, score))
    errors_raw = payload.get("errors") or []
    errors = [str(e) for e in errors_raw if e]
    return JudgeOutcome(passed, score, errors, text)


async def run_judge(
    llm: LLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    tier: ModelTier,
    component: str,
    suggested_next_agent: str | None,
) -> VerificationResult:
    """Send `(system, user)` to `llm`, parse, project to `VerificationResult`."""
    from flybrain.llm.base import Message

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    response = await llm.complete(messages=messages, tier=tier)
    outcome = parse_judge_response(response.content)

    if outcome.passed:
        return VerificationResult(passed=True, score=outcome.score)
    return VerificationResult(
        passed=False,
        score=outcome.score,
        errors=outcome.errors or ["judge marked the answer as failing"],
        failed_component=component,
        suggested_next_agent=suggested_next_agent,
        reward_delta=-0.5,
    )
