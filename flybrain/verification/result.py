"""Python mirror of `flybrain_core::verify::VerificationResult`.

Strongly-typed dataclass instead of dict because the verification
pipeline is called many times per task and we want type-checked field
names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    score: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failed_component: str | None = None
    suggested_next_agent: str | None = None
    reward_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": float(self.score),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "failed_component": self.failed_component,
            "suggested_next_agent": self.suggested_next_agent,
            "reward_delta": float(self.reward_delta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VerificationResult:
        return cls(
            passed=bool(d.get("passed", False)),
            score=float(d.get("score", 0.0)),
            errors=list(d.get("errors") or []),
            warnings=list(d.get("warnings") or []),
            failed_component=d.get("failed_component"),
            suggested_next_agent=d.get("suggested_next_agent"),
            reward_delta=float(d.get("reward_delta", 0.0)),
        )


def passing(score: float = 1.0) -> VerificationResult:
    return VerificationResult(passed=True, score=float(score))


def fail(
    reason: str,
    *,
    component: str | None = None,
    suggested_next_agent: str | None = None,
    reward_delta: float = -1.0,
) -> VerificationResult:
    return VerificationResult(
        passed=False,
        score=0.0,
        errors=[reason],
        failed_component=component,
        suggested_next_agent=suggested_next_agent,
        reward_delta=reward_delta,
    )
