"""Running budget tracker shared between agents and the controller.

Mirrors the Rust `BudgetVerifier` so we can validate behaviour from Python
quickly without crossing the FFI boundary on every call. Phase 0 keeps it
in-memory; persistence and per-task carry-over land in Phase 2/3.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(RuntimeError):
    """Raised when an LLM call would push us past the hard cap."""


@dataclass(slots=True)
class BudgetTracker:
    hard_cap_rub: float
    soft_cap_rub: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    cost_rub: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.soft_cap_rub is None:
            self.soft_cap_rub = 0.8 * self.hard_cap_rub

    @property
    def remaining_rub(self) -> float:
        return self.hard_cap_rub - self.cost_rub

    def will_exceed(self, projected_rub: float) -> bool:
        return self.cost_rub + projected_rub > self.hard_cap_rub

    def reserve(self, projected_rub: float) -> None:
        """Raise if a planned call would exceed the hard cap."""
        if self.will_exceed(projected_rub):
            raise BudgetExceededError(
                f"projected {projected_rub:.4f} ₽ would push cost "
                f"{self.cost_rub:.4f}+{projected_rub:.4f} > {self.hard_cap_rub:.2f} ₽"
            )

    def record(self, *, tokens_in: int, tokens_out: int, cost_rub: float) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.llm_calls += 1
        self.cost_rub += cost_rub
        if self.soft_cap_rub is not None and self.cost_rub > self.soft_cap_rub:
            self.warnings.append(
                f"soft cap reached: {self.cost_rub:.2f} > {self.soft_cap_rub:.2f} ₽"
            )

    def snapshot(self) -> dict[str, float | int]:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "llm_calls": self.llm_calls,
            "cost_rub": round(self.cost_rub, 4),
            "remaining_rub": round(self.remaining_rub, 4),
        }
