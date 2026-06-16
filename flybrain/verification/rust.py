"""Thin wrappers around the Rust verifiers exposed via `flybrain_native`.

Each helper accepts native Python data, calls into the Rust verifier,
and returns a `VerificationResult` dataclass.
"""

from __future__ import annotations

from typing import Any

from flybrain.flybrain_native import (
    budget_check as _native_budget_check,
)
from flybrain.flybrain_native import (
    schema_check as _native_schema_check,
)
from flybrain.flybrain_native import (
    tool_use_check as _native_tool_use_check,
)
from flybrain.flybrain_native import (
    trace_check as _native_trace_check,
)
from flybrain.flybrain_native import (
    unit_test_check as _native_unit_test_check,
)
from flybrain.verification.result import VerificationResult


def schema_check(payload: Any, schema: dict[str, Any]) -> VerificationResult:
    """Validate `payload` against a JSON-schema-like `schema` dict."""
    return VerificationResult.from_dict(_native_schema_check(payload, schema))


def tool_use_check(
    calls: list[dict[str, Any]],
    *,
    allowed: list[str] | None = None,
    requirements: dict[str, list[str]] | None = None,
) -> VerificationResult:
    """Verify a list of `ToolCall` dicts against an optional allow-list and
    per-tool required-arg map."""
    return VerificationResult.from_dict(
        _native_tool_use_check(calls, allowed=allowed, requirements=requirements)
    )


def trace_check(trace: dict[str, Any]) -> VerificationResult:
    """Verify the structural invariants of a `Trace` dict."""
    return VerificationResult.from_dict(_native_trace_check(trace))


def unit_test_check(payload: dict[str, Any]) -> VerificationResult:
    """Verify a `unit_tester`-style result dict (`passed`, `failed`, …)."""
    return VerificationResult.from_dict(_native_unit_test_check(payload))


def budget_check(
    *,
    hard_cap_rub: float,
    cost_rub: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    llm_calls: int = 0,
) -> VerificationResult:
    """Run the budget verifier."""
    return VerificationResult.from_dict(
        _native_budget_check(
            float(hard_cap_rub),
            float(cost_rub),
            int(tokens_in),
            int(tokens_out),
            int(llm_calls),
        )
    )
