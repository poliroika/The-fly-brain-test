"""Verification layer (Phase 3).

Two halves:

* `flybrain.verification.rust`  — thin Python wrappers around the
  deterministic Rust verifiers in `flybrain-verify` (schema, tool_use,
  trace, unit_test, budget).
* `flybrain.verification.llm`   — LLM-backed judges (factual,
  reasoning) that take a Yandex (or mock) `LLMClient` and return a
  `VerificationResult`.

The orchestrator `flybrain.verification.pipeline.VerificationPipeline`
combines the two and dispatches by `task_type`.
"""

from __future__ import annotations

from flybrain.verification.pipeline import (
    VerificationConfig,
    VerificationContext,
    VerificationPipeline,
    aggregate,
)
from flybrain.verification.result import VerificationResult, fail, passing

__all__ = [
    "VerificationConfig",
    "VerificationContext",
    "VerificationPipeline",
    "VerificationResult",
    "aggregate",
    "fail",
    "passing",
]
