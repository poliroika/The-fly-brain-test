"""`VerificationPipeline` — orchestrates the Rust + Python verifiers and
projects them onto a single `VerificationResult`.

The pipeline is parameterised by a `VerificationConfig` describing
which verifiers run for which `task_type`. The Phase-3 default config
matches the table in `docs/verification.md`:

    coding   → schema + tool_use + unit_test
    math     → schema + reasoning (LLM)
    research → schema + factual (LLM)
    tool_use → schema + tool_use
    *        → schema only

`run_sync` is the canonical entry point used by the runtime (`MAS.run`
calls it via the `_DefaultVerifier` shim). LLM judges are async, so we
expose `run_async` for callers that have an event loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from flybrain.verification import rust as rust_verify
from flybrain.verification.llm import FactualJudge, ReasoningJudge
from flybrain.verification.result import VerificationResult, passing


@dataclass(slots=True)
class VerificationConfig:
    """Knobs the runner / trainer can flip to enable individual verifiers."""

    use_schema: bool = True
    use_tool_use: bool = True
    use_unit_test: bool = True
    use_factual_llm: bool = False
    use_reasoning_llm: bool = False

    final_answer_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "required": ["final_answer"],
            "properties": {"final_answer": {"type": "string", "minLength": 1}},
        }
    )
    """Default JSON-schema applied when a `coding` / `tool_use` / `*`
    task is being verified at the trace level. The shape mirrors what
    `MAS.run` passes through `last_output_summary`."""

    allowed_tools: list[str] = field(
        default_factory=lambda: [
            "python_exec",
            "unit_tester",
            "file_tool",
            "web_search",
        ]
    )
    tool_requirements: dict[str, list[str]] = field(
        default_factory=lambda: {
            "python_exec": ["code"],
            "unit_tester": ["code", "tests"],
            "file_tool": ["op"],
            "web_search": ["query"],
        }
    )

    @classmethod
    def for_task_type(cls, task_type: str) -> VerificationConfig:
        """Sensible defaults per task_type."""
        cfg = cls()
        if task_type == "math":
            cfg.use_unit_test = False
            cfg.use_tool_use = False
            cfg.use_reasoning_llm = True
        elif task_type == "research":
            cfg.use_unit_test = False
            cfg.use_tool_use = False
            cfg.use_factual_llm = True
        elif task_type == "tool_use":
            cfg.use_unit_test = False
        elif task_type == "synthetic_routing":
            cfg.use_unit_test = False
            cfg.use_tool_use = False
        return cfg


def aggregate(results: list[tuple[str, VerificationResult]]) -> VerificationResult:
    """Combine a list of `(component, VerificationResult)` into one.

    * `passed` = AND over inputs.
    * `score`  = mean of input scores.
    * `errors` / `warnings` are concatenated (component-prefixed for
      readability when the same string is emitted by multiple
      verifiers).
    * `failed_component` / `suggested_next_agent` are taken from the
      *first* failing input.
    * `reward_delta` is the sum.
    """
    if not results:
        return passing(1.0)

    all_pass = True
    score_sum = 0.0
    errors: list[str] = []
    warnings: list[str] = []
    failed_component: str | None = None
    suggested: str | None = None
    reward = 0.0

    for component, r in results:
        all_pass = all_pass and r.passed
        score_sum += r.score
        for e in r.errors:
            errors.append(f"[{component}] {e}")
        for w in r.warnings:
            warnings.append(f"[{component}] {w}")
        if not r.passed and failed_component is None:
            failed_component = r.failed_component or component
            suggested = r.suggested_next_agent
        reward += r.reward_delta

    score_mean = score_sum / max(1, len(results))
    return VerificationResult(
        passed=all_pass,
        score=score_mean if all_pass else min(score_mean, 0.5),
        errors=errors,
        warnings=warnings,
        failed_component=failed_component,
        suggested_next_agent=suggested,
        reward_delta=reward,
    )


@dataclass(slots=True)
class VerificationContext:
    """Subset of `RuntimeState` the pipeline needs."""

    task_id: str
    task_type: str
    prompt: str
    candidate_answer: str | None
    reference: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """All tool calls observed across the trace so far. The pipeline
    runs `tool_use_check` over them."""

    unit_test_payload: dict[str, Any] | None = None
    """Most recent `unit_tester` output dict, if any."""

    final_envelope: dict[str, Any] | None = None
    """Optional dict to feed to `schema_check`. If `None`, the pipeline
    builds `{"final_answer": candidate_answer}` so the default schema
    matches."""


@dataclass
class VerificationPipeline:
    config: VerificationConfig = field(default_factory=VerificationConfig)
    factual: FactualJudge | None = None
    reasoning: ReasoningJudge | None = None

    async def run_async(self, ctx: VerificationContext) -> VerificationResult:
        results: list[tuple[str, VerificationResult]] = []

        if self.config.use_schema:
            envelope = ctx.final_envelope
            if envelope is None:
                envelope = {"final_answer": ctx.candidate_answer or ""}
            results.append(
                ("schema", rust_verify.schema_check(envelope, self.config.final_answer_schema))
            )

        if self.config.use_tool_use:
            results.append(
                (
                    "tool_use",
                    rust_verify.tool_use_check(
                        ctx.tool_calls,
                        allowed=self.config.allowed_tools,
                        requirements=self.config.tool_requirements,
                    ),
                )
            )

        if self.config.use_unit_test and ctx.unit_test_payload is not None:
            results.append(("unit_test", rust_verify.unit_test_check(ctx.unit_test_payload)))

        if self.config.use_factual_llm and self.factual is not None:
            r = await self.factual.verify(
                candidate=ctx.candidate_answer or "",
                reference=ctx.reference or "",
            )
            results.append(("factual", r))

        if self.config.use_reasoning_llm and self.reasoning is not None:
            r = await self.reasoning.verify(
                task=ctx.prompt,
                candidate=ctx.candidate_answer or "",
            )
            results.append(("reasoning", r))

        return aggregate(results)

    def run_sync(self, ctx: VerificationContext) -> VerificationResult:
        """Synchronous wrapper. Spins up a fresh event loop only when
        no loop is running (so we play nicely with the tests' `pytest-asyncio`).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(self.run_async(ctx))

        # We're already in an event loop (e.g. inside MAS.run): schedule the
        # coroutine on a fresh loop on a thread to avoid `await` from sync
        # context. For Phase 3 the integration tests stay outside this branch.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run_async(ctx)).result()
