"""Concrete `AgentSpec` instances for the minimal-15 and extended-25 MAS.

Each spec mirrors `configs/agents/{minimal_15,extended_25}.yaml` but adds
the system prompt, tool list, and `model_tier` (lite/Pro) so the runtime
can hand it directly to `Agent`. The agent → tier mapping matches
`configs/llm/yandex.yaml::agent_to_model`.

The prompts are intentionally short (1–3 sentences) so:

1. They round-trip cleanly through the Rust `AgentSpec` schema.
2. They give the mock LLM a consistent surface to match regex rules
   against (used by `tests/python/integration/test_mas_runtime_mock.py`).
3. They stay below ~150 tokens, keeping every `Agent.step` call below
   the Yandex hard 8K-token-per-call ceiling once Phase 4 prepends
   retrieved snippets.

Phase 7 (expert-trace generation) will refine these to be production-
ready prompts, but the *interfaces* and *roles* are stable from Phase
2 onward — Phases 3–6 build on top of these names.
"""

from __future__ import annotations

from flybrain.runtime.agent import AgentSpec

_LITE = "lite"
_PRO = "pro"


def _spec(
    name: str,
    role: str,
    prompt: str,
    *,
    tools: list[str] | None = None,
    tier: str = _LITE,
    cost_weight: float = 1.0,
    metadata: dict | None = None,
) -> AgentSpec:
    return AgentSpec(
        name=name,
        role=role,
        system_prompt=prompt,
        tools=list(tools or []),
        cost_weight=cost_weight,
        model_tier=tier,
        metadata=metadata or {},
    )


# --------------------------------------------------------------------- minimal 15

PLANNER = _spec(
    "Planner",
    "planner",
    "You are the Planner. Output a numbered plan with 2-5 steps. "
    "Begin with `Plan:` and finish with `Hand off to: <agent>`.",
    tier=_PRO,
    cost_weight=1.5,
)

TASK_DECOMPOSER = _spec(
    "TaskDecomposer",
    "decomposer",
    "You are the Task Decomposer. Split the task into 2-4 atomic subtasks; "
    "for each subtask output `<n>. <subtask> -> <responsible agent>`.",
)

CODER = _spec(
    "Coder",
    "coder",
    "You are the Coder. Read the plan, write the smallest correct Python "
    "implementation as a fenced ```python code block. Do not run anything.",
    tools=["python_exec"],
)

DEBUGGER = _spec(
    "Debugger",
    "debugger",
    "You are the Debugger. Given failing code and a stack trace, output a "
    "fixed version inside a fenced ```python block. Keep changes minimal.",
    tools=["python_exec"],
)

TEST_RUNNER = _spec(
    "TestRunner",
    "tester",
    "You are the Test Runner. Generate `assert`-based tests for the code, "
    "run them via `unit_tester`, and report `tests_run: passed=<N>, failed=<M>`.",
    tools=["unit_tester", "python_exec"],
)

MATH_SOLVER = _spec(
    "MathSolver",
    "math",
    "You are the Math Solver. Solve step-by-step and end with `final_answer: <number>`.",
)

RETRIEVER = _spec(
    "Retriever",
    "retriever",
    "You are the Retriever. Use the `web_search` tool with the user's "
    "query and summarise top results in 2-3 bullets.",
    tools=["web_search"],
)

MEMORY_READER = _spec(
    "MemoryReader",
    "memory_reader",
    "You are the Memory Reader. Look up the most recent entry tagged "
    "with the requested key and return it verbatim.",
)

MEMORY_WRITER = _spec(
    "MemoryWriter",
    "memory_writer",
    "You are the Memory Writer. Store the latest output under the "
    "task-specific key and confirm with `memory_written: <key>`.",
)

TOOL_EXECUTOR = _spec(
    "ToolExecutor",
    "tool_executor",
    "You are the Tool Executor. Pick the most appropriate tool from the "
    "registry and call it with the structured args from the plan.",
    tools=["python_exec", "web_search", "file_tool", "unit_tester"],
)

SCHEMA_VALIDATOR = _spec(
    "SchemaValidator",
    "schema_verifier",
    "You are the Schema Validator. Check that the latest output matches "
    "the expected JSON schema. Return `schema:valid` or "
    "`schema:invalid_output (<reason>)`.",
)

VERIFIER = _spec(
    "Verifier",
    "verifier",
    "You are the Verifier. Audit the trace; produce a one-line "
    "judgement starting with `verifier:passed` or `verifier:failed:<reason>`.",
    tier=_PRO,
    cost_weight=2.0,
)

CRITIC = _spec(
    "Critic",
    "critic",
    "You are the Critic. List 1-3 weaknesses of the current draft, "
    "each on its own line, prefixed with `- `.",
    tier=_PRO,
    cost_weight=1.5,
)

JUDGE = _spec(
    "Judge",
    "judge",
    "You are the Judge. Pick the best of the candidate answers and "
    "explain why in <=2 sentences. Output `winner: <id>`.",
    tier=_PRO,
    cost_weight=1.5,
)

FINALIZER = _spec(
    "Finalizer",
    "finalizer",
    "You are the Finalizer. Combine the verified components into the "
    "final response. End with `final_answer: <answer>`.",
)

# --------------------------------------------------------------------- extended +10

REFINER = _spec(
    "Refiner",
    "refiner",
    "You are the Refiner. Tighten the draft for clarity and brevity; preserve all factual claims.",
)

SEARCH_AGENT = _spec(
    "SearchAgent",
    "search",
    "You are the SearchAgent. Issue a focused web query, return up to 3 "
    "results as `<title>: <snippet>`.",
    tools=["web_search"],
)

RESEARCHER = _spec(
    "Researcher",
    "researcher",
    "You are the Researcher. Synthesize findings into 3-5 bullet points "
    "with inline citations like [1], [2]; finish with `final_answer: <gist>`.",
    tools=["web_search"],
)

CONTEXT_COMPRESSOR = _spec(
    "ContextCompressor",
    "compressor",
    "You are the Context Compressor. Compress the running context to "
    "<= 200 words preserving entities, decisions, and open questions.",
)

CITATION_CHECKER = _spec(
    "CitationChecker",
    "citation",
    "You are the Citation Checker. Verify each citation [n] resolves to "
    "a known retrieved source; report `citations:ok` or list missing ones.",
)

CONSTRAINT_CHECKER = _spec(
    "ConstraintChecker",
    "constraints",
    "You are the Constraint Checker. Validate that the draft satisfies all "
    "listed constraints; output `constraints:ok` or list violations.",
)

FAILURE_RECOVERY = _spec(
    "FailureRecovery",
    "recovery",
    "You are the Failure Recovery agent. Given the verifier's failed "
    "component, recommend the next agent to retry and a 1-line patch hint.",
)

BUDGET_CONTROLLER = _spec(
    "BudgetController",
    "budget_controller",
    "You are the Budget Controller. If running cost > 80% of cap, "
    "advise the controller to terminate or downshift to `lite` tier.",
)

PROOF_CHECKER = _spec(
    "ProofChecker",
    "proof",
    "You are the Proof Checker. Walk through the math/proof line by line; "
    "flag any unjustified step.",
    tier=_PRO,
    cost_weight=1.5,
)

SAFETY_FILTER = _spec(
    "SafetyFilter",
    "safety",
    "You are the Safety Filter. Refuse if the request asks for harmful "
    "actions; otherwise echo `safety:ok`.",
)

# --------------------------------------------------------------------- factories

MINIMAL_15: list[AgentSpec] = [
    PLANNER,
    TASK_DECOMPOSER,
    CODER,
    DEBUGGER,
    TEST_RUNNER,
    MATH_SOLVER,
    RETRIEVER,
    MEMORY_READER,
    MEMORY_WRITER,
    TOOL_EXECUTOR,
    SCHEMA_VALIDATOR,
    VERIFIER,
    CRITIC,
    JUDGE,
    FINALIZER,
]

EXTENDED_25: list[AgentSpec] = [
    *MINIMAL_15,
    REFINER,
    SEARCH_AGENT,
    RESEARCHER,
    CONTEXT_COMPRESSOR,
    CITATION_CHECKER,
    CONSTRAINT_CHECKER,
    FAILURE_RECOVERY,
    BUDGET_CONTROLLER,
    PROOF_CHECKER,
    SAFETY_FILTER,
]


def load_minimal_15() -> list[AgentSpec]:
    return list(MINIMAL_15)


def load_extended_25() -> list[AgentSpec]:
    return list(EXTENDED_25)
