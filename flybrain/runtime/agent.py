"""Agent layer: wraps an `AgentSpec` + an `LLMClient` + a `ToolRegistry`.

`Agent.step(state, inbox)` is what the runner calls when the scheduler
decides this agent should run on this tick. The agent:

1. Builds a chat prompt from `spec.system_prompt` + the most recent
   `inbox` message + a short summary of `state`.
2. Calls `llm.complete(...)` honouring the spec's `model_tier`.
3. Optionally calls a tool (currently chosen by the agent's
   `default_tool` field; full free-form tool selection lands in a
   later phase together with structured-output prompts).
4. Returns an `AgentStepResult` with the produced output, the tool
   calls it made, and a "produced component" tag the runner will
   advertise to the controller via `RuntimeState.produced_components`.

For Phase 2 we keep the chat assembly intentionally simple — a richer
prompt scaffold (with retrieved snippets / memory excerpts / verifier
hints) ships with Phase 4 once embeddings and retriever are in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flybrain.llm.base import LLMClient, LLMResponse, Message, ModelTier
from flybrain.runtime.state import RuntimeState
from flybrain.runtime.tools import ToolRegistry, ToolResult


@dataclass(slots=True)
class AgentSpec:
    """Python mirror of `flybrain_core::AgentSpec`.

    Round-trips through `flybrain_native.agent_spec_round_trip` for
    schema validation. This is *not* the same class as the Rust struct;
    it is a Python dataclass keyed on the same field names so we don't
    have to hand-marshal in every test."""

    name: str
    role: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    cost_weight: float = 1.0
    model_tier: str = "lite"
    metadata: dict[str, Any] = field(default_factory=dict)

    def tier(self) -> ModelTier:
        return ModelTier.PRO if self.model_tier == "pro" else ModelTier.LITE

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "system_prompt": self.system_prompt,
            "tools": list(self.tools),
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "cost_weight": self.cost_weight,
            "model_tier": self.model_tier,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentSpec:
        return cls(
            name=d["name"],
            role=d.get("role", ""),
            system_prompt=d.get("system_prompt", ""),
            tools=list(d.get("tools", [])),
            input_schema=dict(d.get("input_schema") or {}),
            output_schema=dict(d.get("output_schema") or {}),
            cost_weight=float(d.get("cost_weight", 1.0)),
            model_tier=d.get("model_tier", "lite"),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass(slots=True)
class AgentStepResult:
    """What `Agent.step` returns to the runner."""

    agent_name: str
    output_summary: str
    output_text: str
    produced_components: list[str] = field(default_factory=list)
    tool_calls: list[ToolResult] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    cost_rub: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Agent:
    spec: AgentSpec
    llm: LLMClient
    tools: ToolRegistry
    """Spec-level tool registry; agents only ever see tools listed in `spec.tools`."""

    default_tool: str | None = None
    """If set, this tool runs once per `step` with `args = {"query": <prompt>}` for
    research-style agents. Phase 2 keeps tool selection deterministic; Phase 7
    moves it inside the prompt."""

    async def step(
        self,
        state: RuntimeState,
        inbox: list[dict[str, Any]],
    ) -> AgentStepResult:
        """Run one turn of this agent."""
        # 1. Tool call (if any). Done before the LLM call so the LLM can see results.
        tool_calls: list[ToolResult] = []
        tool_summary = ""
        if self.default_tool and self.default_tool in self.spec.tools:
            tool_args = self._tool_args_for(state, inbox)
            tr = self.tools.call(self.default_tool, tool_args)
            tool_calls.append(tr)
            tool_summary = self._summarise_tool_result(tr)

        # 2. LLM call.
        last_inbox = inbox[-1]["content"] if inbox else None
        user_text = self._render_user_message(state, last_inbox, tool_summary)
        messages = [
            Message(role="system", content=self.spec.system_prompt),
            Message(role="user", content=user_text),
        ]
        resp: LLMResponse = await self.llm.complete(messages, tier=self.spec.tier())

        produced = self._infer_produced_components(resp.content)
        return AgentStepResult(
            agent_name=self.spec.name,
            output_summary=self._summarise(resp.content),
            output_text=resp.content,
            produced_components=produced,
            tool_calls=tool_calls,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
            cost_rub=resp.cost_rub,
            errors=[],
        )

    # ------------------------------------------------------------------ helpers

    def _render_user_message(
        self,
        state: RuntimeState,
        last_inbox: Any,
        tool_summary: str,
    ) -> str:
        parts = [
            f"Task ({state.task_type}): {state.prompt}",
        ]
        if last_inbox is not None:
            parts.append(f"Previous message: {last_inbox}")
        if tool_summary:
            parts.append(f"Tool output: {tool_summary}")
        if state.last_verifier_failed_component:
            parts.append(
                f"Verifier failed on: {state.last_verifier_failed_component}"
                f" (errors: {state.last_errors[:3]})"
            )
        produced = sorted(state.produced_components)
        if produced:
            parts.append("Produced so far: " + ", ".join(produced))
        parts.append(f"Your role: {self.spec.role}.")
        return "\n".join(parts)

    def _summarise(self, text: str, n: int = 120) -> str:
        text = (text or "").strip().replace("\n", " ")
        return text if len(text) <= n else text[: n - 1] + "…"

    def _summarise_tool_result(self, tr: ToolResult) -> str:
        if not tr.ok:
            return f"[{tr.name} failed: {tr.error}]"
        # Trim long outputs — agents only need a hint, not the full payload.
        out = tr.output
        if isinstance(out, dict):
            keys = ", ".join(sorted(out.keys()))
            return f"[{tr.name} ok: keys=({keys})]"
        return f"[{tr.name} ok]"

    def _infer_produced_components(self, output: str) -> list[str]:
        """Fast, prompt-style tagging.

        Looks for lightweight markers in the agent's output (e.g.
        `final_answer:`, ```` ```python ````) and the agent's `role`
        to declare what kind of component this turn produced. The
        controller / verifier pipeline reads those tags to decide
        whether prerequisites are satisfied without parsing the full
        output."""

        out = (output or "").lower()
        tags: list[str] = [self.spec.role]
        if "final_answer" in out or "final answer" in out:
            tags.append("final_answer")
        if "```python" in out or "```py" in out:
            tags.append("code")
        if "tests passed" in out or "all tests passed" in out or "tests_run" in out:
            tags.append("tests_run")
        if "plan:" in out or "step 1" in out or "decompose" in out:
            tags.append("plan")
        return list(dict.fromkeys(tags))  # de-dup, preserve order

    def _tool_args_for(
        self,
        state: RuntimeState,
        inbox: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.default_tool == "web_search":
            return {"query": state.prompt}
        if self.default_tool == "python_exec":
            # Default code is the inbox content if it parses as code; else a no-op.
            code = ""
            if inbox:
                last = inbox[-1].get("content")
                if isinstance(last, dict) and isinstance(last.get("code"), str):
                    code = last["code"]
            return {"code": code or "print('noop')"}
        return {}
