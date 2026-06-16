"""`MAS.run` — the per-task event loop tying together:

* the Rust [`Scheduler`](`flybrain.flybrain_native.Scheduler`) that owns the
  dynamic `AgentGraph` + step counter,
* the Rust [`MessageBus`](`flybrain.flybrain_native.MessageBus`) that routes
  inter-agent messages,
* the Rust [`TraceWriter`](`flybrain.flybrain_native.TraceWriter`) that
  persists every step,
* the Python [`Agent`](flybrain.runtime.agent.Agent) layer that talks to
  the LLM,
* the Python [`Controller`](flybrain.controller.base.Controller)
  protocol — `ManualController` for Phase 2, GNN/RNN/learned-router for
  Phase 5.

The loop executes at most `max_steps` ticks per task and returns the
fully-realised [`Trace`](`flybrain_core::Trace`) dict on completion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flybrain.flybrain_native import MessageBus, Scheduler, TraceWriter
from flybrain.llm.base import LLMClient
from flybrain.runtime.agent import Agent, AgentSpec, AgentStepResult
from flybrain.runtime.memory import EpisodicMemory
from flybrain.runtime.retriever import BM25Retriever
from flybrain.runtime.state import RuntimeState
from flybrain.verification import (
    VerificationConfig,
    VerificationContext,
    VerificationPipeline,
)


@dataclass(slots=True)
class Task:
    """Lightweight task descriptor passed to `MAS.run`. Mirrors the Rust
    `TaskSpec` but keeps Python-native types for ground_truth / budget."""

    task_id: str
    task_type: str
    prompt: str
    ground_truth: Any = None


@dataclass(slots=True)
class _RuleVerifier:
    """Cheap component-presence checker used as the *cheap* part of the
    Phase-3 verification pipeline.

    The Rust + LLM verifiers in `flybrain.verification.pipeline` cover
    schema / tool_use / unit_test / factual / reasoning. They only fire
    when the runner has something to check (a candidate answer, tool
    calls, a unit-test payload). This rule-based check stays in the
    loop because it is what tells a Phase-2 manual controller "you've
    skipped the Coder, the run is incomplete" — i.e. it ensures the
    plan itself was followed, not just the artefacts.
    """

    def check(self, state: RuntimeState) -> dict[str, Any]:
        produced = state.produced_components
        if state.task_type == "coding":
            need = {"plan", "code", "tests_run"}
        elif state.task_type == "math":
            need = {"final_answer"}
        elif state.task_type == "research":
            need = {"plan", "final_answer"}
        elif state.task_type == "tool_use":
            need = {"final_answer"}
        else:
            need = {"final_answer"}

        missing = sorted(need - produced)
        if not missing:
            return {
                "passed": True,
                "score": 1.0,
                "errors": [],
                "warnings": [],
                "failed_component": None,
                "suggested_next_agent": None,
                "reward_delta": 0.5,
            }
        return {
            "passed": False,
            "score": max(0.0, 1.0 - 0.3 * len(missing)),
            "errors": [f"missing component: {m}" for m in missing],
            "warnings": [],
            "failed_component": missing[0],
            "suggested_next_agent": None,
            "reward_delta": -0.1 * len(missing),
        }


@dataclass(slots=True)
class MASConfig:
    max_steps: int = 32
    trace_dir: Path | None = None
    """If set, each run writes `<trace_dir>/<task_id>.steps.jsonl` and
    `<trace_dir>/<task_id>.trace.json`."""

    seed: int = 42


@dataclass(slots=True)
class MAS:
    agents: dict[str, Agent]
    """name → Agent registry."""

    llm: LLMClient
    """Used directly only for the Verifier role; everything else goes
    through the per-agent `Agent.llm`."""

    config: MASConfig = field(default_factory=MASConfig)

    verification: VerificationPipeline | None = None
    """Optional Phase-3 verification pipeline. When present, every
    `call_verifier` action runs the full Rust + LLM pipeline in
    addition to the cheap rule-based component check; when ``None``
    the runner falls back to the rule-based verdict only."""

    @classmethod
    def from_specs(
        cls,
        specs: list[AgentSpec],
        llm: LLMClient,
        agent_factory,
        config: MASConfig | None = None,
    ) -> MAS:
        """Build a `MAS` by instantiating one `Agent` per `AgentSpec` via
        `agent_factory(spec, llm) -> Agent`."""
        agents = {s.name: agent_factory(s, llm) for s in specs}
        return cls(agents=agents, llm=llm, config=config or MASConfig())

    async def run(
        self,
        task: Task,
        controller,
        *,
        initial_graph: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one task end-to-end and return the finalised `Trace` dict."""

        agent_names = sorted(self.agents.keys())
        graph = initial_graph or {"nodes": agent_names, "edges": {}}
        sched = Scheduler(graph)
        bus = MessageBus()

        sink_path: str | None = None
        if self.config.trace_dir is not None:
            self.config.trace_dir.mkdir(parents=True, exist_ok=True)
            sink_path = str(self.config.trace_dir / f"{task.task_id}.steps.jsonl")
        trace = TraceWriter(task.task_id, task.task_type, sink_path)

        memory = EpisodicMemory()
        retriever = BM25Retriever()

        state = RuntimeState(
            task_id=task.task_id,
            task_type=task.task_type,
            prompt=task.prompt,
            step_id=0,
            available_agents=agent_names,
            pending_inbox={},
            last_active_agent=None,
        )

        verifier = _RuleVerifier()
        pipeline = self.verification or VerificationPipeline(
            config=VerificationConfig.for_task_type(task.task_type)
        )
        observed_tool_calls: list[dict[str, Any]] = []
        last_unit_test_payload: dict[str, Any] | None = None
        last_step_for_trace: dict[str, Any] = {}
        last_nonempty_output: str | None = None

        for _ in range(self.config.max_steps):
            action = controller.select_action(state)
            t_start = time.perf_counter()
            try:
                status = sched.apply(action)
            except ValueError as e:
                trace.record_step(
                    self._make_step(
                        sched.step_id,
                        sched.current_graph_hash,
                        action,
                        active_agent=state.last_active_agent,
                        errors=[f"scheduler rejected action: {e}"],
                        latency_ms=int((time.perf_counter() - t_start) * 1000),
                    )
                )
                sched.advance_step()
                state = self._refresh_state(state, sched, bus, last_step_for_trace, None, [])
                break

            kind = status["kind"]
            step_extras: dict[str, Any] = {
                "tool_calls": [],
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_rub": 0.0,
                "errors": [],
                "verifier_score": None,
                "active_agent": sched.last_active_agent,
                "output_summary": None,
                "input_msg_id": None,
            }

            verifier_payload: dict[str, Any] | None = None

            if kind == "run_agent":
                agent_name = status["agent"]
                agent = self.agents.get(agent_name)
                if agent is None:
                    step_extras["errors"].append(f"unregistered agent {agent_name}")
                else:
                    inbox: list[dict[str, Any]] = []
                    while True:
                        msg = bus.pop(agent_name)
                        if msg is None:
                            break
                        inbox.append(msg)
                    if inbox:
                        step_extras["input_msg_id"] = str(inbox[-1]["id"])
                    result: AgentStepResult = await agent.step(state, inbox)
                    step_extras["tool_calls"] = [tr.as_tool_call() for tr in result.tool_calls]
                    step_extras["tokens_in"] = result.tokens_in
                    step_extras["tokens_out"] = result.tokens_out
                    step_extras["cost_rub"] = result.cost_rub
                    step_extras["errors"].extend(result.errors)
                    step_extras["output_summary"] = result.output_summary

                    state.produced_components |= set(result.produced_components)
                    memory.append(
                        tag=f"agent:{agent_name}",
                        content=result.output_text,
                        step_id=sched.step_id,
                    )
                    retriever.add(result.output_text)
                    for tr in result.tool_calls:
                        observed_tool_calls.append(tr.as_tool_call())
                        if tr.name == "unit_tester" and tr.ok:
                            last_unit_test_payload = dict(tr.output)

                    # Forward output to all neighbours of agent_name in the
                    # current AgentGraph. Empty by default (Phase 2 manual
                    # controller doesn't add edges) — Phase 5 controllers
                    # will populate the graph and rely on this routing.
                    edges = sched.agent_graph().get("edges", {}).get(agent_name, {})
                    for recipient in edges:
                        bus.send(
                            agent_name,
                            recipient,
                            {"text": result.output_text, "from": agent_name},
                            sched.step_id,
                        )

            elif kind == "call_verifier":
                rule_payload = verifier.check(state)
                pipeline_ctx = VerificationContext(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    prompt=task.prompt,
                    candidate_answer=last_nonempty_output,
                    reference=str(task.ground_truth) if task.ground_truth is not None else None,
                    tool_calls=list(observed_tool_calls),
                    unit_test_payload=last_unit_test_payload,
                )
                pipeline_result = await pipeline.run_async(pipeline_ctx)
                verifier_payload = self._merge_verifier_results(
                    rule_payload, pipeline_result.to_dict()
                )
                step_extras["verifier_score"] = verifier_payload["score"]
                step_extras["errors"].extend(verifier_payload["errors"])

            elif kind in ("call_memory", "call_retriever", "call_tool_executor"):
                # Phase 2 stub: just record the side effect on the trace.
                step_extras["output_summary"] = (
                    f"{kind} invoked (memory={len(memory)}, corpus={len(retriever.corpus)})"
                )

            latency_ms = int((time.perf_counter() - t_start) * 1000)
            trace.record_step(
                self._make_step(
                    sched.step_id,
                    sched.current_graph_hash,
                    action,
                    active_agent=step_extras["active_agent"],
                    output_summary=step_extras["output_summary"],
                    tool_calls=step_extras["tool_calls"],
                    errors=step_extras["errors"],
                    tokens_in=step_extras["tokens_in"],
                    tokens_out=step_extras["tokens_out"],
                    cost_rub=step_extras["cost_rub"],
                    latency_ms=latency_ms,
                    verifier_score=step_extras["verifier_score"],
                    input_msg_id=step_extras["input_msg_id"],
                )
            )
            last_step_for_trace = {
                "output_summary": step_extras["output_summary"],
                "errors": step_extras["errors"],
            }
            if step_extras["output_summary"]:
                last_nonempty_output = step_extras["output_summary"]

            sched.advance_step()
            state = self._refresh_state(
                state,
                sched,
                bus,
                last_step_for_trace,
                verifier_payload,
                produced_extras=[],
            )

            if kind == "terminated":
                break

        rule_final = verifier.check(state)
        final_ctx = VerificationContext(
            task_id=task.task_id,
            task_type=task.task_type,
            prompt=task.prompt,
            candidate_answer=last_nonempty_output,
            reference=str(task.ground_truth) if task.ground_truth is not None else None,
            tool_calls=list(observed_tool_calls),
            unit_test_payload=last_unit_test_payload,
        )
        pipeline_final = await pipeline.run_async(final_ctx)
        verification = self._merge_verifier_results(rule_final, pipeline_final.to_dict())
        finalised = trace.finalize(
            final_answer=last_nonempty_output,
            verification=verification,
            metadata={
                "controller": getattr(controller, "name", controller.__class__.__name__),
                "seed": self.config.seed,
                "agent_count": len(self.agents),
            },
        )
        return finalised

    # ----------------------------------------------------------- helpers

    def _make_step(
        self,
        step_id: int,
        graph_hash: str,
        action: dict[str, Any],
        *,
        active_agent: str | None = None,
        output_summary: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        errors: list[str] | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_rub: float = 0.0,
        latency_ms: int = 0,
        verifier_score: float | None = None,
        input_msg_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "step_id": step_id,
            "t_unix_ms": int(time.time() * 1000),
            "active_agent": active_agent,
            "input_msg_id": input_msg_id,
            "output_summary": output_summary,
            "tool_calls": tool_calls or [],
            "errors": errors or [],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "verifier_score": verifier_score,
            "current_graph_hash": graph_hash,
            "graph_action": action,
            "cost_rub": cost_rub,
        }

    @staticmethod
    def _merge_verifier_results(rule: dict[str, Any], pipeline: dict[str, Any]) -> dict[str, Any]:
        """Combine the rule-based component check with the pipeline result.

        * `passed` = AND.
        * `score`  = mean.
        * `errors` / `warnings` are concatenated (pipeline already prefixes
          its messages with `[component]`; rule-based ones are tagged
          `[rule]` here).
        * `failed_component` / `suggested_next_agent` come from whichever
          half failed first (rule has priority because a missing component
          is a more actionable signal for the controller).
        * `reward_delta` is the sum.
        """
        rule_pass = bool(rule.get("passed", False))
        pipe_pass = bool(pipeline.get("passed", False))
        scores = [float(rule.get("score", 0.0)), float(pipeline.get("score", 0.0))]

        errors: list[str] = []
        for e in rule.get("errors", []):
            errors.append(f"[rule] {e}")
        errors.extend(pipeline.get("errors", []))

        warnings: list[str] = []
        for w in rule.get("warnings", []):
            warnings.append(f"[rule] {w}")
        warnings.extend(pipeline.get("warnings", []))

        if not rule_pass:
            failed_component = rule.get("failed_component")
            suggested = rule.get("suggested_next_agent")
        else:
            failed_component = pipeline.get("failed_component")
            suggested = pipeline.get("suggested_next_agent")

        return {
            "passed": rule_pass and pipe_pass,
            "score": sum(scores) / len(scores),
            "errors": errors,
            "warnings": warnings,
            "failed_component": failed_component,
            "suggested_next_agent": suggested,
            "reward_delta": float(rule.get("reward_delta", 0.0))
            + float(pipeline.get("reward_delta", 0.0)),
        }

    def _refresh_state(
        self,
        state: RuntimeState,
        sched,
        bus,
        last_step: dict[str, Any],
        verifier_payload: dict[str, Any] | None,
        produced_extras: list[str],
    ) -> RuntimeState:
        return RuntimeState(
            task_id=state.task_id,
            task_type=state.task_type,
            prompt=state.prompt,
            step_id=sched.step_id,
            available_agents=state.available_agents,
            pending_inbox={a: bus.pending(a) for a in state.available_agents},
            last_active_agent=sched.last_active_agent,
            last_output_summary=last_step.get("output_summary"),
            last_verifier_score=(verifier_payload or {}).get("score", state.last_verifier_score),
            last_verifier_passed=(verifier_payload or {}).get("passed", state.last_verifier_passed),
            last_verifier_failed_component=(verifier_payload or {}).get(
                "failed_component", state.last_verifier_failed_component
            ),
            last_errors=last_step.get("errors") or state.last_errors,
            produced_components=state.produced_components | set(produced_extras),
            extras=state.extras,
        )
