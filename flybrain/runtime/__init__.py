"""MAS runtime: Python wrapper around `flybrain_native.{Scheduler,MessageBus,
TraceWriter}` plus the per-agent loop.

Phase 2 ships:

* `Agent`, `AgentSpec`, `AgentStepResult` — Python wrapper of one MAS agent.
* `MAS`, `MASConfig`, `Task` — top-level runner that ties controller,
  scheduler, message bus, trace writer, agents, memory, retriever
  together.
* `RuntimeState` — controller observation.
* `tools/` (`PythonExecTool`, `FileTool`, `WebSearchTool`,
  `UnitTesterTool`, `ToolRegistry`).
* `memory/` (`EpisodicMemory`, `VectorMemory`).
* `retriever/` (`BM25Retriever`).
"""

from __future__ import annotations

from flybrain.runtime.agent import Agent, AgentSpec, AgentStepResult
from flybrain.runtime.runner import MAS, MASConfig, Task
from flybrain.runtime.state import RuntimeState

__all__ = [
    "MAS",
    "Agent",
    "AgentSpec",
    "AgentStepResult",
    "MASConfig",
    "RuntimeState",
    "Task",
]
