"""Tool protocol shared by `python_exec`, `web_search`, `file_tool`,
`unit_tester`. All concrete tools are sync — async tools (HTTP search)
will arrive in a later phase together with a real network impl.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ToolResult:
    """Return value from `Tool.run(args)`. Mirrors `ToolCall` in the Rust
    trace schema (one extra field, `output`, that the agent layer reads
    but does not persist verbatim into the trace)."""

    name: str
    ok: bool
    output: Any
    error: str | None = None
    latency_ms: int = 0
    args: dict[str, Any] = field(default_factory=dict)

    def as_tool_call(self) -> dict[str, Any]:
        """Project to the `ToolCall` JSON shape used by the trace."""
        return {
            "name": self.name,
            "args": self.args,
            "ok": self.ok,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class Tool(Protocol):
    """A side-effect tool callable from inside an agent step."""

    name: str

    def run(self, args: dict[str, Any]) -> ToolResult: ...


def time_call(name: str, args: dict[str, Any], fn) -> ToolResult:
    """Helper that times `fn(args)` and packs its return into a `ToolResult`.
    `fn` returns either an `(ok, output, error)` tuple or raises."""

    started = time.perf_counter()
    try:
        result = fn(args)
        latency = int((time.perf_counter() - started) * 1000)
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, tuple) and len(result) == 3:
            ok, output, error = result
            return ToolResult(
                name=name,
                ok=ok,
                output=output,
                error=error,
                latency_ms=latency,
                args=args,
            )
        return ToolResult(
            name=name,
            ok=True,
            output=result,
            error=None,
            latency_ms=latency,
            args=args,
        )
    except Exception as e:
        latency = int((time.perf_counter() - started) * 1000)
        return ToolResult(
            name=name,
            ok=False,
            output=None,
            error=f"{type(e).__name__}: {e}",
            latency_ms=latency,
            args=args,
        )


@dataclass(slots=True)
class ToolRegistry:
    """Name → Tool lookup that agents consult by tool name."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                name=name,
                ok=False,
                output=None,
                error=f"unknown tool {name}",
                args=args,
            )
        return time_call(name, args, lambda a: tool.run(a))
