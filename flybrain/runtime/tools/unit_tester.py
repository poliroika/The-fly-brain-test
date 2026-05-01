"""`unit_tester` — run a code snippet and an `assert`-based test snippet,
return pass/fail. Built on top of `PythonExecTool` so we share its timeout
behaviour and stdout capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flybrain.runtime.tools.base import ToolResult
from flybrain.runtime.tools.python_exec import PythonExecTool


@dataclass(slots=True)
class UnitTesterTool:
    name: str = "unit_tester"
    timeout_s: float = 5.0
    runner: PythonExecTool = field(default_factory=PythonExecTool)

    def __post_init__(self) -> None:
        # Match our timeout into the wrapped runner.
        self.runner.timeout_s = self.timeout_s

    def run(self, args: dict[str, Any]) -> ToolResult:
        code = args.get("code")
        tests = args.get("tests")
        if not isinstance(code, str) or not isinstance(tests, str):
            return ToolResult(
                name=self.name,
                ok=False,
                output=None,
                error="missing `code` and/or `tests` (both must be strings)",
                args=args,
            )

        program = code + "\n\n" + tests
        sub = self.runner.run({"code": program})
        if not sub.ok:
            return ToolResult(
                name=self.name,
                ok=False,
                output=sub.output,
                error=sub.error,
                latency_ms=sub.latency_ms,
                args=args,
            )
        return ToolResult(
            name=self.name,
            ok=True,
            output={"all_passed": True, "stdout": sub.output["stdout"] if sub.output else ""},
            latency_ms=sub.latency_ms,
            args=args,
        )
