"""`python_exec` — run a snippet of Python in a fresh subprocess with a
timeout. Used by the Coder / Debugger agents.

Stdout / stderr are captured. The tool refuses to run if the snippet
contains obvious filesystem-mutating patterns; this is a *safety net*,
not a sandbox — for production we would call out to a real sandbox
(Firecracker / Docker). For Phase 2 we just want a deterministic,
timeout-safe runner that is good enough for `(2 + 2)` and small
`unittest`-style assertions.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from flybrain.runtime.tools.base import ToolResult

_FORBIDDEN_TOKENS = (
    "import os",
    "import shutil",
    "import subprocess",
    "open(",
    "__import__",
    "eval(",
    "exec(",
    "socket",
)


@dataclass(slots=True)
class PythonExecTool:
    name: str = "python_exec"
    timeout_s: float = 5.0
    allow_unsafe: bool = False

    def run(self, args: dict[str, Any]) -> ToolResult:
        code = args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(
                name=self.name,
                ok=False,
                output=None,
                error="missing required arg `code`",
                args=args,
            )
        if not self.allow_unsafe:
            for forbidden in _FORBIDDEN_TOKENS:
                if forbidden in code:
                    return ToolResult(
                        name=self.name,
                        ok=False,
                        output=None,
                        error=f"forbidden token in code: {forbidden!r}",
                        args=args,
                    )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                name=self.name,
                ok=False,
                output=None,
                error=f"timeout after {self.timeout_s}s",
                args=args,
            )

        ok = proc.returncode == 0
        return ToolResult(
            name=self.name,
            ok=ok,
            output={"stdout": proc.stdout, "stderr": proc.stderr},
            error=None if ok else f"exit={proc.returncode}: {proc.stderr.strip()[:200]}",
            args=args,
        )
