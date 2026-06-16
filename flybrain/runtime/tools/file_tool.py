"""`file_tool` — read-only file inspection inside a sandbox root.

Supports `read` (full file) and `list` (directory listing). Refuses any
path that escapes the sandbox via `..` or absolute paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flybrain.runtime.tools.base import ToolResult


@dataclass(slots=True)
class FileTool:
    name: str = "file_tool"
    sandbox_root: Path = Path(".")
    max_bytes: int = 64_000

    def _resolve(self, raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            raise ValueError(f"absolute paths not allowed: {raw}")
        target = (self.sandbox_root / p).resolve()
        root = self.sandbox_root.resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"path escapes sandbox: {raw}")
        return target

    def run(self, args: dict[str, Any]) -> ToolResult:
        op = args.get("op", "read")
        path_arg = args.get("path", "")
        if not isinstance(path_arg, str) or not path_arg:
            return ToolResult(
                name=self.name,
                ok=False,
                output=None,
                error="missing `path`",
                args=args,
            )
        try:
            target = self._resolve(path_arg)
        except ValueError as e:
            return ToolResult(name=self.name, ok=False, output=None, error=str(e), args=args)

        if op == "read":
            if not target.is_file():
                return ToolResult(
                    name=self.name,
                    ok=False,
                    output=None,
                    error=f"not a file: {path_arg}",
                    args=args,
                )
            data = target.read_bytes()[: self.max_bytes]
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")
            return ToolResult(
                name=self.name,
                ok=True,
                output={"text": text, "bytes_read": len(data)},
                args=args,
            )
        if op == "list":
            if not target.is_dir():
                return ToolResult(
                    name=self.name,
                    ok=False,
                    output=None,
                    error=f"not a directory: {path_arg}",
                    args=args,
                )
            entries = sorted(p.name for p in target.iterdir())
            return ToolResult(
                name=self.name,
                ok=True,
                output={"entries": entries},
                args=args,
            )
        return ToolResult(
            name=self.name,
            ok=False,
            output=None,
            error=f"unknown op {op!r}; expected 'read' or 'list'",
            args=args,
        )
