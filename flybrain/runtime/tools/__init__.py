"""Runtime tools (`python_exec`, `web_search`, `file_tool`, `unit_tester`)."""

from __future__ import annotations

from flybrain.runtime.tools.base import Tool, ToolRegistry, ToolResult
from flybrain.runtime.tools.file_tool import FileTool
from flybrain.runtime.tools.python_exec import PythonExecTool
from flybrain.runtime.tools.unit_tester import UnitTesterTool
from flybrain.runtime.tools.web_search import WebSearchTool


def default_tool_registry() -> ToolRegistry:
    """Registry with deterministic-by-default tools wired in."""
    reg = ToolRegistry()
    reg.register(PythonExecTool())
    reg.register(FileTool())
    reg.register(WebSearchTool())
    reg.register(UnitTesterTool())
    return reg


__all__ = [
    "FileTool",
    "PythonExecTool",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "UnitTesterTool",
    "WebSearchTool",
    "default_tool_registry",
]
