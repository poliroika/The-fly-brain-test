"""Unit tests for the runtime tool registry and concrete tools."""

from __future__ import annotations

from pathlib import Path

from flybrain.runtime.tools import (
    FileTool,
    PythonExecTool,
    ToolRegistry,
    UnitTesterTool,
    WebSearchTool,
    default_tool_registry,
)


def test_default_registry_lists_four_tools() -> None:
    reg = default_tool_registry()
    assert reg.names() == ["file_tool", "python_exec", "unit_tester", "web_search"]


def test_registry_unknown_tool_returns_failure() -> None:
    reg = ToolRegistry()
    res = reg.call("ghost", {})
    assert res.ok is False
    assert "unknown tool" in (res.error or "")


def test_python_exec_runs_simple_print() -> None:
    res = PythonExecTool().run({"code": "print(2 + 2)"})
    assert res.ok is True
    assert "4" in res.output["stdout"]


def test_python_exec_rejects_forbidden_token() -> None:
    res = PythonExecTool().run({"code": "import os\nprint(os.listdir('.'))"})
    assert res.ok is False
    assert "forbidden token" in (res.error or "")


def test_python_exec_times_out_on_infinite_loop() -> None:
    res = PythonExecTool(timeout_s=0.5).run({"code": "while True: pass"})
    assert res.ok is False
    assert "timeout" in (res.error or "")


def test_python_exec_missing_code_arg() -> None:
    res = PythonExecTool().run({})
    assert res.ok is False


def test_file_tool_reads_file(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    res = FileTool(sandbox_root=tmp_path).run({"op": "read", "path": "hello.txt"})
    assert res.ok is True
    assert res.output["text"] == "hello world"


def test_file_tool_lists_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    res = FileTool(sandbox_root=tmp_path).run({"op": "list", "path": "."})
    assert res.ok is True
    assert sorted(res.output["entries"]) == ["a.txt", "b.txt"]


def test_file_tool_rejects_path_escape(tmp_path: Path) -> None:
    res = FileTool(sandbox_root=tmp_path).run({"op": "read", "path": "../etc/passwd"})
    assert res.ok is False
    assert "escapes sandbox" in (res.error or "")


def test_file_tool_rejects_absolute_path(tmp_path: Path) -> None:
    res = FileTool(sandbox_root=tmp_path).run({"op": "read", "path": "/etc/passwd"})
    assert res.ok is False
    assert "absolute paths" in (res.error or "")


def test_web_search_returns_canned_results() -> None:
    tool = WebSearchTool(
        fixture={
            "fly brain": [{"title": "FlyWire", "snippet": "Drosophila connectome", "url": "x"}]
        }
    )
    res = tool.run({"query": "tell me about the fly brain"})
    assert res.ok is True
    assert res.output["results"][0]["title"] == "FlyWire"


def test_web_search_unknown_query_returns_empty() -> None:
    res = WebSearchTool().run({"query": "anything"})
    assert res.ok is True
    assert res.output["results"] == []


def test_unit_tester_passes_with_correct_implementation() -> None:
    res = UnitTesterTool().run(
        {
            "code": "def add(a,b): return a+b",
            "tests": "assert add(2,3) == 5",
        }
    )
    assert res.ok is True
    assert res.output["all_passed"] is True


def test_unit_tester_fails_on_assertion_error() -> None:
    res = UnitTesterTool().run(
        {
            "code": "def add(a,b): return a-b",
            "tests": "assert add(2,3) == 5",
        }
    )
    assert res.ok is False
    assert "AssertionError" in (res.error or "")
