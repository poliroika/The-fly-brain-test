"""`web_search` — stub returning canned results from an injected fixture.

A real impl (HTTP / SerpAPI / etc.) lands later. For Phase 2 we just need
a deterministic, dependency-free tool the runtime can hand to research
agents. The fixture maps lower-cased query substrings → list of
`{title, snippet, url}` dicts; first matching key wins. If no key
matches, an empty result list is returned (still `ok=True`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flybrain.runtime.tools.base import ToolResult


@dataclass(slots=True)
class WebSearchTool:
    name: str = "web_search"
    fixture: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    """Lower-cased substring → list of result dicts."""

    def run(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                name=self.name,
                ok=False,
                output=None,
                error="missing `query`",
                args=args,
            )
        q = query.lower()
        for key, results in self.fixture.items():
            if key.lower() in q:
                return ToolResult(
                    name=self.name,
                    ok=True,
                    output={"query": query, "results": list(results)},
                    args=args,
                )
        return ToolResult(
            name=self.name,
            ok=True,
            output={"query": query, "results": []},
            args=args,
        )
