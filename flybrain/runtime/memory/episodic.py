"""Append-only episodic memory: a flat list of `(tag, content)` entries.

Each MAS run starts with a fresh `EpisodicMemory`. Cross-run / persistent
memory will arrive in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryEntry:
    tag: str
    content: Any
    step_id: int = 0


@dataclass(slots=True)
class EpisodicMemory:
    entries: list[MemoryEntry] = field(default_factory=list)

    def append(self, tag: str, content: Any, step_id: int = 0) -> None:
        self.entries.append(MemoryEntry(tag=tag, content=content, step_id=step_id))

    def by_tag(self, tag: str) -> list[MemoryEntry]:
        return [e for e in self.entries if e.tag == tag]

    def latest(self, tag: str) -> MemoryEntry | None:
        for e in reversed(self.entries):
            if e.tag == tag:
                return e
        return None

    def __len__(self) -> int:
        return len(self.entries)
