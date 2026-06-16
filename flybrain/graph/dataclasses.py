"""Typed wrappers around the JSON dicts the Rust bindings return.

The Rust side returns plain dicts (cheap to serialise and avoids churn while
the data contract evolves). These dataclasses just provide nicer attribute
access and let mypy + IDEs catch field typos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeMetadata:
    id: int
    node_type: str = ""
    region: str = ""
    features: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeMetadata:
        return cls(
            id=int(d.get("id", 0)),
            node_type=str(d.get("node_type", "")),
            region=str(d.get("region", "")),
            features=[float(x) for x in d.get("features", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "region": self.region,
            "features": list(self.features),
        }


@dataclass
class FlyGraph:
    """A fly-connectome graph (or a compressed K-node prior).

    Edge layout is COO: `edge_index[i] = (src, dst)` with corresponding
    `edge_weight[i]` and optional `is_excitatory[i]`.
    """

    num_nodes: int
    edge_index: list[tuple[int, int]]
    edge_weight: list[float]
    is_excitatory: list[bool]
    nodes: list[NodeMetadata]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        return len(self.edge_index)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FlyGraph:
        return cls(
            num_nodes=int(d["num_nodes"]),
            edge_index=[(int(s), int(t)) for s, t in d.get("edge_index", [])],
            edge_weight=[float(x) for x in d.get("edge_weight", [])],
            is_excitatory=[bool(x) for x in d.get("is_excitatory", [])],
            nodes=[NodeMetadata.from_dict(n) for n in d.get("nodes", [])],
            provenance=dict(d.get("provenance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "edge_index": [list(e) for e in self.edge_index],
            "edge_weight": list(self.edge_weight),
            "is_excitatory": list(self.is_excitatory),
            "nodes": [n.to_dict() for n in self.nodes],
            "provenance": dict(self.provenance),
        }


@dataclass
class ClusterAssignment:
    """Output of a compression pass."""

    assignment: list[int]
    num_clusters: int
    labels: list[str] | None = None
    modularity: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterAssignment:
        return cls(
            assignment=[int(x) for x in d.get("assignment", [])],
            num_clusters=int(d.get("num_clusters", 0)),
            labels=list(d["labels"]) if d.get("labels") is not None else None,
            modularity=float(d.get("modularity", 0.0)),
        )


@dataclass
class BuildReport:
    """Result of a full builder run."""

    source_num_nodes: int
    source_num_edges: int
    compressed_num_nodes: int
    compressed_num_edges: int
    method: str
    target_k: int
    fbg_path: str
    metadata_json_path: str
    modularity_directed: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BuildReport:
        return cls(
            source_num_nodes=int(d["source_num_nodes"]),
            source_num_edges=int(d["source_num_edges"]),
            compressed_num_nodes=int(d["compressed_num_nodes"]),
            compressed_num_edges=int(d["compressed_num_edges"]),
            method=str(d["method"]),
            target_k=int(d["target_k"]),
            fbg_path=str(d["fbg_path"]),
            metadata_json_path=str(d["metadata_json_path"]),
            modularity_directed=float(d["modularity_directed"]),
        )
