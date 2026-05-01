"""High-level Python entry points for the Rust graph builder.

Each function imports `flybrain.flybrain_native` lazily so unit tests that
only touch dataclasses still work in environments where the PyO3 module
hasn't been built (e.g. doc-builds, mypy-only runs).
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from .dataclasses import BuildReport, ClusterAssignment, FlyGraph

CompressionMethod = Literal["region_agg", "celltype_agg", "louvain", "leiden", "spectral"]
DEFAULT_K_VALUES: tuple[int, ...] = (32, 64, 128, 256)


def _native() -> Any:
    return import_module("flybrain.flybrain_native")


def build_synthetic(num_nodes: int, seed: int = 42) -> FlyGraph:
    """Deterministic fly-inspired graph generator (Rust side)."""
    return FlyGraph.from_dict(_native().build_synthetic(num_nodes, seed))


def load_zenodo(
    dir: str | Path | None = None,
    *,
    neurons: str | Path | None = None,
    connections: str | Path | None = None,
) -> FlyGraph:
    """Parse a FlyWire / Zenodo CSV pair from a local directory or explicit paths."""
    n = _native()
    if neurons is not None and connections is not None:
        return FlyGraph.from_dict(n.load_zenodo_pair(str(neurons), str(connections)))
    if dir is None:
        raise ValueError("either `dir` or both `neurons` and `connections` must be provided")
    return FlyGraph.from_dict(n.load_zenodo(str(dir)))


def save(graph: FlyGraph, path: str | Path) -> None:
    """Write a `FlyGraph` to a `.fbg` file (gzip-wrapped JSON)."""
    _native().save_graph(graph.to_dict(), str(path))


def load(path: str | Path) -> FlyGraph:
    """Read a `.fbg` file produced by `save` (or `flybrain build`)."""
    return FlyGraph.from_dict(_native().load_graph(str(path)))


def compress(
    graph: FlyGraph,
    method: CompressionMethod,
    target_k: int,
    seed: int = 42,
) -> ClusterAssignment:
    """Run a compression method and return the cluster assignment."""
    return ClusterAssignment.from_dict(_native().compress(graph.to_dict(), method, target_k, seed))


def compress_and_aggregate(
    graph: FlyGraph,
    method: CompressionMethod,
    target_k: int,
    seed: int = 42,
) -> FlyGraph:
    """Compress + collapse to a K-node graph."""
    return FlyGraph.from_dict(
        _native().compress_and_aggregate(graph.to_dict(), method, target_k, seed)
    )


def modularity(graph: FlyGraph, assignment: list[int]) -> float:
    """Directed Newman / Leicht modularity Q for a partition."""
    return float(_native().graph_modularity(graph.to_dict(), assignment))


def build(
    *,
    source_spec: dict[str, Any],
    method: CompressionMethod,
    target_k: int,
    output: str | Path,
    seed: int = 42,
) -> BuildReport:
    """Full pipeline: source → compression → `.fbg` + node_metadata.json on disk.

    `source_spec` must be one of:
        {"kind": "synthetic", "num_nodes": 2048, "seed": 42}
        {"kind": "zenodo_dir", "dir": "data/flywire/"}
        {"kind": "zenodo_csv", "neurons": "...", "connections": "..."}
    """
    return BuildReport.from_dict(_native().build(source_spec, method, target_k, str(output), seed))


def build_default_set(
    out_dir: str | Path = "data/flybrain",
    *,
    method: CompressionMethod = "louvain",
    seed: int = 42,
    num_nodes: int = 2048,
) -> list[BuildReport]:
    """Build the four canonical K∈{32,64,128,256} `.fbg` files used by the controller.

    All four use the same synthetic source + seed, so the only thing that
    changes between K values is the cluster count. Fast (<1 s for K=32 on a
    laptop) and deterministic.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[BuildReport] = []
    source_spec = {"kind": "synthetic", "num_nodes": num_nodes, "seed": seed}
    for k in DEFAULT_K_VALUES:
        reports.append(
            build(
                source_spec=source_spec,
                method=method,
                target_k=k,
                output=out_dir / f"fly_graph_{k}.fbg",
                seed=seed,
            )
        )
    return reports
