"""Phase-1 tests for the Python graph builder + native bindings."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from flybrain.graph import (
    BuildReport,
    ClusterAssignment,
    FlyGraph,
    NodeMetadata,
    build,
    build_default_set,
    build_synthetic,
    compress,
    compress_and_aggregate,
    load,
    load_zenodo,
    modularity,
    save,
)


@pytest.fixture
def small_graph() -> FlyGraph:
    return build_synthetic(num_nodes=128, seed=7)


def test_synthetic_is_deterministic_python_side() -> None:
    a = build_synthetic(num_nodes=64, seed=1)
    b = build_synthetic(num_nodes=64, seed=1)
    assert a.num_nodes == b.num_nodes == 64
    assert a.edge_index == b.edge_index
    assert a.edge_weight == b.edge_weight


def test_dataclass_round_trip(small_graph: FlyGraph) -> None:
    d = small_graph.to_dict()
    g2 = FlyGraph.from_dict(d)
    assert g2.num_nodes == small_graph.num_nodes
    assert g2.num_edges == small_graph.num_edges
    assert g2.edge_index == small_graph.edge_index
    assert all(isinstance(n, NodeMetadata) for n in g2.nodes)


def test_save_load_round_trip(tmp_path: Path, small_graph: FlyGraph) -> None:
    p = tmp_path / "g.fbg"
    save(small_graph, p)
    g2 = load(p)
    assert g2.num_nodes == small_graph.num_nodes
    assert g2.num_edges == small_graph.num_edges
    assert g2.edge_index == small_graph.edge_index
    assert g2.is_excitatory == small_graph.is_excitatory


@pytest.mark.parametrize("method", ["region_agg", "celltype_agg", "louvain", "leiden", "spectral"])
def test_compress_returns_assignment(small_graph: FlyGraph, method: str) -> None:
    target_k = 8 if method in {"louvain", "leiden", "spectral"} else None
    if target_k is None:
        # Trivial methods don't take a K — pass any value, it's ignored.
        a: ClusterAssignment = compress(small_graph, method, 8, seed=0)  # type: ignore[arg-type]
    else:
        a = compress(small_graph, method, target_k, seed=0)  # type: ignore[arg-type]
    assert len(a.assignment) == small_graph.num_nodes
    assert a.num_clusters >= 1
    if method in {"louvain", "leiden", "spectral"}:
        assert a.num_clusters == 8


@pytest.mark.parametrize("k", [4, 8, 16])
def test_louvain_deterministic(small_graph: FlyGraph, k: int) -> None:
    a = compress(small_graph, "louvain", k, seed=42)
    b = compress(small_graph, "louvain", k, seed=42)
    assert a.assignment == b.assignment
    assert a.num_clusters == b.num_clusters == k


def test_compress_and_aggregate_returns_compressed_graph(small_graph: FlyGraph) -> None:
    g2 = compress_and_aggregate(small_graph, "louvain", 8, seed=0)
    assert g2.num_nodes == 8
    assert g2.num_edges <= small_graph.num_edges
    assert "compression" in g2.provenance


def test_modularity_zero_for_singletons(small_graph: FlyGraph) -> None:
    assn = list(range(small_graph.num_nodes))
    q = modularity(small_graph, assn)
    # Singleton partition modularity is non-positive (and small in magnitude).
    assert q <= 1e-6


@pytest.mark.parametrize("k", [16, 32])
def test_build_writes_files(tmp_path: Path, k: int) -> None:
    out = tmp_path / f"fly_graph_{k}.fbg"
    report = build(
        source_spec={"kind": "synthetic", "num_nodes": 256, "seed": 1},
        method="louvain",
        target_k=k,
        output=out,
        seed=1,
    )
    assert isinstance(report, BuildReport)
    assert report.compressed_num_nodes == k
    assert Path(report.fbg_path).exists()
    assert Path(report.metadata_json_path).exists()


def test_build_default_set_creates_four_files(tmp_path: Path) -> None:
    reports = build_default_set(out_dir=tmp_path, num_nodes=512, seed=1)
    assert [r.target_k for r in reports] == [32, 64, 128, 256]
    for r in reports:
        assert Path(r.fbg_path).exists()
        assert Path(r.metadata_json_path).exists()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: ("true" if v is True else "false" if v is False else v) for k, v in row.items()}
            )


def test_zenodo_csv_loader(tmp_path: Path) -> None:
    neurons = tmp_path / "neurons.csv"
    connections = tmp_path / "connections.csv"
    _write_csv(
        neurons,
        [
            {"id": 100, "cell_type": "kc", "region": "MB"},
            {"id": 101, "cell_type": "pn", "region": "AL"},
            {"id": 102, "cell_type": "kc", "region": "MB"},
        ],
    )
    _write_csv(
        connections,
        [
            {"pre_root_id": 100, "post_root_id": 101, "syn_count": 4, "is_excitatory": True},
            {"pre_root_id": 101, "post_root_id": 102, "syn_count": 2, "is_excitatory": False},
            # orphan — should be dropped:
            {"pre_root_id": 100, "post_root_id": 999, "syn_count": 7, "is_excitatory": True},
        ],
    )
    g = load_zenodo(dir=None, neurons=neurons, connections=connections)
    assert g.num_nodes == 3
    assert g.num_edges == 2
    assert int(g.provenance.get("dropped_orphans", 0)) == 1
