# Graph pipeline (Phase 1)

The graph pipeline turns a fly connectome (real or synthetic) into a
compressed `K`-node graph used as the structural prior for the Phase 5
controller. Everything in this document lives in
`crates/flybrain-graph/` (Rust) with PyO3 bindings exposed as
`flybrain.graph` (Python).

## Data flow

```
       ┌──────────────────┐
       │  source loader   │   Synthetic / Zenodo CSV
       └────────┬─────────┘
                │ FlyGraph (full connectome)
                ▼
       ┌──────────────────┐
       │  compression     │   region_agg / celltype_agg
       │                  │   louvain / leiden / spectral
       └────────┬─────────┘
                │ ClusterAssignment (assignment + num_clusters)
                ▼
       ┌──────────────────┐
       │  aggregate       │   collapse clusters → K-node FlyGraph
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │  format::save_fbg│   gzip-wrapped JSON, magic = "FBG1"
       └────────┬─────────┘
                │
                ▼
       data/flybrain/fly_graph_<K>.fbg
       data/flybrain/fly_graph_<K>.fbg.node_metadata.json
```

## Data contracts

| Type | Where | Purpose |
| --- | --- | --- |
| `FlyGraph` | `flybrain-core::graph` | COO-form weighted directed graph + per-node metadata + provenance map |
| `ClusterAssignment` | `flybrain-graph::compression` | `assignment[i] = cluster_id`, `num_clusters`, optional cluster `labels` |
| `BuildRequest` | `flybrain-graph::builder` | source spec + method + K + seed + output path |
| `BuildReport` | `flybrain-graph::builder` | counts + paths + measured modularity |

`FlyGraph.provenance` is a `BTreeMap<String, serde_json::Value>` so we can
attach arbitrary metadata (compression method, K, cluster sizes, dropped
orphan count, source kind) without a schema migration. Notebooks can read
the same data from the companion `.node_metadata.json`.

## Compression methods

| Method | Type | K input | Determinism | Time complexity (n nodes, m edges) | Use case |
| --- | --- | --- | --- | --- | --- |
| `region_agg` | trivial | ignored | none needed | O(n + m) | Sanity baseline; uses `node.region` |
| `celltype_agg` | trivial | ignored | none needed | O(n + m) | Sanity baseline; uses `node.node_type` |
| `louvain` | modularity max | clamped to K | seeded LCG node order | O(iters · m) | Default — strong Q, fast |
| `leiden` | louvain + refinement | clamped to K | seeded LCG | O(iters · m + n) | Slightly higher Q than Louvain on tree-like graphs |
| `spectral` | bottom-K Laplacian eigvecs + k-means | exact K | seeded init vectors + spread init | O(K · iters · m + K² · n) | Use when modularity is a poor objective for your graph |

All methods are **deterministic for a fixed (graph, K, seed) tuple** —
required by the Phase 1 stability tests
(`compression::louvain::tests::deterministic_for_same_seed`,
`compression::spectral::tests::deterministic_for_same_seed`,
`compression::leiden::tests::deterministic`).

### Clamping to K

Louvain and Leiden naturally produce a *variable* number of communities.
We then `clamp_to_k`:

* If too few communities → split the largest community by re-running a
  cheap deterministic 2-way partition until we reach K.
* If too many communities → merge the two smallest until we reach K.

Spectral runs k-means at K directly, then routes through `clamp_to_k` to
guarantee the exact count after empty centroids drop out.

## File format: `.fbg`

```
+--------+----------------------------------------------+
| MAGIC  |               gzip(json(FlyGraph))           |
| (4 B)  |                                              |
+--------+----------------------------------------------+
  "FBG1"
```

* `MAGIC` is checked on `load_fbg` to guard against feeding the loader
  arbitrary gzip files.
* The body is JSON-of-`FlyGraph` (not bincode) because `FlyGraph.provenance`
  uses `serde_json::Value` and bincode 1.x cannot deserialise untyped JSON
  values. JSON adds <2× size overhead vs. bincode for our K ≤ 256 graphs.
* The companion `.node_metadata.json` contains the same `nodes` +
  `provenance` fields uncompressed for fast inspection.

## Public API surface

### Rust

```rust
use flybrain_graph::{
    builder::{build, build_default, BuildRequest, BuildSource, CompressionMethod},
    compression::{louvain::louvain, leiden::leiden, spectral::spectral, modularity},
    format::{load_fbg, save_fbg},
    synthetic::synthetic_fly_graph,
    zenodo::{load_zenodo_csv, load_zenodo_dir},
};
```

### Python

```python
from flybrain.graph import (
    build, build_default_set,
    build_synthetic, load_zenodo,
    compress, compress_and_aggregate,
    save, load,
    modularity,
    BuildReport, ClusterAssignment, FlyGraph, NodeMetadata,
)
```

## CLI

The Rust binary handles fast deterministic builds:

```bash
# Single build
flybrain build --source synthetic --num-nodes 4096 --method louvain -k 64 \
    --seed 42 -o data/flybrain/fly_graph_64.fbg

# Default K∈{32,64,128,256} batch
flybrain build --all
```

The Python CLI (`flybrain-py build …`) takes the same flags and is the
recommended entry point for notebooks and ML scripts because it returns
typed dataclasses and is easy to import.

## Testing

* `cargo test -p flybrain-graph` — 26 tests covering compression
  determinism, aggregation correctness, format round-trip, builder smoke
  tests, Zenodo CSV parsing, and the K∈{32,64,128,256} default set.
* `pytest tests/python/unit/test_graph_builder.py` — 17 tests covering
  the PyO3 bindings, dataclasses, and pipeline.

## Phase 1 exit criteria

* For `K ∈ {32, 64, 128, 256}` we produce `.fbg` + `.node_metadata.json`
  files (covered by `build_default_writes_four_files`).
* All compression methods stable for fixed `(graph, K, seed)` tuples
  (covered by per-method `deterministic_*` tests).
* Round-trip `load_fbg(save_fbg(g)) == g` (covered by `format::round_trip`).
* CLI integration: `flybrain build --all` runs end-to-end on a fresh
  checkout (manual smoke test; CI runs cargo + pytest).
