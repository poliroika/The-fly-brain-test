"""Phase-1 Python wrapper around the Rust graph builder.

Most users just want:

    from flybrain.graph import build_default_set, build, FlyGraph

The native module (`flybrain.flybrain_native`) is the source of truth; this
package adds typed dataclasses and convenience helpers so notebooks /
training scripts don't have to deal with raw dicts.
"""

from __future__ import annotations

from .dataclasses import BuildReport, ClusterAssignment, FlyGraph, NodeMetadata
from .pipeline import (
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

__all__ = [
    "BuildReport",
    "ClusterAssignment",
    "FlyGraph",
    "NodeMetadata",
    "build",
    "build_default_set",
    "build_synthetic",
    "compress",
    "compress_and_aggregate",
    "load",
    "load_zenodo",
    "modularity",
    "save",
]
