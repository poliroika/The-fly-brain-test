"""Python-side CLI parallel to the Rust `flybrain` binary.

Phase 1 wires `info` and `build`. Other subcommands land alongside their
respective phases. The Rust CLI handles deterministic, native-only commands
(graph build / sim / etc.); the Python CLI is the entry point for anything
that needs PyTorch or the LLM, and a thin convenience wrapper for the
graph builder so notebooks / scripts have one consistent API.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from flybrain import __version__


def _add_build_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "build",
        help="Build a compressed FlyBrain graph (Phase 1).",
        description=(
            "Build a compressed FlyBrain graph from a connectome source. "
            "Wraps `flybrain.graph.build` / the Rust `flybrain build` CLI. "
            "Pass --all to produce the K∈{32,64,128,256} default set in one go."
        ),
    )
    p.add_argument(
        "--source",
        choices=["synthetic", "zenodo_dir", "zenodo_csv"],
        default="synthetic",
    )
    p.add_argument("--num-nodes", type=int, default=2048)
    p.add_argument("--zenodo-dir", type=Path, default=None)
    p.add_argument("--zenodo-neurons", type=Path, default=None)
    p.add_argument("--zenodo-connections", type=Path, default=None)
    p.add_argument(
        "--method",
        choices=["region_agg", "celltype_agg", "louvain", "leiden", "spectral"],
        default="louvain",
    )
    p.add_argument("-k", "--k", type=int, default=64, dest="k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", "-o", type=Path, default=None)
    p.add_argument(
        "--all",
        action="store_true",
        help="Run the K∈{32,64,128,256} batch instead of a single build.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/flybrain"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flybrain-py",
        description="FlyBrain Optimizer Python CLI",
    )
    parser.add_argument("--version", action="version", version=f"flybrain-py {__version__}")
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("info", help="Print build info and detected native module")
    _add_build_subparser(sub)
    sub.add_parser("sim", help="Run simulation pretraining (Phase 6; not yet implemented)")
    sub.add_parser("train", help="Run training (Phase 7/8; not yet implemented)")
    sub.add_parser("bench", help="Run benchmark suite (Phase 10; not yet implemented)")
    sub.add_parser("report", help="Build the final report (Phase 11; not yet implemented)")

    return parser


def _cmd_info() -> int:
    try:
        import importlib

        native = importlib.import_module("flybrain.flybrain_native")
    except ImportError:
        native = None
    print(f"flybrain-py {__version__}")
    if native is not None:
        print(f"flybrain_native {getattr(native, '__version__', '?')}")
        modinfo = getattr(native, "__modinfo__", None)
        if modinfo is not None:
            print(f"phase: {modinfo.get('phase')}")
    else:
        print("flybrain_native: not built (run `maturin develop`)")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from flybrain.graph import build, build_default_set

    if args.all:
        reports = build_default_set(
            out_dir=args.out_dir, method=args.method, seed=args.seed, num_nodes=args.num_nodes
        )
        for r in reports:
            print(json.dumps(asdict(r), indent=2))
        return 0

    if args.source == "synthetic":
        spec: dict[str, object] = {
            "kind": "synthetic",
            "num_nodes": args.num_nodes,
            "seed": args.seed,
        }
    elif args.source == "zenodo_dir":
        if args.zenodo_dir is None:
            print("--source zenodo_dir requires --zenodo-dir", file=sys.stderr)
            return 2
        spec = {"kind": "zenodo_dir", "dir": str(args.zenodo_dir)}
    else:  # zenodo_csv
        if args.zenodo_neurons is None or args.zenodo_connections is None:
            print(
                "--source zenodo_csv requires --zenodo-neurons and --zenodo-connections",
                file=sys.stderr,
            )
            return 2
        spec = {
            "kind": "zenodo_csv",
            "neurons": str(args.zenodo_neurons),
            "connections": str(args.zenodo_connections),
        }

    output = args.output
    if output is None:
        output = Path(f"data/flybrain/fly_graph_{args.k}.fbg")
    output.parent.mkdir(parents=True, exist_ok=True)

    report = build(
        source_spec=spec,
        method=args.method,
        target_k=args.k,
        output=output,
        seed=args.seed,
    )
    print(json.dumps(asdict(report), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None or args.command == "info":
        return _cmd_info()
    if args.command == "build":
        return _cmd_build(args)
    print(f"flybrain-py {args.command}: not yet implemented", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
