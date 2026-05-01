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


def _add_bench_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "bench",
        help="Run benchmark suite (Phase 10).",
        description=(
            "Drive the four Phase-10 benchmarks (HumanEval, GSM8K, BBH-mini, "
            "synthetic_routing) through every baseline in --suite and emit "
            "comparison tables in --output. By default uses the deterministic "
            "MockLLMClient so the smoke run is CI-friendly; pass --backend yandex "
            "to run live against YandexGPT."
        ),
    )
    p.add_argument("--suite", default="full_min")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--backend", choices=("mock", "yandex"), default="mock")
    p.add_argument("--tasks-per-benchmark", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--budget-rub", type=float, default=300.0)
    p.add_argument("--parallelism", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("runs/benchmarks"))


def _add_report_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "report",
        help="Render the final research report (Phase 11).",
        description=(
            "Read a directory produced by `flybrain-py bench` (or "
            "`scripts/run_benchmarks.py`) and stitch the comparison tables "
            "and cherry-picked traces into a single Markdown report."
        ),
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing comparison_*.json (output of `bench`).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path. Defaults to <input>/report.md.",
    )
    p.add_argument(
        "--cherry-picks",
        nargs="*",
        type=Path,
        default=None,
        help="Optional paths to *.trace.json to feature in §4 of the report.",
    )


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
    _add_bench_subparser(sub)
    _add_report_subparser(sub)

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


def _cmd_bench(args: argparse.Namespace) -> int:
    # Lazy import: `bench` pulls in the LLM stack and torch-adjacent
    # modules; we don't want that on the import path for `flybrain-py info`.
    import asyncio

    from flybrain.benchmarks.cli import run as _run_bench

    return asyncio.run(_run_bench(args))


def _cmd_report(args: argparse.Namespace) -> int:
    from flybrain.eval import (
        AggregateMetrics,
        ReportInputs,
        write_report,
    )

    in_dir: Path = args.input
    overall_path = in_dir / "comparison_overall.json"
    if not overall_path.exists():
        print(f"missing {overall_path}; run `flybrain-py bench` first", file=sys.stderr)
        return 2
    overall_rows = [AggregateMetrics(**row) for row in json.loads(overall_path.read_text())]

    per_benchmark: dict[str, list[AggregateMetrics]] = {}
    for path in sorted(in_dir.glob("comparison_*.json")):
        if path.name == "comparison_overall.json":
            continue
        # comparison_<benchmark>.json
        name = path.stem[len("comparison_") :]
        per_benchmark[name] = [AggregateMetrics(**row) for row in json.loads(path.read_text())]

    out_path: Path = args.output or in_dir / "report.md"
    cherry = list(args.cherry_picks or [])
    write_report(
        ReportInputs(
            suite_name=in_dir.name,
            overall=overall_rows,
            per_benchmark=per_benchmark,
            trace_paths=cherry,
        ),
        out_path,
    )
    print(out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None or args.command == "info":
        return _cmd_info()
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "report":
        return _cmd_report(args)
    print(f"flybrain-py {args.command}: not yet implemented", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
