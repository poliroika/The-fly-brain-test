"""Python-side CLI parallel to the Rust `flybrain` binary.

Phase 0 wires only `--help` and `--version`; subcommands land alongside their
respective phases. The Rust CLI handles deterministic, native-only commands
(graph build / sim / etc.); the Python CLI is the entry point for anything
that needs PyTorch or the LLM.
"""

from __future__ import annotations

import argparse
import sys

from flybrain import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flybrain-py",
        description="FlyBrain Optimizer Python CLI",
    )
    parser.add_argument("--version", action="version", version=f"flybrain-py {__version__}")
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("info", help="Print build info and detected native module")
    sub.add_parser("sim", help="Run simulation pretraining (Phase 6; not yet implemented)")
    sub.add_parser("train", help="Run training (Phase 7/8; not yet implemented)")
    sub.add_parser("bench", help="Run benchmark suite (Phase 10; not yet implemented)")
    sub.add_parser("report", help="Build the final report (Phase 11; not yet implemented)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None or args.command == "info":
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
    print(f"flybrain-py {args.command}: not yet implemented in Phase 0", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
