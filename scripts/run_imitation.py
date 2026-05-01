#!/usr/bin/env python3
"""Phase-7 entrypoint — supervised cloning over expert traces.

Workflow::

    # 1. Collect expert traces (Phase 7.c).
    python scripts/collect_expert_traces.py \\
        --output data/traces/expert/v1 --backend yandex --tasks 100 --budget-rub 200

    # 2. Pretrain on synthetic data (Phase 6) — optional but recommended.
    python scripts/run_simulation_pretrain.py \\
        --controller gnn --epochs 30 --output runs/sim/gnn.pt

    # 3. Imitate the expert.
    python scripts/run_imitation.py \\
        --controller gnn \\
        --traces data/traces/expert/v1 \\
        --warm-from runs/sim/gnn.pt \\
        --epochs 8 --batch-size 16 \\
        --output runs/il/gnn.pt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from flybrain.agents.specs import MINIMAL_15
from flybrain.embeddings import (
    AgentEmbedder,
    AgentGraphEmbedder,
    ControllerStateBuilder,
    FlyGraphEmbedder,
    MockEmbeddingClient,
    TaskEmbedder,
    TraceEmbedder,
)
from flybrain.training import ImitationConfig, imitation_train

CTRL_CHOICES = ("gnn", "rnn", "router")


def _build_controller(name: str, builder: ControllerStateBuilder, hidden_dim: int):  # type: ignore[no-untyped-def]
    common = dict(
        builder=builder,
        task_dim=builder.task.client.dim,
        agent_dim=builder.agents.client.dim,
        graph_dim=builder.agent_graph.out_dim,
        trace_dim=builder.trace.client.dim + builder.trace.feature_dim,
        fly_dim=builder.fly.dim,
        produced_dim=len(builder.component_tags),
        hidden_dim=hidden_dim,
    )
    if name == "gnn":
        from flybrain.controller import FlyBrainGNNController

        return FlyBrainGNNController(**common)
    if name == "rnn":
        from flybrain.controller import FlyBrainRNNController

        return FlyBrainRNNController(**common)
    if name == "router":
        from flybrain.controller import LearnedRouterController

        return LearnedRouterController(**common)
    raise ValueError(f"unknown controller {name!r}, expected one of {CTRL_CHOICES}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CTRL_CHOICES, default="gnn")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument(
        "--warm-from",
        type=Path,
        default=None,
        help="Path to a sim-pretrain checkpoint to warm-start from.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-passed", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    client = MockEmbeddingClient(output_dim=64)
    agent_emb = AgentEmbedder(client)
    asyncio.run(agent_emb.precompute(MINIMAL_15))

    builder = ControllerStateBuilder(
        task=TaskEmbedder(client),
        agents=agent_emb,
        trace=TraceEmbedder(client),
        fly=FlyGraphEmbedder(dim=16),
        agent_graph=AgentGraphEmbedder(in_dim=64, hidden_dim=32, out_dim=64),
    )

    ctrl = _build_controller(args.controller, builder, hidden_dim=args.hidden_dim)
    if args.warm_from is not None:
        import torch

        ckpt = torch.load(args.warm_from, map_location="cpu", weights_only=False)
        ctrl.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[warm-start] loaded {args.warm_from}")

    cfg = ImitationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        only_passed=args.only_passed,
    )

    print(f"[imitation] controller={args.controller} cfg={cfg}")
    t0 = time.perf_counter()
    res = imitation_train(
        ctrl,
        traces_dir=args.traces,
        agent_names=[a.name for a in MINIMAL_15],
        config=cfg,
    )
    elapsed = time.perf_counter() - t0
    if res.num_examples == 0:
        print("[imitation] no examples found; expected *.trace.json files in traces dir.")
        return

    print(
        f"[imitation] examples={res.num_examples} "
        f"train={res.num_train} eval={res.num_eval} "
        f"loss[first/last]={res.losses[0]:.3f}/{res.losses[-1]:.3f} "
        f"final_acc={res.final_accuracy:.3f} wall={elapsed:.1f}s"
    )

    if args.output is not None:
        try:
            import torch
        except ImportError:  # pragma: no cover - script only
            print("[warn] torch not installed, skipping checkpoint save")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "controller": args.controller,
                    "state_dict": ctrl.state_dict(),
                    "config": cfg.__dict__,
                    "result": {
                        "num_examples": res.num_examples,
                        "epoch_accuracy": list(res.epoch_accuracy),
                        "final_accuracy": res.final_accuracy,
                    },
                },
                args.output,
            )
            sidecar = args.output.with_suffix(".json")
            sidecar.write_text(
                json.dumps(
                    {
                        "controller": args.controller,
                        "config": cfg.__dict__,
                        "wall_seconds": elapsed,
                        "epoch_accuracy": list(res.epoch_accuracy),
                        "final_accuracy": res.final_accuracy,
                    },
                    indent=2,
                )
            )
            print(f"[imitation] saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
