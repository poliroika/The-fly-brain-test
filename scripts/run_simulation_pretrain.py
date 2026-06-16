#!/usr/bin/env python3
"""Phase-6 entrypoint — supervised pretraining on synthetic tasks.

Usage::

    python scripts/run_simulation_pretrain.py \\
        --controller gnn --epochs 30 --n-per-type 64 --batch-size 64 \\
        --lr 3e-3 --hidden-dim 64 --output runs/sim_pretrain.pt

Reads no external data and never makes an LLM call. The exit
criterion (PLAN.md §590) is "controller сходится за <10 минут на CPU
и решает sim-задачи на ≥0.85 success" — the script logs both the
training loss and the held-out per-step accuracy after every epoch
so the operator can decide when to stop.
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
from flybrain.sim import SyntheticMAS, TaskGenerator
from flybrain.training import PretrainConfig, simulation_pretrain

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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-per-type", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--evaluate-on-sim",
        action="store_true",
        help="After training, run the controller through SyntheticMAS on a "
        "fresh batch and print mean success.",
    )
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
    cfg = PretrainConfig(
        n_per_type=args.n_per_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print(f"[pretrain] controller={args.controller} cfg={cfg}")
    t0 = time.perf_counter()
    res = simulation_pretrain(
        ctrl,
        agent_names=[a.name for a in MINIMAL_15],
        config=cfg,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[pretrain] done examples={res.num_examples} "
        f"loss[first/last]={res.losses[0]:.3f}/{res.losses[-1]:.3f} "
        f"final_acc={res.final_accuracy:.3f} wall={elapsed:.1f}s"
    )

    if args.evaluate_on_sim:
        sim = SyntheticMAS(
            agent_names=[a.name for a in MINIMAL_15],
            seed=args.seed,
            max_steps=12,
        )
        gen = TaskGenerator(seed=args.seed + 1)
        tasks = gen.balanced_dataset(n_per_type=8)
        outcomes = [sim.run(ctrl, t) for t in tasks]
        success_rate = sum(o.success for o in outcomes) / len(outcomes)
        mean_score = sum(o.final_score for o in outcomes) / len(outcomes)
        print(
            f"[eval] tasks={len(tasks)} success_rate={success_rate:.3f} "
            f"mean_final_score={mean_score:.3f}"
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
            print(f"[pretrain] saved checkpoint to {args.output}")
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


if __name__ == "__main__":
    main()
