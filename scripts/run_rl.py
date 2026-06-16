#!/usr/bin/env python3
"""Phase-8 RL/bandit training CLI (PLAN.md §597-601).

Examples::

    # Offline REINFORCE on Phase-7 expert traces.
    python scripts/run_rl.py reinforce \\
        --controller gnn \\
        --traces data/traces/v1 \\
        --warm-from runs/imitation/gnn.pt \\
        --epochs 5 \\
        --output runs/rl/gnn_reinforce.pt

    # PPO on the same offline batch.
    python scripts/run_rl.py ppo \\
        --controller gnn \\
        --traces data/traces/v1 \\
        --iterations 4 --epochs-per-batch 4 \\
        --output runs/rl/gnn_ppo.pt
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any


def _build_controller(name: str, agent_names: list[str]) -> Any:
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

    client = MockEmbeddingClient(output_dim=32)
    agents_emb = AgentEmbedder(client)
    asyncio.run(agents_emb.precompute(MINIMAL_15))
    builder = ControllerStateBuilder(
        task=TaskEmbedder(client),
        agents=agents_emb,
        trace=TraceEmbedder(client),
        fly=FlyGraphEmbedder(dim=8),
        agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
    )

    if name == "gnn":
        from flybrain.controller import FlyBrainGNNController

        cls: Any = FlyBrainGNNController
    elif name == "rnn":
        from flybrain.controller import FlyBrainRNNController

        cls = FlyBrainRNNController
    elif name == "router":
        from flybrain.controller import LearnedRouterController

        cls = LearnedRouterController
    else:
        raise ValueError(f"unknown controller {name!r}")

    return cls(
        builder=builder,
        task_dim=32,
        agent_dim=32,
        graph_dim=32,
        trace_dim=32 + 13,
        fly_dim=8,
        produced_dim=6,
        hidden_dim=32,
    )


def _maybe_load(controller: Any, path: Path | None) -> None:
    if path is None or not path.exists():
        return
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    controller.load_state_dict(sd, strict=False)
    print(f"[warm-start] loaded {path}")


def _save(controller: Any, sidecar: dict, output: Path) -> None:
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": controller.state_dict(), "sidecar": sidecar}, output)
    sidecar_path = output.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"[saved] {output}  +  {sidecar_path}")


def _cmd_reinforce(args: argparse.Namespace) -> None:
    from flybrain.agents.specs import MINIMAL_15
    from flybrain.training.expert_dataset import iter_traces
    from flybrain.training.rl import (
        ReinforceConfig,
        RewardConfig,
        reinforce_train,
    )

    agent_names = [s.name for s in MINIMAL_15]
    controller = _build_controller(args.controller, agent_names)
    _maybe_load(controller, args.warm_from)

    traces = list(iter_traces(Path(args.traces)))
    print(f"[reinforce] {len(traces)} traces, controller={args.controller}")
    cfg = ReinforceConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        only_passed=args.only_passed,
        reward=RewardConfig(),
    )
    res = reinforce_train(controller, traces=traces, agent_names=agent_names, config=cfg)
    sidecar = {
        "controller": args.controller,
        "algorithm": "reinforce",
        "traces": str(args.traces),
        "epochs": args.epochs,
        "epoch_returns": res.epoch_returns,
        "epoch_losses": res.epoch_losses,
        "epoch_entropy": res.epoch_entropy,
        "num_episodes": res.num_episodes,
    }
    _save(controller, sidecar, args.output)
    print(json.dumps(sidecar, indent=2))


def _cmd_ppo(args: argparse.Namespace) -> None:
    from flybrain.agents.specs import MINIMAL_15
    from flybrain.training.expert_dataset import iter_traces
    from flybrain.training.rl import PPOConfig, RewardConfig, ppo_train

    agent_names = [s.name for s in MINIMAL_15]
    controller = _build_controller(args.controller, agent_names)
    _maybe_load(controller, args.warm_from)
    traces = list(iter_traces(Path(args.traces)))
    print(f"[ppo] {len(traces)} traces, controller={args.controller}")
    cfg = PPOConfig(
        iterations=args.iterations,
        epochs_per_batch=args.epochs_per_batch,
        learning_rate=args.lr,
        seed=args.seed,
        only_passed=args.only_passed,
        reward=RewardConfig(),
    )
    res = ppo_train(controller, traces=traces, agent_names=agent_names, config=cfg)
    sidecar = {
        "controller": args.controller,
        "algorithm": "ppo",
        "traces": str(args.traces),
        "iterations": args.iterations,
        "iteration_returns": res.iteration_returns,
        "iteration_losses": res.iteration_losses,
        "clip_fraction": res.final_clip_fraction,
        "num_episodes": res.num_episodes,
    }
    _save(controller, sidecar, args.output)
    print(json.dumps(sidecar, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--controller", choices=("gnn", "rnn", "router"), default="gnn")
    common.add_argument("--traces", type=Path, required=True)
    common.add_argument("--warm-from", type=Path, default=None)
    common.add_argument("--lr", type=float, default=1e-4)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--only-passed", action="store_true")
    common.add_argument("--output", type=Path, required=True)

    rein = subs.add_parser("reinforce", parents=[common])
    rein.add_argument("--epochs", type=int, default=5)
    rein.set_defaults(func=_cmd_reinforce)

    ppo = subs.add_parser("ppo", parents=[common])
    ppo.add_argument("--iterations", type=int, default=4)
    ppo.add_argument("--epochs-per-batch", type=int, default=4)
    ppo.set_defaults(func=_cmd_ppo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
